"""Tool execution utilities for automatic function calling in LLM providers."""

import inspect
import json
import re
import signal
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Optional, get_args, get_origin

from llmtools.utils.logging import setup_logger


class ToolExecutionError(Exception):
    """Custom exception for tool execution errors."""

    pass


class ToolTimeoutError(ToolExecutionError):
    """Exception raised when tool execution times out."""

    pass


def execute_tool_function(
    function: Callable[..., Any],
    arguments: dict[str, Any],
    timeout: Optional[float] = None,
) -> Any:
    """Execute a tool function with the given arguments.

    Args:
        function: The Python function to execute
        arguments: Dictionary of arguments to pass to the function
        timeout: Optional timeout in seconds

    Returns:
        The result of the function execution

    Raises:
        ToolExecutionError: If function execution fails
        ToolTimeoutError: If function execution times out
    """
    logger = setup_logger(__name__)

    try:
        # Validate function signature
        sig = inspect.signature(function)

        # Filter arguments to match function parameters
        filtered_args = {}
        for param_name, param in sig.parameters.items():
            if param_name in arguments:
                filtered_args[param_name] = arguments[param_name]
            elif param.default is param.empty and param.kind not in (
                param.VAR_POSITIONAL,
                param.VAR_KEYWORD,
            ):
                raise ToolExecutionError(
                    f"Missing required parameter '{param_name}' for function '{function.__name__}'"
                )

        logger.debug(
            f"Executing function '{function.__name__}' with args: {filtered_args}"
        )

        # Execute with optional timeout
        if timeout:
            with _timeout(timeout):
                result = function(**filtered_args)
        else:
            result = function(**filtered_args)

        logger.debug(f"Function '{function.__name__}' executed successfully")
        return result

    except ToolTimeoutError:
        logger.error(f"Function '{function.__name__}' timed out after {timeout}s")
        raise
    except Exception as e:
        logger.error(f"Error executing function '{function.__name__}': {e}")
        raise ToolExecutionError(
            f"Error executing function '{function.__name__}': {str(e)}"
        ) from e


def parse_tool_arguments(arguments_json: str) -> dict[str, Any]:
    """Parse tool arguments from JSON string.

    Args:
        arguments_json: JSON string containing function arguments

    Returns:
        Dictionary of parsed arguments

    Raises:
        ToolExecutionError: If JSON parsing fails
    """
    try:
        result = json.loads(arguments_json)
        if not isinstance(result, dict):
            raise ToolExecutionError("Tool arguments must be a JSON object")
        return result
    except json.JSONDecodeError as e:
        raise ToolExecutionError(f"Invalid JSON in tool arguments: {e}") from e


def format_tool_result(result: Any) -> str:
    """Format tool execution result for LLM consumption.

    Args:
        result: The result from tool execution

    Returns:
        String representation of the result suitable for LLM
    """
    if result is None:
        return "Function executed successfully with no return value"

    if isinstance(result, (str, int, float, bool)):
        return str(result)

    if isinstance(result, (dict, list)):
        try:
            return json.dumps(result, indent=2)
        except (TypeError, ValueError):
            return str(result)

    return str(result)


def create_tool_error_message(
    function_name: str, error: Exception, handle_gracefully: bool = True
) -> str:
    """Create an error message for tool execution failures.

    Args:
        function_name: Name of the function that failed
        error: The exception that occurred
        handle_gracefully: Whether to create a user-friendly message

    Returns:
        Error message string for LLM
    """
    if handle_gracefully:
        if isinstance(error, ToolTimeoutError):
            return f"Function '{function_name}' timed out during execution. Please try a simpler approach."
        elif isinstance(error, ToolExecutionError):
            return f"Error calling function '{function_name}': {str(error)}"
        else:
            return f"Function '{function_name}' encountered an unexpected error. Please try a different approach."
    else:
        return f"Tool execution failed for '{function_name}': {str(error)}"


def create_function_not_found_message(function_name: str) -> str:
    """Create an error message when a requested function is not available.

    Args:
        function_name: Name of the requested function

    Returns:
        Error message string for LLM
    """
    return (
        f"Function '{function_name}' is not available. "
        f"Please use only the functions that were provided in the tool list."
    )


def validate_tool_functions(
    tools: list[dict[str, Any]], tool_functions: dict[str, Callable[..., Any]]
) -> list[str]:
    """Validate that tool functions match the provided tool schemas.

    Args:
        tools: List of tool schemas
        tool_functions: Dictionary of function name to callable mapping

    Returns:
        List of validation warning messages (empty if all valid)
    """
    warnings = []

    # Get function names from tools
    tool_names = set()
    for tool in tools:
        if "function" in tool and "name" in tool["function"]:
            tool_names.add(tool["function"]["name"])

    # Check for missing functions
    missing_functions = tool_names - set(tool_functions.keys())
    if missing_functions:
        warnings.append(f"Tool functions missing for: {', '.join(missing_functions)}")

    # Check for extra functions (not necessarily a problem, but worth noting)
    extra_functions = set(tool_functions.keys()) - tool_names
    if extra_functions:
        warnings.append(
            f"Extra tool functions provided (not in schema): {', '.join(extra_functions)}"
        )

    return warnings


def convert_functions_to_map(
    functions: list[Callable[..., Any]],
) -> dict[Callable[..., Any], dict[str, Any]]:
    """Convert a list of Python functions to OpenAI function calling format.

    Args:
        functions: List of Python functions to convert

    Returns:
        Dictionary mapping actual function objects to their OpenAI function schemas

    Raises:
        ValueError: If function analysis fails
    """
    function_map = {}

    for func in functions:
        try:
            # Get function name
            func_name = func.__name__

            # Get function signature
            sig = inspect.signature(func)

            # Parse docstring for description and parameter descriptions
            doc = inspect.getdoc(func) or ""
            func_description = _extract_function_description(doc)
            param_descriptions = _extract_parameter_descriptions(doc)

            # Build parameters schema
            parameters: dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }

            for param_name, param in sig.parameters.items():
                # Skip *args and **kwargs
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue

                # Get parameter type info (now returns dict with type and optional enum)
                param_type_info = _get_parameter_type(param.annotation)

                # Build parameter schema starting with type info
                param_schema = param_type_info.copy()

                # Add description if available
                if param_name in param_descriptions:
                    param_schema["description"] = param_descriptions[param_name]

                parameters["properties"][param_name] = param_schema

                # Add to required if no default value
                if param.default is param.empty:
                    parameters["required"].append(param_name)

            # Build function schema
            function_schema = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_description,
                    "parameters": parameters,
                },
            }

            # Map the actual function to its schema
            function_map[func] = function_schema

        except Exception as e:
            raise ValueError(
                f"Failed to convert function '{func.__name__}': {str(e)}"
            ) from e

    return function_map


def _extract_function_description(docstring: str) -> str:
    """Extract main function description from docstring."""
    if not docstring:
        return "No description available"

    # Split by first empty line or Args: section
    lines = docstring.strip().split("\n")
    description_lines = []

    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("args:"):
            break
        description_lines.append(line)

    return " ".join(description_lines) or "No description available"


def _extract_parameter_descriptions(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from Args: section of docstring."""
    if not docstring:
        return {}

    param_descriptions: dict[str, str] = {}

    # Find Args: section
    args_match = re.search(
        r"Args:\s*\n(.*?)(?:\n\s*(?:Returns?|Raises?|Yields?|Note):|$)",
        docstring,
        re.DOTALL | re.IGNORECASE,
    )

    if not args_match:
        return param_descriptions

    args_section = args_match.group(1)

    # Parse parameter descriptions
    for line in args_section.split("\n"):
        line = line.strip()
        if ":" in line:
            param_match = re.match(r"(\w+):\s*(.*)", line)
            if param_match:
                param_name, description = param_match.groups()
                param_descriptions[param_name] = description.strip()

    return param_descriptions


def _get_parameter_type(annotation: Any) -> dict[str, Any]:
    """Convert Python type annotation to JSON schema type info with enum support."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}  # Default to string if no annotation

    # Handle Enum classes
    if inspect.isclass(annotation) and issubclass(annotation, Enum):
        # Extract enum values
        enum_values = [member.value for member in annotation]
        # Determine the base type from the first enum value
        if enum_values:
            first_value = enum_values[0]
            if isinstance(first_value, str):
                return {"type": "string", "enum": enum_values}
            elif isinstance(first_value, int):
                return {"type": "integer", "enum": enum_values}
            elif isinstance(first_value, float):
                return {"type": "number", "enum": enum_values}
        return {"type": "string", "enum": enum_values}

    # Handle Literal types (e.g., Literal["celsius", "fahrenheit"])
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is not None:
        # Handle Literal types
        if hasattr(origin, "__name__") and origin.__name__ == "Literal":
            # Extract literal values
            enum_values = list(args)
            # Determine type from first value
            if enum_values:
                first_value = enum_values[0]
                if isinstance(first_value, str):
                    return {"type": "string", "enum": enum_values}
                elif isinstance(first_value, int):
                    return {"type": "integer", "enum": enum_values}
                elif isinstance(first_value, float):
                    return {"type": "number", "enum": enum_values}
            return {"type": "string", "enum": enum_values}

        # Handle Union types (including Optional and Union of Literals)
        if (
            origin is type(None)
            or (hasattr(origin, "__name__") and origin.__name__ == "UnionType")
            or origin.__name__ == "Union"
        ):
            # For Union types, look for Literal types or use first non-None type
            enum_values = []
            non_none_type = None

            for arg in args:
                if arg is type(None):
                    continue
                elif (
                    get_origin(arg) is not None
                    and hasattr(get_origin(arg), "__name__")
                    and get_origin(arg).__name__ == "Literal"
                ):
                    # Collect literal values from Union of Literals
                    enum_values.extend(get_args(arg))
                elif non_none_type is None:
                    non_none_type = arg

            if enum_values:
                # We found literals in the Union
                if isinstance(enum_values[0], str):
                    return {"type": "string", "enum": enum_values}
                elif isinstance(enum_values[0], int):
                    return {"type": "integer", "enum": enum_values}
                elif isinstance(enum_values[0], float):
                    return {"type": "number", "enum": enum_values}
                return {"type": "string", "enum": enum_values}
            elif non_none_type is not None:
                # Recursively handle the non-None type
                return _get_parameter_type(non_none_type)

            return {"type": "string"}

        # Handle container types
        if origin is list:
            return {"type": "array"}
        elif origin is dict:
            return {"type": "object"}

    # Handle common basic types
    if annotation is str:
        return {"type": "string"}
    elif annotation is int:
        return {"type": "integer"}
    elif annotation is float:
        return {"type": "number"}
    elif annotation is bool:
        return {"type": "boolean"}
    elif annotation is list:
        return {"type": "array"}
    elif annotation is dict:
        return {"type": "object"}

    # Default to string for unknown types
    return {"type": "string"}


@contextmanager
def _timeout(seconds: float) -> Generator[None, None, None]:
    """Context manager for function execution timeout.

    Args:
        seconds: Timeout in seconds

    Raises:
        ToolTimeoutError: If timeout is exceeded
    """

    def timeout_handler(signum: int, frame: Any) -> None:
        raise ToolTimeoutError(f"Function execution timed out after {seconds} seconds")

    # Set up the timeout
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        # Clean up
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

"""Tool execution utilities for automatic function calling in LLM providers."""

import inspect
import json
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


def _convert_argument_to_type(value: Any, param_annotation: Any) -> Any:
    """Convert an argument value to the expected type annotation.

    Args:
        value: The argument value to convert (typically a string from LLM)
        param_annotation: The expected type annotation

    Returns:
        The converted value or the original value if conversion not needed

    Raises:
        ToolExecutionError: If conversion fails
    """
    # If no annotation or value is None, return as-is
    if param_annotation is inspect.Parameter.empty or value is None:
        return value

    # Handle Enum classes
    if inspect.isclass(param_annotation) and issubclass(param_annotation, Enum):
        if isinstance(value, param_annotation):
            # Already the right enum type
            return value
        # Convert string value to enum
        try:
            return param_annotation(value)
        except ValueError as e:
            # Try to find enum by name if value doesn't work
            try:
                return getattr(
                    param_annotation,
                    value.upper() if isinstance(value, str) else str(value),
                )
            except (AttributeError, ValueError):
                valid_values = [member.value for member in param_annotation]
                raise ToolExecutionError(
                    f"Invalid enum value '{value}' for {param_annotation.__name__}. "
                    f"Valid values are: {valid_values}"
                ) from e

    # Handle Optional/Union types (including Optional[Enum])
    origin = get_origin(param_annotation)
    args = get_args(param_annotation)

    if origin is not None:
        # Handle Union types (including Optional)
        if (
            origin is type(None)
            or (hasattr(origin, "__name__") and origin.__name__ == "UnionType")
            or getattr(origin, "__name__", None) == "Union"
        ):
            # For Union types, try to convert to the first non-None type that's an Enum
            for arg in args:
                if arg is type(None):
                    continue
                if inspect.isclass(arg) and issubclass(arg, Enum):
                    try:
                        return _convert_argument_to_type(value, arg)
                    except ToolExecutionError:
                        continue
            # If no enum type found, return the value as-is
            return value

    # For all other types, return the value as-is
    # (basic types like str, int, etc. should already be correct from JSON parsing)
    return value


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

        # Filter and convert arguments to match function parameters
        filtered_args = {}
        for param_name, param in sig.parameters.items():
            if param_name in arguments:
                # Convert argument to the expected type (especially enums)
                converted_value = _convert_argument_to_type(
                    arguments[param_name], param.annotation
                )
                filtered_args[param_name] = converted_value
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

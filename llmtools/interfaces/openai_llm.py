"""OpenAI LLM provider implementation using OpenAI SDK."""

import inspect
import json
import logging
import os
import re
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable, Iterator, Optional, cast, get_args, get_origin

from dotenv import load_dotenv

from llmtools.interfaces.llm import LLMInterface, T
from llmtools.utils.logger_config import setup_logger
from llmtools.utils.tools import (
    ToolExecutionError,
    create_function_not_found_message,
    create_tool_error_message,
    execute_tool_function,
    format_tool_result,
    parse_tool_arguments,
    validate_tool_functions,
)

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "OpenAI SDK is required for OpenAI provider. "
        "Install with: pip install 'llmtools[openai]' or pip install openai"
    ) from e


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

                # Add to required if no default value AND not Optional
                if param.default is param.empty and not _is_optional_type(param.annotation):
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

            # Debug output
            logging.debug(f"\n=== Schema for {func_name} ===\n{json.dumps(function_schema, indent=2)}\n")

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


def _is_optional_type(annotation: Any) -> bool:
    """Check if a type annotation represents an Optional type."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    if origin is None:
        return False
    
    # Check for Union types (including Optional)
    if (
        origin is type(None)
        or (hasattr(origin, "__name__") and origin.__name__ == "UnionType")
        or origin.__name__ == "Union"
    ):
        # Optional[T] is Union[T, None] - check if None is one of the args
        return type(None) in args
    
    return False


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
            # Check if we have type arguments (e.g., List[int])
            if args:
                # Get the inner type and create items schema
                inner_type = args[0]
                items_schema = _get_parameter_type(inner_type)
                return {"type": "array", "items": items_schema}
            else:
                # Bare list without type arguments
                return {
                    "type": "array",
                    "items": {"type": "string"},
                }  # Default to string items
        elif origin is tuple:
            # Handle tuple types like tuple[int, int]
            if args:
                # Check if all tuple elements are the same type
                first_arg_schema = _get_parameter_type(args[0])
                all_same = all(_get_parameter_type(arg) == first_arg_schema for arg in args)
                
                if all_same:
                    # Homogeneous tuple - use single items schema with length constraints
                    return {
                        "type": "array",
                        "minItems": len(args),
                        "maxItems": len(args),
                        "items": first_arg_schema
                    }
                else:
                    # Heterogeneous tuple - fallback to generic array
                    # OpenAI doesn't support per-position item schemas
                    return {
                        "type": "array",
                        "minItems": len(args),
                        "maxItems": len(args),
                        "items": {"type": "string"}  # Default to string for mixed types
                    }
            else:
                # Bare tuple without type arguments
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
        return {"type": "array", "items": {"type": "string"}}  # Default to string items
    elif annotation is dict:
        return {"type": "object"}

    # Default to string for unknown types
    return {"type": "string"}


class OpenAIProvider(LLMInterface):
    """OpenAI LLM provider using OpenAI SDK with configurable base URL."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 1,
        max_tokens: Optional[int] = None,
        timeout: int = 60,
        max_tool_iterations: int = 20,
        tool_timeout: Optional[float] = None,
        handle_tool_errors: bool = True,
        tool_choice: str = "required",
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = None,
        **client_kwargs: Any,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If not provided, will try to load from
                    OPENAI_API_KEY environment variable
            model: Model to use (default: gpt-4o-mini)
            base_url: Custom base URL (e.g., for OpenRouter)
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            max_tool_iterations: Maximum number of tool calling rounds
            tool_timeout: Timeout for individual tool execution
            handle_tool_errors: Whether to handle tool execution errors gracefully
            tool_choice: Tool choice strategy ('required', 'auto', etc.)
            reasoning_effort: Effort level for reasoning models (minimal, low, medium, high)
            reasoning_summary: Reasoning summary level (auto, concise, detailed)
            **client_kwargs: Additional arguments to pass to OpenAI client
        """
        # Load environment variables from .env file
        load_dotenv()

        # Handle API credentials with environment variable fallback
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter"
            )

        self.model = model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        # Store configuration
        self.config: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "max_tool_iterations": max_tool_iterations,
            "tool_timeout": tool_timeout,
            "handle_tool_errors": handle_tool_errors,
            "tool_choice": tool_choice,
            "reasoning_effort": reasoning_effort,
            "reasoning_summary": reasoning_summary,
        }

        # Initialize OpenAI client (only pass valid OpenAI client parameters)
        client_params = {
            "api_key": self.api_key,
            **client_kwargs,
        }
        if self.base_url:
            client_params["base_url"] = self.base_url

        self.client = OpenAI(**client_params)

        # Set up logging
        self.logger = setup_logger(__name__)
        self.logger.info(f"OpenAI provider initialized with model: {self.model}")
        if self.base_url:
            self.logger.info(f"Using custom base URL: {self.base_url}")
        self.logger.debug(f"Configuration: {self.config}")

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
    ) -> list[dict[str, str]]:
        """Build messages array for OpenAI chat format.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            history: Optional conversation history

        Returns:
            List of message dictionaries in OpenAI format
        """
        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add history messages if provided
        if history:
            messages.extend(history)

        # Add current user prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text response from the LLM.

        Args:
            prompt: The user prompt/input text
            system_prompt: Optional system prompt to guide behavior
            history: Optional conversation history as list of {"role": str, "content": str}
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response from the LLM
        """
        self.logger.info(f"Generating text response with model: {self.model}")
        self.logger.debug(f"Prompt length: {len(prompt)} characters")
        if system_prompt:
            self.logger.debug(f"System prompt length: {len(system_prompt)} characters")
        if history:
            self.logger.debug(f"History: {len(history)} messages")

        messages = self._build_messages(prompt, system_prompt, history)

        # Build request parameters with proper precedence
        request_params = {
            "model": self.model,
            "messages": messages,
        }
        
        # Add configuration parameters (skip client-specific and non-API params)
        config_params = {k: v for k, v in self.config.items() 
                        if k in ["temperature", "max_tokens", "timeout"] and v is not None}
        request_params.update(config_params)
        
        # Add reasoning parameters for supported models
        if self.config.get("reasoning_effort") or self.config.get("reasoning_summary"):
            reasoning = {}
            if self.config.get("reasoning_effort"):
                reasoning["effort"] = self.config["reasoning_effort"]
            if self.config.get("reasoning_summary"):
                reasoning["summary"] = self.config["reasoning_summary"]
            if reasoning:
                request_params["reasoning"] = reasoning
        
        # Method-level overrides have highest precedence
        request_params.update(kwargs)
        self.logger.debug(
            f"Request params: {
                {k: v for k, v in request_params.items() if k != 'messages'}
            }"
        )

        try:
            response = self.client.chat.completions.create(**request_params)
            content = response.choices[0].message.content or ""

            # Log token usage if available
            if hasattr(response, "usage") and response.usage:
                self.logger.info(
                    f"Token usage - prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}, total: {response.usage.total_tokens}"
                )

            self.logger.info(f"Generated response: {len(content)} characters")
            return content
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate structured output conforming to a JSON schema.

        Args:
            prompt: The user prompt/input text
            schema: JSON schema the response must conform to
            system_prompt: Optional system prompt to guide behavior
            history: Optional conversation history as list of {"role": str, "content": str}
            **kwargs: Additional provider-specific parameters

        Returns:
            Structured response as a dictionary conforming to schema
        """
        messages = self._build_messages(prompt, system_prompt, history)

        # Configure structured output using OpenAI's format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": True,
                "schema": schema,
            },
        }

        # Build request parameters with proper precedence
        request_params = {
            "model": self.model,
            "messages": messages,
            "response_format": response_format,
        }
        
        # Add configuration parameters (skip client-specific and non-API params)
        config_params = {k: v for k, v in self.config.items() 
                        if k in ["temperature", "max_tokens", "timeout"] and v is not None}
        request_params.update(config_params)
        
        # Add reasoning parameters for supported models
        if self.config.get("reasoning_effort") or self.config.get("reasoning_summary"):
            reasoning = {}
            if self.config.get("reasoning_effort"):
                reasoning["effort"] = self.config["reasoning_effort"]
            if self.config.get("reasoning_summary"):
                reasoning["summary"] = self.config["reasoning_summary"]
            if reasoning:
                request_params["reasoning"] = reasoning
        
        # Method-level overrides have highest precedence
        request_params.update(kwargs)

        try:
            self.logger.info(f"Generating structured response with model: {self.model}")
            self.logger.debug("Using JSON schema for structured output")

            response = self.client.chat.completions.create(**request_params)
            content = response.choices[0].message.content or "{}"

            # Log token usage if available
            if hasattr(response, "usage") and response.usage:
                self.logger.info(
                    f"Token usage - prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}, total: {response.usage.total_tokens}"
                )

            parsed_result = json.loads(content)
            if isinstance(parsed_result, dict):
                self.logger.info("Successfully generated structured response")
                self.logger.debug(f"Response keys: {list(parsed_result.keys())}")
                return parsed_result
            raise ValueError("Expected JSON object, got other type")
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse structured response as JSON: {e}")
            raise ValueError(f"Failed to parse structured response as JSON: {e}") from e
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def generate_model(
        self,
        prompt: str,
        model_class: type[T],
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        **kwargs: Any,
    ) -> T:
        """Generate structured output using a Pydantic model.

        Args:
            prompt: The user prompt/input text
            model_class: Pydantic model class to structure the response
            system_prompt: Optional system prompt to guide behavior
            history: Optional conversation history as list of {"role": str, "content": str}
            **kwargs: Additional provider-specific parameters

        Returns:
            Instance of the specified Pydantic model with validated data
        """
        self.logger.info(
            f"Generating Pydantic model response with model: {self.model}, target: {model_class.__name__}"
        )
        self.logger.debug(f"Target model class: {model_class.__name__}")
        self.logger.debug(f"Prompt length: {len(prompt)} characters")
        if system_prompt:
            self.logger.debug(f"System prompt length: {len(system_prompt)} characters")
        if history:
            self.logger.debug(f"History: {len(history)} messages")

        messages = self._build_messages(prompt, system_prompt, history)

        # Build request parameters with proper precedence
        request_params = {
            "model": self.model,
            "messages": messages,
            "response_format": model_class,
        }
        
        # Add configuration parameters (skip client-specific and non-API params)
        config_params = {k: v for k, v in self.config.items() 
                        if k in ["temperature", "max_tokens", "timeout"] and v is not None}
        request_params.update(config_params)
        
        # Add reasoning parameters for supported models
        if self.config.get("reasoning_effort") or self.config.get("reasoning_summary"):
            reasoning = {}
            if self.config.get("reasoning_effort"):
                reasoning["effort"] = self.config["reasoning_effort"]
            if self.config.get("reasoning_summary"):
                reasoning["summary"] = self.config["reasoning_summary"]
            if reasoning:
                request_params["reasoning"] = reasoning
        
        # Method-level overrides have highest precedence
        request_params.update(kwargs)
        self.logger.debug(
            f"Request params: {
                {k: v for k, v in request_params.items() if k != 'messages'}
            }"
        )

        try:
            response = self.client.chat.completions.parse(**request_params)

            # Log token usage if available
            if hasattr(response, "usage") and response.usage:
                self.logger.info(
                    f"Token usage - prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}, total: {response.usage.total_tokens}"
                )

            parsed_output = response.choices[0].message.parsed
            if parsed_output is not None:
                self.logger.info(
                    f"Successfully generated {model_class.__name__} instance"
                )
                # Cast to T since OpenAI's parse returns the correct type
                return parsed_output  # type: ignore[no-any-return]
            else:
                self.logger.error("No parsed output returned from OpenAI")
                raise ValueError("No parsed output returned from OpenAI")

        except ValueError:
            # Re-raise ValueError directly without wrapping
            raise
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def generate_with_tools(
        self,
        prompt: str,
        functions: Optional[list[Callable[..., Any]]] = None,
        function_map: Optional[dict[Callable[..., Any], dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        max_tool_iterations: Optional[int] = None,
        handle_tool_errors: Optional[bool] = None,
        tool_timeout: Optional[float] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate response with access to function/tool calling. Either functions or function_map (or both) must be provided.

        Args:
            prompt: The user prompt/input text
            functions: Optional list of Python functions to auto-convert to tools.
                      Function schemas are generated from type hints and docstrings.
            function_map: Optional dict mapping Python functions to their schema definitions.
            system_prompt: Optional system prompt to guide behavior
            history: Optional conversation history as list of {"role": str, "content": str}
            max_tool_iterations: Maximum number of tool calling rounds (uses provider config if None)
            handle_tool_errors: Whether to handle tool execution errors gracefully (uses provider config if None)
            tool_timeout: Timeout in seconds for individual tool execution (uses provider config if None)
            tool_choice: Tool choice strategy (uses provider config if None)
            **kwargs: Additional provider-specific parameters

        Returns:
            Final text response after all tool calls are completed
        """
        # Use provider configuration as defaults when parameters are None
        max_tool_iterations = max_tool_iterations if max_tool_iterations is not None else self.config.get("max_tool_iterations", 20)
        handle_tool_errors = handle_tool_errors if handle_tool_errors is not None else self.config.get("handle_tool_errors", True)
        tool_timeout = tool_timeout if tool_timeout is not None else self.config.get("tool_timeout")
        tool_choice = tool_choice if tool_choice is not None else self.config.get("tool_choice", "required")
        
        # Validate that at least one tool source is provided
        if not functions and not function_map:
            raise ValueError("Either 'functions' or 'function_map' must be provided")

        # Combine auto-generated and manual function maps
        combined_function_map = {}

        # Handle functions parameter (auto-convert)
        if functions:
            auto_generated_map = convert_functions_to_map(functions)
            combined_function_map.update(auto_generated_map)

        # Handle function_map parameter (manual schemas)
        if function_map:
            combined_function_map.update(function_map)

        # Convert combined function_map to tools and tool_functions format
        tools, tool_functions = self._convert_function_map(combined_function_map)

        # Validate tool functions match tool schemas
        validation_warnings = validate_tool_functions(tools, tool_functions)
        for warning in validation_warnings:
            self.logger.warning(warning)

        # Start automatic tool execution loop
        return self._execute_tools_automatically(
            prompt,
            tools,
            tool_functions,
            system_prompt,
            history,
            max_tool_iterations,
            handle_tool_errors,
            tool_timeout,
            tool_choice,
            **kwargs,
        )

    def _convert_function_map(
        self, function_map: dict[Callable[..., Any], dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
        """Convert function_map to tools and tool_functions format.

        Args:
            function_map: Dict mapping functions to their schema definitions

        Returns:
            Tuple of (tools list, tool_functions dict) for internal use
        """
        tools = []
        tool_functions = {}

        for func, definition in function_map.items():
            function_name = func.__name__

            # Check if definition is already in full OpenAI format or simplified format
            if (
                "type" in definition
                and definition["type"] == "function"
                and "function" in definition
            ):
                # Already in full OpenAI format (from convert_functions_to_map)
                tool = definition
            else:
                # Simplified format - convert to full JSON schema format
                converted_definition = self._convert_simplified_schema(definition)
                # Build OpenAI tool format
                tool = {
                    "type": "function",
                    "function": {"name": function_name, **converted_definition},
                }

            tools.append(tool)
            tool_functions[function_name] = func

        self.logger.debug(f"Converted {len(function_map)} functions to tools format")
        return tools, tool_functions

    def _convert_simplified_schema(self, definition: dict[str, Any]) -> dict[str, Any]:
        """Convert simplified schema format to full JSON schema format.

        Args:
            definition: Simplified schema with parameters as list

        Returns:
            Full JSON schema format for OpenAI
        """
        converted = {}

        # Copy description if present
        if "description" in definition:
            converted["description"] = definition["description"]

        # Convert parameters list to JSON schema format
        if "parameters" in definition and isinstance(definition["parameters"], list):
            properties = {}

            for param in definition["parameters"]:
                param_name = param["name"]
                param_schema = {"type": param["type"]}

                if "description" in param:
                    param_schema["description"] = param["description"]
                if "enum" in param:
                    param_schema["enum"] = param["enum"]

                properties[param_name] = param_schema

            # Build full parameters schema
            parameters_schema: dict[str, Any] = {
                "type": "object",
                "properties": properties,
            }

            # Add required fields if specified
            if "required" in definition:
                parameters_schema["required"] = definition["required"]

            # Add strict mode if specified
            if definition.get("strict", False):
                parameters_schema["additionalProperties"] = False

            converted["parameters"] = parameters_schema

        return converted

    def _execute_tools_automatically(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        tool_functions: dict[str, Callable[..., Any]],
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        max_tool_iterations: int = 20,
        handle_tool_errors: bool = True,
        tool_timeout: Optional[float] = None,
        tool_choice: str = "required",
        **kwargs: Any,
    ) -> str:
        """Execute tools automatically and return final text response.

        Handles the complete tool calling loop until LLM returns a text response.
        """
        messages = self._build_messages(prompt, system_prompt, history)
        iteration = 0
        total_tokens_used = 0

        self.logger.info(f"Starting automatic tool execution loop, model: {self.model}")
        self.logger.debug(f"Max iterations: {max_tool_iterations}")
        self.logger.debug(f"Available functions: {list(tool_functions.keys())}")

        # Build request parameters (tool_choice will be set dynamically in the loop)
        request_params = {
            "model": self.model,
            "tools": tools,
        }
        
        # Add configuration parameters (skip tool-specific and non-API params)
        config_params = {k: v for k, v in self.config.items() 
                        if k in ["temperature", "max_tokens", "timeout"] and v is not None}
        request_params.update(config_params)
        
        # Add reasoning parameters for supported models
        if self.config.get("reasoning_effort") or self.config.get("reasoning_summary"):
            reasoning = {}
            if self.config.get("reasoning_effort"):
                reasoning["effort"] = self.config["reasoning_effort"]
            if self.config.get("reasoning_summary"):
                reasoning["summary"] = self.config["reasoning_summary"]
            if reasoning:
                request_params["reasoning"] = reasoning
        
        # Method-level overrides
        request_params.update(kwargs)

        while iteration < max_tool_iterations:
            iteration += 1
            self.logger.debug(
                f"Tool execution iteration {iteration}/{max_tool_iterations}"
            )

            # Dynamic tool_choice: required for first iteration, auto for subsequent
            current_tool_choice = tool_choice if iteration == 1 else "auto"
            current_request_params = {
                **request_params,
                "tool_choice": current_tool_choice,
            }
            
            self.logger.debug(f"Using tool_choice: {current_tool_choice}")

            try:
                # Make API call
                response = self.client.chat.completions.create(
                    messages=cast(Any, messages), **current_request_params
                )
                message = response.choices[0].message
                if message.content:
                    # log the message
                    self.logger.info(f"LLM response: {str(message.content)[:500]}...")

                # Log token usage
                if hasattr(response, "usage") and response.usage:
                    iteration_tokens = response.usage.total_tokens
                    total_tokens_used += iteration_tokens
                    self.logger.debug(
                        f"Iteration {iteration} token usage: {iteration_tokens} (total: {total_tokens_used})"
                    )

                # Check if model wants to call tools
                if not message.tool_calls:
                    # Final text response
                    final_content = message.content or ""
                    self.logger.info(
                        f"Tool execution completed after {iteration} iterations, "
                        f"total tokens: {total_tokens_used}, response length: {len(final_content)} chars"
                    )
                    return final_content

                # Execute tool calls
                self.logger.info(f"Executing {len(message.tool_calls)} tool call(s)")

                # Add assistant message with tool calls to conversation
                messages.append(
                    cast(
                        Any,
                        {
                            "role": "assistant",
                            "content": message.content or "",
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": getattr(tc, "function", tc).name
                                        if hasattr(tc, "function")
                                        else tc.name,
                                        "arguments": getattr(
                                            tc, "function", tc
                                        ).arguments
                                        if hasattr(tc, "function")
                                        else tc.arguments,
                                    },
                                }
                                for tc in (message.tool_calls or [])
                            ],
                        },
                    )
                )

                # Execute each tool call
                for tool_call in message.tool_calls or []:
                    # Handle both function tool calls and custom tool calls
                    if hasattr(tool_call, "function"):
                        function_name = tool_call.function.name
                        function_args = tool_call.function.arguments
                    else:
                        # Custom tool call type
                        function_name = getattr(tool_call, "name", "unknown")
                        function_args = getattr(tool_call, "arguments", "{}")

                    tool_call_id = tool_call.id

                    try:
                        # Check if function exists
                        if function_name not in tool_functions:
                            error_msg = create_function_not_found_message(function_name)
                            self.logger.warning(f"Function not found: {function_name}")
                        else:
                            # Parse arguments and execute function
                            arguments = parse_tool_arguments(function_args)
                            function = tool_functions[function_name]

                            self.logger.info(
                                f"Executing function: {function_name} with arguments: {arguments}"
                            )

                            result = execute_tool_function(
                                function, arguments, timeout=tool_timeout
                            )

                            # Format result for LLM
                            formatted_result = format_tool_result(result)
                            self.logger.debug(
                                f"Function {function_name} executed successfully, result length: {len(formatted_result)} chars"
                            )

                            # Add tool result to conversation
                            messages.append(
                                cast(
                                    Any,
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": formatted_result,
                                    },
                                )
                            )
                            continue

                    except (ToolExecutionError, json.JSONDecodeError) as e:
                        error_msg = create_tool_error_message(
                            function_name, e, handle_tool_errors
                        )
                        self.logger.error(
                            f"Tool execution error for {function_name}: {e}"
                        )

                    # Add error message to conversation
                    messages.append(
                        cast(
                            Any,
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": error_msg,
                            },
                        )
                    )

            except Exception as e:
                self.logger.error(f"OpenAI API error in tool execution loop: {e}")
                raise RuntimeError(
                    f"OpenAI API error in tool execution loop: {e}"
                ) from e

        # Max iterations reached
        self.logger.warning(
            f"Maximum tool iterations ({max_tool_iterations}) reached without final response"
        )
        raise RuntimeError(
            f"Tool execution exceeded maximum iterations ({max_tool_iterations}). "
            f"The conversation may be stuck in a loop."
        )

    def configure(self, **kwargs: Any) -> None:
        """Update provider configuration.

        Args:
            **kwargs: Configuration options to update
        """
        self.logger.info("Updating provider configuration")
        config_changes = []

        # Handle critical settings that affect client
        if "api_key" in kwargs:
            old_key = self.api_key
            self.api_key = kwargs["api_key"]
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            config_changes.append("api_key")

        if "model" in kwargs:
            old_model = self.model
            self.model = kwargs["model"]
            config_changes.append(f"model: {old_model} -> {self.model}")

        if "base_url" in kwargs:
            old_url = self.base_url
            self.base_url = kwargs["base_url"]
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            config_changes.append(f"base_url: {old_url} -> {self.base_url}")

        # Update all config parameters
        for key, value in kwargs.items():
            if key in self.config:
                old_value = self.config.get(key)
                self.config[key] = value
                if key not in ["api_key", "model", "base_url"]:  # Already logged above
                    config_changes.append(f"{key}: {old_value} -> {value}")
            else:
                # Add new configuration options
                self.config[key] = value
                config_changes.append(f"{key}: added -> {value}")

        if config_changes:
            self.logger.info(f"Configuration updated: {', '.join(config_changes)}")
        else:
            self.logger.debug("No configuration changes applied")

    @contextmanager
    def temp_config(self, **kwargs: Any) -> Iterator[None]:
        """Temporarily override configuration settings.
        
        This context manager allows temporary changes to provider configuration
        that are automatically restored when exiting the context.
        
        Args:
            **kwargs: Configuration overrides to apply temporarily
            
        Examples:
            >>> with provider.temp_config(temperature=0.1, max_tokens=100):
            ...     result = provider.generate("Precise question")
            >>> # Configuration restored after context
        """
        # Save current state
        old_config = self.config.copy()
        old_model = self.model
        old_api_key = self.api_key
        old_base_url = self.base_url
        old_client = self.client
        
        try:
            # Apply temporary configuration
            self.configure(**kwargs)
            yield
        finally:
            # Restore original state
            self.config = old_config
            self.model = old_model
            self.api_key = old_api_key
            self.base_url = old_base_url
            self.client = old_client

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model name, version, capabilities, etc.
        """
        return {
            "provider": "OpenAI",
            "model": self.model,
            "base_url": self.base_url,
            "supports_structured": True,
            "supports_pydantic": True,
            "supports_tools": True,
            "supports_history": True,
        }

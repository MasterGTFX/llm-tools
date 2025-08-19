"""OpenAI LLM provider implementation using OpenAI SDK."""

import json
import os
from typing import Any, Optional, Union

from dotenv import load_dotenv  # type: ignore[import-not-found]

from llmtools.interfaces.llm import LLMInterface
from llmtools.utils.logging import setup_logger

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "OpenAI SDK is required for OpenAI provider. "
        "Install with: pip install 'llmtools[openai]' or pip install openai"
    ) from e


class OpenAIProvider(LLMInterface):
    """OpenAI LLM provider using OpenAI SDK with configurable base URL."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-nano",
        base_url: Optional[str] = None,
        **client_kwargs: Any,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If not provided, will try to load from
                    OPENAI_API_KEY environment variable
            model: Model to use (default: gpt-5-nano)
            base_url: Custom base URL (e.g., for OpenRouter). If not provided,
                     uses OpenAI's default endpoint
            **client_kwargs: Additional arguments to pass to OpenAI client
        """
        # Load environment variables from .env file
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter"
            )

        self.model = model
        self.base_url = base_url

        # Initialize OpenAI client with optional custom base URL
        client_params = {
            "api_key": self.api_key,
            **client_kwargs,
        }
        if self.base_url:
            client_params["base_url"] = self.base_url

        self.client = OpenAI(**client_params)

        # Default configuration
        self.config = {
            "temperature": 0.7,
            "max_tokens": None,
            "timeout": 30,
        }

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

        # Merge config with kwargs
        request_params = {
            "model": self.model,
            "messages": messages,
            **self.config,
            **kwargs,
        }
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

        # Merge config with kwargs
        request_params = {
            "model": self.model,
            "messages": messages,
            "response_format": response_format,
            **self.config,
            **kwargs,
        }

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

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Union[str, dict[str, Any]]:
        """Generate response with access to function/tool calling.

        Args:
            prompt: The user prompt/input text
            tools: List of available tools/functions with their schemas
            system_prompt: Optional system prompt to guide behavior
            history: Optional conversation history as list of {"role": str, "content": str}
            **kwargs: Additional provider-specific parameters

        Returns:
            Either a text response or structured tool call data
        """
        messages = self._build_messages(prompt, system_prompt, history)

        # Merge config with kwargs
        request_params = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            **self.config,
            **kwargs,
        }

        try:
            self.logger.info(f"Generating response with tools, model: {self.model}")
            self.logger.debug(
                f"Available tools: {[tool.get('function', {}).get('name', 'unknown') for tool in tools]}"
            )

            response = self.client.chat.completions.create(**request_params)
            message = response.choices[0].message

            # Log token usage if available
            if hasattr(response, "usage") and response.usage:
                self.logger.info(
                    f"Token usage - prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}, total: {response.usage.total_tokens}"
                )

            # Check if the model wants to call a tool
            if message.tool_calls:
                tool_names = [tc.function.name for tc in message.tool_calls]
                self.logger.info(f"Model requested tool calls: {tool_names}")
                return {
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in message.tool_calls
                    ],
                }
            else:
                # Regular text response
                self.logger.info("Model returned text response (no tool calls)")
                content = message.content or ""
                self.logger.debug(f"Response length: {len(content)} characters")
                return content

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}") from e

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the LLM provider with settings.

        Args:
            config: Configuration dictionary (API keys, model params, etc.)
        """
        self.logger.info("Updating configuration")
        config_changes = []

        if "api_key" in config:
            self.api_key = config["api_key"]
            # Recreate client with new API key
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            config_changes.append("api_key")

        if "model" in config:
            old_model = self.model
            self.model = config["model"]
            config_changes.append(f"model: {old_model} -> {self.model}")

        if "base_url" in config:
            old_url = self.base_url
            self.base_url = config["base_url"]
            # Recreate client with new base URL
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            config_changes.append(f"base_url: {old_url} -> {self.base_url}")

        # Update other config parameters
        for key, value in config.items():
            if key in ["temperature", "max_tokens", "timeout"]:
                old_value = self.config.get(key)
                self.config[key] = value
                config_changes.append(f"{key}: {old_value} -> {value}")

        if config_changes:
            self.logger.info(f"Configuration updated: {', '.join(config_changes)}")
        else:
            self.logger.debug("No configuration changes applied")

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
            "supports_tools": True,
            "supports_history": True,
        }

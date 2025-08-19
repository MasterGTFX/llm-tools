"""OpenRouter LLM provider implementation using OpenAI SDK."""

import json
import os
from typing import Any, Optional, Union

from llmtools.interfaces.llm import LLMInterface

try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "OpenAI SDK is required for OpenRouter provider. "
        "Install with: pip install 'llmtools[openrouter]' or pip install openai"
    ) from e


class OpenRouterProvider(LLMInterface):
    """OpenRouter LLM provider using OpenAI SDK with OpenRouter API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o",
        base_url: str = "https://openrouter.ai/api/v1",
        **client_kwargs: Any,
    ):
        """Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key. If not provided, will try to load from
                    OPENROUTER_API_KEY environment variable
            model: Model to use (default: openai/gpt-4o)
            base_url: OpenRouter API base URL
            **client_kwargs: Additional arguments to pass to OpenAI client
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. Set OPENROUTER_API_KEY environment "
                "variable or pass api_key parameter"
            )

        self.model = model
        self.base_url = base_url

        # Initialize OpenAI client with OpenRouter settings
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **client_kwargs,
        )

        # Default configuration
        self.config = {
            "temperature": 0.7,
            "max_tokens": None,
            "timeout": 30,
        }

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
        messages = self._build_messages(prompt, system_prompt, history)

        # Merge config with kwargs
        request_params = {
            "model": self.model,
            "messages": messages,
            **self.config,
            **kwargs,
        }

        try:
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {e}") from e

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

        # Configure structured output using OpenRouter's format
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
            response = self.client.chat.completions.create(**request_params)
            content = response.choices[0].message.content or "{}"
            parsed_result = json.loads(content)
            if isinstance(parsed_result, dict):
                return parsed_result
            raise ValueError("Expected JSON object, got other type")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse structured response as JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {e}") from e

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
            response = self.client.chat.completions.create(**request_params)
            message = response.choices[0].message

            # Check if the model wants to call a tool
            if message.tool_calls:
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
                return message.content or ""

        except Exception as e:
            raise RuntimeError(f"OpenRouter API error: {e}") from e

    def configure(self, config: dict[str, Any]) -> None:
        """Configure the LLM provider with settings.

        Args:
            config: Configuration dictionary (API keys, model params, etc.)
        """
        if "api_key" in config:
            self.api_key = config["api_key"]
            # Recreate client with new API key
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        if "model" in config:
            self.model = config["model"]

        if "base_url" in config:
            self.base_url = config["base_url"]
            # Recreate client with new base URL
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Update other config parameters
        for key, value in config.items():
            if key in ["temperature", "max_tokens", "timeout"]:
                self.config[key] = value

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model name, version, capabilities, etc.
        """
        return {
            "provider": "OpenRouter",
            "model": self.model,
            "base_url": self.base_url,
            "supports_structured": True,
            "supports_tools": True,
            "supports_history": True,
        }

"""Abstract LLM interface defining the contract for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union


class LLMInterface(ABC):
    """Abstract base class for LLM providers.

    This interface defines the minimal contract that LLM providers must implement
    to work with llmtools components like KnowledgeBase and Sorter.
    """

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def configure(self, config: dict[str, Any]) -> None:
        """Configure the LLM provider with settings.

        Args:
            config: Configuration dictionary (API keys, model params, etc.)
        """
        pass

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model.

        Returns:
            Dictionary with model name, version, capabilities, etc.
        """
        return {
            "provider": self.__class__.__name__,
            "model": "unknown",
            "supports_structured": True,
            "supports_tools": True,
        }

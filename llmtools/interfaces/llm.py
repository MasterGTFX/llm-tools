"""Abstract LLM interface defining the contract for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


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
        **kwargs: Any,
    ) -> str:
        """Generate a text response from the LLM.

        Args:
            prompt: The user prompt/input text
            system_prompt: Optional system prompt to guide behavior
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response from the LLM
        """
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate structured output conforming to a JSON schema.

        Args:
            prompt: The user prompt/input text
            schema: JSON schema the response must conform to
            system_prompt: Optional system prompt to guide behavior
            **kwargs: Additional provider-specific parameters

        Returns:
            Structured response as a dictionary conforming to schema
        """
        pass

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Generate response with access to function/tool calling.

        Args:
            prompt: The user prompt/input text
            tools: List of available tools/functions with their schemas
            system_prompt: Optional system prompt to guide behavior
            **kwargs: Additional provider-specific parameters

        Returns:
            Either a text response or structured tool call data
        """
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the LLM provider with settings.

        Args:
            config: Configuration dictionary (API keys, model params, etc.)
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
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

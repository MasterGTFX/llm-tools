"""Abstract LLM interface defining the contract for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMInterface(ABC):
    """Abstract base class for LLM providers.

    This interface defines the minimal contract that LLM providers must implement
    to work with vibetools components like KnowledgeBase and Sorter.
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
        pass

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        functions: Optional[list[Callable[..., Any]]] = None,
        function_map: Optional[dict[Callable[..., Any], dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        max_tool_iterations: int = 10,
        handle_tool_errors: bool = True,
        tool_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """Generate response with access to function/tool calling.

        Args:
            prompt: The user prompt/input text
            functions: Optional list of Python functions to auto-convert to tools.
                      Function schemas are generated from type hints and docstrings.
            function_map: Optional dict mapping Python functions to their schema definitions.
                         Either functions or function_map (or both) must be provided.
            system_prompt: Optional system prompt to guide behavior
            history: Optional conversation history as list of {"role": str, "content": str}
            max_tool_iterations: Maximum number of tool calling rounds to prevent infinite loops
            handle_tool_errors: Whether to handle tool execution errors gracefully by informing the LLM
            tool_timeout: Optional timeout in seconds for individual tool execution
            **kwargs: Additional provider-specific parameters

        Returns:
            Final text response after all tool calls are completed
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
            "supports_pydantic": True,
            "supports_tools": True,
        }

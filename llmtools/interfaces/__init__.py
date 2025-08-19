"""Abstract interfaces for LLM providers and other components."""

from llmtools.interfaces.llm import LLMInterface
from llmtools.interfaces.openai_llm import OpenAIProvider

__all__ = ["LLMInterface", "OpenAIProvider"]

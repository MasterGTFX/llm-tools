"""Abstract interfaces for LLM providers and other components."""

from llmtools.interfaces.llm import LLMInterface

# Import OpenRouter provider with optional dependency handling
try:
    from llmtools.interfaces.openrouter_llm import OpenRouterProvider
    __all__ = ["LLMInterface", "OpenRouterProvider"]
except ImportError:
    # OpenAI SDK not installed
    __all__ = ["LLMInterface"]

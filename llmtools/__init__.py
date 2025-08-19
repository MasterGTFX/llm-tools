"""
llmtools: A modular set of LLM utilities for personal and project use.

This package provides reusable building blocks for working with LLMs:
- KnowledgeBase: Build and maintain versioned knowledge bases from documents
- Sorter: Sort/filter Python lists using LLM instructions
- Interfaces: Abstract definitions for LLM providers
- Utils: Shared utilities for diff management, chunking, etc.
"""

from llmtools.interfaces import LLMInterface
from llmtools.knowledge_base import KnowledgeBase
from llmtools.sorter import Sorter

# Import OpenRouter provider with optional dependency handling
try:
    from llmtools.interfaces import OpenRouterProvider

    __all__ = ["KnowledgeBase", "Sorter", "LLMInterface", "OpenRouterProvider"]
except ImportError:
    # OpenAI SDK not installed
    __all__ = ["KnowledgeBase", "Sorter", "LLMInterface"]

__version__ = "0.1.0"

"""
llmtools: A modular set of LLM utilities for personal and project use.

This package provides reusable building blocks for working with LLMs:
- KnowledgeBase: Build and maintain versioned knowledge bases from documents
- Sorter: Sort/filter Python lists using LLM instructions
- Interfaces: Abstract definitions for LLM providers
- Utils: Shared utilities for diff management, chunking, etc.
"""

from llmtools.interfaces import LLMInterface, OpenAIProvider
from llmtools.knowledge_base import KnowledgeBase
from llmtools.sorter import Sorter

__all__ = ["KnowledgeBase", "Sorter", "LLMInterface", "OpenAIProvider"]

__version__ = "0.1.0"

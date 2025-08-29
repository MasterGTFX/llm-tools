"""
llmtools: A collection of simple, self-contained LLM functions.

This package provides simple function-based tools that solve specific problems:
- llm_filter: Filter lists using natural language instructions
- llm_sorter: Sort lists using natural language instructions
- llm_knowledge_base: Build knowledge bases from documents (coming soon)
- llm_edit: Edit text using LLM instructions (coming soon)
- Interfaces: Abstract definitions for LLM providers
- Utils: Shared utilities for diff management, chunking, etc.
"""

from llmtools.interfaces import LLMInterface, OpenAIProvider
from llmtools.tools import llm_edit, llm_filter, llm_sorter
from llmtools import prompts

__all__ = ["llm_filter", "llm_sorter", "llm_edit", "LLMInterface", "OpenAIProvider", "prompts"]

__version__ = "0.1.0"

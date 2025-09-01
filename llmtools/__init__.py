"""
llmtools: A collection of simple, self-contained LLM functions.

This package provides simple function-based tools that solve specific problems:
- llm_ask: Ask yes/no questions and get boolean answers
- llm_filter: Filter lists using natural language instructions
- llm_sorter: Sort lists using natural language instructions
- llm_summary: Create iterative summaries from multiple documents
- llm_edit: Edit text using LLM instructions
- Interfaces: Abstract definitions for LLM providers
- Utils: Shared utilities for diff management, chunking, etc.
"""

from llmtools.interfaces import LLMInterface, OpenAIProvider
from llmtools.tools import llm_ask, llm_edit, llm_filter, llm_sorter, llm_summary
from llmtools.defaults import configure, get_provider, temp_config, reset_default_provider, reset_configuration
from llmtools import prompts

__all__ = [
    # Tool functions
    "llm_ask", "llm_filter", "llm_sorter", "llm_edit", "llm_summary", 
    # Interfaces
    "LLMInterface", "OpenAIProvider", 
    # Configuration
    "configure", "get_provider", "temp_config", "reset_default_provider", "reset_configuration",
    # Other
    "prompts"
]

__version__ = "0.1.0"

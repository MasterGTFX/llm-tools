"""
vibetools: A collection of simple, self-contained AI functions.

This package provides simple function-based tools that solve specific problems:
- ai_ask: Ask yes/no questions and get boolean answers
- ai_filter: Filter lists using natural language instructions
- ai_sort: Sort lists using natural language instructions
- ai_summary: Create iterative summaries from multiple documents
- ai_edit: Edit text using AI instructions
- Interfaces: Abstract definitions for LLM providers
- Utils: Shared utilities for diff management, chunking, etc.
"""

from vibetools.interfaces import LLMInterface, OpenAIProvider
from vibetools.tools import ai_ask, ai_edit, ai_filter, ai_sort, ai_summary
from vibetools.defaults import configure, get_provider, temp_config, reset_default_provider, reset_configuration
from vibetools import prompts

__all__ = [
    # Tool functions
    "ai_ask", "ai_filter", "ai_sort", "ai_edit", "ai_summary", 
    # Interfaces
    "LLMInterface", "OpenAIProvider", 
    # Configuration
    "configure", "get_provider", "temp_config", "reset_default_provider", "reset_configuration",
    # Other
    "prompts"
]

__version__ = "0.1.0"

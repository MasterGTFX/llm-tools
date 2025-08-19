# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a modular Python toolkit for LLM utilities designed to be minimal, configurable, and composable. Each tool can be imported independently without heavy dependencies.

## Development Setup

Install the package in development mode with all dependencies:

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (recommended)
pre-commit install

# Run all quality checks
ruff check llmtools/ tests/        # Linting
ruff format llmtools/ tests/       # Formatting
mypy llmtools/ --strict           # Type checking
pytest tests/ -v --cov=llmtools  # Testing with coverage

# Run tests only
pytest tests/

# Run specific test file
pytest tests/test_sorter.py -v
```

## Architecture

The project follows a modular structure with working implementations:

### Core Modules
- `llmtools.knowledge_base` - Build and update knowledge bases from documents with versioning
- `llmtools.sorter` - Sort/filter Python lists using LLM instructions with strict/filter modes
- `llmtools.interfaces` - Abstract LLM interface definitions
- `llmtools.utils` - Shared utilities (diff management, chunking, embeddings, structured output)
- `llmtools.config` - Pydantic-based configuration models

### Key Design Principles
- **Minimal dependencies**: Only Pydantic required, optional integrations available
- **Pythonic interfaces**: Clean module boundaries and clear abstractions
- **Structured output**: Leverage LLM tool calling for predictable JSON/diff responses
- **Composable tools**: Each component works independently or together
- **Configuration-driven**: Pydantic models for flexible configuration

### Knowledge Base Component
- Supports incremental updates with `.history/` versioning
- Uses structured LLM output for diff management
- Can initialize with existing knowledge base
- Automatically creates metadata.json tracking versions
- API matches README examples exactly

### Sorter Component
- **Strict mode**: Preserves all input items, only reorders
- **Filter mode**: Returns subset matching conditions
- Includes validation and type restoration
- Configurable retry logic for LLM failures

## LLM Provider Integration

The project uses an abstract `LLMInterface` in `llmtools.interfaces.llm` that requires:
- `generate()` - Basic text generation
- `generate_structured()` - JSON schema-conforming output
- `generate_with_tools()` - Function/tool calling capability

Example LLM provider implementations should inherit from this interface.

## Current Implementation Status

✅ **Complete and functional**:
- Package structure with proper imports
- Pydantic configuration models (LLMConfig, KnowledgeBaseConfig, SorterConfig)
- KnowledgeBase class with document processing and versioning
- Sorter class with strict/filter modes and validation
- DiffManager utility for version tracking
- Comprehensive test suite (16 tests passing)
- Modern development tooling (ruff, mypy, pre-commit)
- GitHub Actions CI/CD

🔄 **Ready for extension**:
- Additional LLM provider implementations
- More utility functions in `llmtools.utils`
- Enhanced chunking and embedding features
- Real LLM provider integrations (OpenAI, Anthropic, etc.)

## Working Examples

See `examples/basic_usage.py` for functional demonstrations of both KnowledgeBase and Sorter components using mock LLM providers.

## Developer notes
- make sure that always latest possible version of libraries are installed
- examples should be simple, basic, straightforward to show up usage to User - do not cover all scenarios etc
- Check code quality with tools like ruff, mypy
- Do not mock in tests. Create just simple unit/functional tests
- we're in active development, not clients yet - do not bother about backward compatible
- use the local /venv
- while logging, be clear, concise yet meaningful. use proper logging level. use getlogger with specific logger names

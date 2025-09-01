# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a collection of simple, self-contained AI functions that solve specific problems with a single function call. Each function uses OpenAI by default and requires no configuration or setup.

## Development Setup

Install the package in development mode with all dependencies:

```bash
# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (recommended)
pre-commit install

# Run all quality checks
ruff check vibetools/ tests/        # Linting
ruff format vibetools/ tests/       # Formatting
mypy vibetools/ --strict           # Type checking
pytest tests/ -v --cov=vibetools  # Testing with coverage

# Run tests only
pytest tests/

# Run specific test file
pytest tests/test_sorter.py -v
```

## Architecture

The project provides simple function-based tools:

### Available Tools
- `ai_ask(question, context="")` - Ask yes/no questions and get boolean answers
- `ai_filter(items, instruction)` - Filter lists using natural language
- `ai_sort(items, instruction)` - Sort lists using natural language
- `ai_summary(documents, instruction)` - Create iterative summaries from multiple documents
- `ai_edit(text, instruction)` - Edit text using AI instructions

### Key Design Principles
- **Function-based**: Everything is a simple function call
- **Self-contained**: Each function handles its own LLM setup internally
- **Default OpenAI**: Uses OpenAI provider with sensible defaults
- **No configuration**: Works immediately with just OPENAI_API_KEY
- **Structured output**: Leverages JSON schema and function calling internally
- **Type restoration**: Maintains original data types where possible

### Function Implementation Pattern
Each tool function follows this pattern:
1. **Input validation**: Check parameters and types
2. **LLM setup**: Create OpenAI provider with defaults
3. **Structured call**: Use JSON schema for predictable output
4. **Result processing**: Convert back to expected types
5. **Error handling**: Graceful fallbacks and clear error messages

## LLM Integration

All functions use the existing `OpenAIProvider` from `vibetools.interfaces.openai_llm`:
- **Default provider**: Creates OpenAI client automatically
- **Environment-based**: Uses `OPENAI_API_KEY` from environment
- **Structured output**: Leverages `generate_structured()` method
- **Function calling**: Uses `generate_with_tools()` when needed

Functions create their own provider instances internally with sensible defaults.

## Current Implementation Status

✅ **Infrastructure ready**:
- OpenAI provider implementation with structured output
- Tool execution utilities and function calling
- Development tooling (ruff, mypy, pre-commit)
- Test framework setup

✅ **Tools implemented**:
- `ai_ask()` function in `tools/ask.py`
- `ai_filter()` function in `tools/filter.py`
- `ai_sort()` function in `tools/sorter.py`
- `ai_summary()` function in `tools/summary.py`
- `ai_edit()` function in `tools/edit.py`
- All functions exported in `__init__.py`
- Usage examples created in `examples/`

🔄 **Future enhancements**:
- Additional tool functions as needed
- Expanded test coverage
- Performance optimizations

## Working Examples

See `examples/` directory for simple function usage examples demonstrating each tool.

## Tool Creation Guidelines

### Prompt Organization
- Create dedicated prompt files in `vibetools/prompts/[tool_name]_prompts.py`
- Export `SYSTEM_PROMPT` constant and `user_prompt()` function
- Users can override by modifying these prompt files directly

### Type Hints & Docstrings
- Use comprehensive type hints on all parameters and returns
- Main function must have detailed docstring with Args/Returns/Raises sections
- Internal tool functions must have docstrings with Args descriptions
- Use modern syntax: `list[Any]` not `List[Any]`, `LLMInterface` for provider

### Function Architecture
- Accept `LLMInterface` as parameter for dependency injection
- Define internal tool functions within main function using `nonlocal` for state
- Use `llm_provider.generate_with_tools()` with system prompts from prompt files
- Handle exceptions with clear error messages

### Code Quality
- Set up logger with `setup_logger(__name__)`
- Add meaningful logging (info/warning/error levels)
- Validate inputs and provide graceful fallbacks
- Follow existing patterns from `ask.py`, `filter.py`, and `edit.py`

## Developer Notes

### Function Development Guidelines
- **Self-contained**: Each function handles its own OpenAI provider setup
- **Default config**: Use `OpenAIProvider(model="gpt-5-mini")` as default
- **Structured output**: Always use `generate_structured()` with JSON schema
- **Type restoration**: Convert results back to original input types
- **Error handling**: Graceful fallbacks, clear error messages
- **Logging**: Use specific logger names, meaningful messages

### Code Quality
- Run `ruff check` and `ruff format` before committing
- Use `mypy --strict` for type checking
- Keep functions simple and focused on single tasks
- Examples should be basic and straightforward
- Use latest library versions
- No backward compatibility concerns during active development

### Testing
- Create simple unit tests for each function
- Test with real OpenAI calls (no mocking)
- Use local `/venv` for development
- Make sure the examples are easy to read&understand, clear, concise and demonstrates the functionality very well. Include simple/advanced example(s)
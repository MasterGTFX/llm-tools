"""Diff management utilities for tracking changes in knowledge bases."""

import json
import re

from llmtools.interfaces.llm import LLMInterface
from llmtools.utils.logging import setup_logger

# Module-level constants
DEFAULT_MAX_RETRIES = 2
PARTIAL_MATCH_THRESHOLD = 0.7
MAX_SEARCH_DISPLAY_LENGTH = 100
MAX_REPLACE_DISPLAY_LENGTH = 100
INDENTATION_STEP = 4
MAX_INDENTATION_LEVELS = 5

# Module-level logger
logger = setup_logger(__name__)

# Function tool specification for edit_content
EDIT_CONTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_content",
        "description": "Apply search-replace edit to content",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why this specific change is needed",
                },
                "search": {"type": "string", "description": "Exact text to find"},
                "replace": {
                    "type": "string",
                    "description": "Exact text to replace it with",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Whether to replace all occurrences",
                    "default": False,
                },
            },
            "required": ["reasoning", "search", "replace"],
        },
    },
}

# System prompt for LLM diff generation
SYSTEM_PROMPT = """You are an expert content editor. Use the edit_content function to modify text content.

Call edit_content multiple times as needed to make all required changes.

Guidelines:
- Copy search text EXACTLY from original (including whitespace/indentation)
- Use replace_all=true only when you want to replace ALL occurrences
- Include enough context to make search text unique when replace_all=false
- Make one edit at a time and wait for confirmation before proceeding"""


def apply_llm_diff(
    original_content: str,
    prompt: str,
    llm_provider: LLMInterface,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    """Generate and apply LLM diff with retry on failure.

    Args:
        original_content: Original content to modify
        prompt: Instruction for what changes to make
        llm_provider: LLM interface for generating diffs
        max_retries: Maximum number of retry attempts on failure

    Returns:
        Modified content after applying LLM-generated diff

    Raises:
        ValueError: If diff cannot be applied after all retries
    """
    logger.info("Starting LLM diff generation and application")

    user_prompt = f"""<user_instruction>
{prompt}
</user_instruction>

<text_to_modify>
{original_content}
</text_to_modify>"""

    return _generate_and_apply_with_tools(
        original_content, user_prompt, llm_provider, max_retries
    )


def _generate_and_apply_with_tools(
    original_content: str,
    user_prompt: str,
    llm_provider: LLMInterface,
    max_retries: int,
) -> str:
    """Handle diff generation using function calling."""
    current_content = original_content
    conversation_history: list[dict[str, str]] = []

    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Attempt {attempt + 1} of {max_retries + 1}")

            # Create edit handler function
            def edit_content(
                reasoning: str, search: str, replace: str, replace_all: bool = False
            ) -> tuple[bool, str]:
                nonlocal current_content
                try:
                    result = _apply_search_replace_simple(
                        current_content, search, replace, replace_all
                    )
                    current_content = result
                    return (True, "OK")
                except ValueError as e:
                    error_msg = str(e)
                    if "not found" in error_msg.lower():
                        return (False, f"SEARCH_NOT_FOUND: {search[:50]}...")
                    elif "appears" in error_msg and "times" in error_msg:
                        count = current_content.count(search)
                        return (
                            False,
                            f"MULTIPLE_MATCHES: found {count} occurrences, use replace_all=true",
                        )
                    else:
                        return (False, f"ERROR: {error_msg}")

            # Use function calling
            response = llm_provider.generate_with_tools(
                prompt=user_prompt,
                tools=[EDIT_CONTENT_TOOL],
                system_prompt=SYSTEM_PROMPT,
                history=conversation_history,
            )

            # Handle tool calls if present
            if isinstance(response, dict) and response.get("tool_calls"):
                for tool_call in response["tool_calls"]:
                    if tool_call["function"]["name"] == "edit_content":
                        args = json.loads(tool_call["function"]["arguments"])
                        reasoning = args.get("reasoning", "")
                        logger.info(f"Applying edit: {reasoning}")

                        success, message = edit_content(
                            reasoning=reasoning,
                            search=args.get("search", ""),
                            replace=args.get("replace", ""),
                            replace_all=args.get("replace_all", False),
                        )

                        # Log the result
                        if success:
                            logger.info(f"✓ Edit successful: {reasoning}")
                        else:
                            logger.warning(f"✗ Edit failed: {reasoning} - {message}")

                        # Add tool result to conversation
                        conversation_history.append(
                            {
                                "role": "assistant",
                                "content": f"Called edit_content: {reasoning or 'No reason provided'}",
                            }
                        )

                        # For failed edits, include current content in response
                        if not success:
                            response_content = (
                                f"{message}\n\nCurrent content:\n{current_content}"
                            )
                        else:
                            response_content = message

                        conversation_history.append(
                            {"role": "user", "content": response_content}
                        )
            elif isinstance(response, str):
                # No tool calls, just a text response - this might be an error or completion
                logger.info(f"LLM returned text response: {response[:100]}...")
                raise ValueError(f"No edits were made: {response}")

            # Check if we've made any changes
            if current_content != original_content:
                logger.info("Successfully applied all LLM edits")
                return current_content
            else:
                # No changes made, treat as failure for retry
                raise ValueError("No changes were applied to the content")

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")

            if attempt < max_retries:
                logger.info("Retrying with updated context...")
                _add_retry_context_to_conversation(
                    conversation_history, current_content, error_msg
                )
            else:
                logger.error("All attempts failed")
                raise ValueError(
                    f"Failed to apply LLM diffs after {max_retries + 1} attempts: {error_msg}"
                ) from e

    raise RuntimeError("Unreachable code")


def _add_retry_context_to_conversation(
    conversation_history: list[dict[str, str]], current_content: str, error_msg: str
) -> None:
    """Add retry context to conversation history."""
    conversation_history.append(
        {"role": "assistant", "content": f"Error occurred: {error_msg}"}
    )
    conversation_history.append(
        {
            "role": "user",
            "content": f"Current content:\n{current_content}\n\nPlease try again with a different approach.",
        }
    )


def _apply_search_replace_simple(
    original: str, search: str, replace: str, replace_all: bool
) -> str:
    """Apply a search-replace with flexible matching strategies.

    Args:
        original: Original text content
        search: Text to find
        replace: Text to replace it with
        replace_all: Whether to replace all occurrences

    Returns:
        Modified text content

    Raises:
        ValueError: If search text cannot be found or applied
    """

    if not search.strip():
        logger.warning("Empty search text, skipping")
        return original

    # Check for multiple matches when replace_all is False
    if not replace_all and original.count(search) > 1:
        raise ValueError(
            f"Search text '{search[:50]}...' appears {original.count(search)} times. Set replace_all=true or be more specific with context."
        )

    # Try exact match first
    if search in original:
        if replace_all:
            result = original.replace(search, replace)
            logger.debug(
                f"Applied search-replace with replace_all=True ({original.count(search)} matches)"
            )
        else:
            result = original.replace(search, replace, 1)
            logger.debug("Applied search-replace with exact match")
        return result

    # Try flexible matching strategies
    result = _apply_search_flexible(original, search, replace, replace_all)
    if result != original:
        return result

    raise ValueError(f"Search text not found: '{search[:100]}...'")


def _apply_search_flexible(
    original: str, search: str, replace: str, replace_all: bool
) -> str:
    """Apply search-replace with flexible matching strategies."""

    # Strategy 1: Normalize whitespace
    normalized_search = _normalize_whitespace(search)
    if normalized_search in _normalize_whitespace(original):
        result = _replace_with_normalization(original, search, replace)
        logger.debug("Applied search-replace with whitespace normalization")
        return result

    # Strategy 2: Try with different indentation levels
    result = _apply_with_indentation_fix(original, search, replace)
    if result != original:
        logger.debug("Applied search-replace with indentation correction")
        return result

    # Strategy 3: Try partial line matching
    result = _apply_partial_line_matching(original, search, replace)
    if result != original:
        logger.debug("Applied search-replace with partial line matching")
        return result

    return original


def _apply_partial_line_matching(original: str, search: str, replace: str) -> str:
    """Try to match individual lines when full search text doesn't match."""
    search_lines = search.splitlines()
    original_lines = original.splitlines()

    # Try to find a sequence of lines that match
    for start_idx in range(len(original_lines)):
        matched_lines = 0
        for search_line in search_lines:
            check_idx = start_idx + matched_lines
            if (
                check_idx < len(original_lines)
                and original_lines[check_idx].strip() == search_line.strip()
            ):
                matched_lines += 1
            else:
                break

        # If we matched most lines, try the replacement
        if matched_lines >= len(search_lines) * PARTIAL_MATCH_THRESHOLD:
            end_idx = start_idx + len(search_lines)
            new_lines = (
                original_lines[:start_idx]
                + replace.splitlines()
                + original_lines[end_idx:]
            )
            return "\n".join(new_lines)

    return original


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace for flexible matching."""
    # Replace multiple whitespace with single space
    normalized = re.sub(r"\s+", " ", text)
    return normalized.strip()


def _replace_with_normalization(original: str, search: str, replace: str) -> str:
    """Replace text using whitespace normalization to find the match."""
    # Escape special regex characters in search text
    escaped_search = re.escape(search)
    # Replace escaped whitespace with flexible whitespace pattern
    flexible_pattern = re.sub(r"\\\s+", r"\\s+", escaped_search)

    try:
        result = re.sub(flexible_pattern, replace, original, count=1)
        return result
    except re.error:
        # Fallback to simple replacement if regex fails
        return original.replace(search, replace, 1)


def _apply_with_indentation_fix(original: str, search: str, replace: str) -> str:
    """Handle case where GPT removes consistent leading whitespace."""
    search_lines = search.splitlines()
    replace_lines = replace.splitlines()

    if not search_lines:
        return original

    # Try to find the search content with various indentation levels
    max_indent = MAX_INDENTATION_LEVELS * INDENTATION_STEP
    for base_indent in range(0, max_indent, INDENTATION_STEP):
        indented_search = []
        indented_replace = []

        for line in search_lines:
            if line.strip():  # Non-empty line
                indented_search.append(" " * base_indent + line.strip())
            else:
                indented_search.append(line)

        for line in replace_lines:
            if line.strip():  # Non-empty line
                indented_replace.append(" " * base_indent + line.strip())
            else:
                indented_replace.append(line)

        search_text = "\n".join(indented_search)
        replace_text = "\n".join(indented_replace)

        if search_text in original:
            return original.replace(search_text, replace_text, 1)

    return original

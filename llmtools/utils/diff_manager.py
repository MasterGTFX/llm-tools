"""Diff management utilities for tracking changes in knowledge bases.

Provides functions to apply LLM-generated diffs to text content using structured
function calling with automatic retry and flexible matching strategies.
"""

from typing import Optional

from llmtools.interfaces.llm import LLMInterface
from llmtools.utils.logging import setup_logger

logger = setup_logger(__name__)


def llm_edit(
    original_content: str,
    prompt: str,
    llm_provider: LLMInterface,
    expect_edit: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate and apply LLM edit using function calling.

    Args:
        original_content: Original content to modify
        prompt: Instruction for what changes to make
        llm_provider: LLM interface for generating diffs
        expect_edit: Whether to expect content changes (default: False)
        system_prompt: Custom system prompt (default: use built-in prompt)

    Returns:
        Modified content after applying LLM-generated edit

    Raises:
        ValueError: If edit cannot be applied
    """
    logger.info("Starting LLM edit generation and application")

    user_prompt = f"""<user_instruction>
{prompt}
</user_instruction>

<text_to_modify>
{original_content}
</text_to_modify>"""

    current_content = original_content
    failed_attempts = 0

    def edit_content_tool(search: str, replace: str, replace_all: bool = False) -> str:
        """Apply search-replace edit to content."""
        nonlocal current_content, failed_attempts

        if not search.strip():
            failed_attempts += 1
            message = "ERROR: Search text cannot be empty. Please provide the exact text you want to find and replace."
            if failed_attempts >= 2:
                message += f"\n\n=== CURRENT CONTENT (after {failed_attempts} failures) ===\n{current_content}\n=== END CURRENT CONTENT ==="
            logger.warning(f"Edit failed: {message}")
            return message

        search_count = current_content.count(search)

        if search_count == 0:
            failed_attempts += 1
            message = f"SEARCH_NOT_FOUND: Could not find text '{search[:50]}...'. Check for exact whitespace/indentation match or try a shorter, more specific pattern."
            if failed_attempts >= 2:
                message += f"\n\n=== CURRENT CONTENT (after {failed_attempts} failures) ===\n{current_content}\n=== END CURRENT CONTENT ==="
            logger.warning(f"Edit failed: {message}")
            return message

        if search_count > 1 and not replace_all:
            failed_attempts += 1
            message = f"MULTIPLE_MATCHES: Found {search_count} occurrences of '{search[:30]}...'. Add more surrounding context to make it unique, or use replace_all=True to replace all."
            if failed_attempts >= 2:
                message += f"\n\n=== CURRENT CONTENT (after {failed_attempts} failures) ===\n{current_content}\n=== END CURRENT CONTENT ==="
            logger.warning(f"Edit failed: {message}")
            return message

        if replace_all:
            current_content = current_content.replace(search, replace)
            message = f"OK: Replaced {search_count} occurrences"
        else:
            current_content = current_content.replace(search, replace, 1)
            message = "OK: Edit applied successfully"

        failed_attempts = 0  # Reset counter on successful edit
        logger.info(f"Edit applied: {message}")
        return message

    if system_prompt is None:
        system_prompt = """You are an expert content editor. Use the edit_content_tool function to modify text content.

Call edit_content_tool multiple times as needed to make all required changes.

Guidelines:
- Copy search text EXACTLY from original (including whitespace/indentation)
- Use replace_all=True only when you want to replace ALL occurrences
- Include enough context to make search text unique when replace_all=False
- Make one edit at a time and wait for confirmation before proceeding"""
    else:
        system_prompt = (
            f"You are an expert content editor. Use the edit_content_tool function to modify text content."
            f"{system_prompt}"
        )

    llm_provider.generate_with_tools(
        prompt=user_prompt,
        functions=[edit_content_tool],
        system_prompt=system_prompt,
    )

    if expect_edit and current_content == original_content:
        logger.warning("No changes detected but edit was expected")
        raise ValueError("No changes were applied to the content")

    logger.info(f"LLM edit completed: {len(current_content)} chars")
    return current_content

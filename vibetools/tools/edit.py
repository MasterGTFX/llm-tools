"""AI-powered text editing utilities using natural language instructions."""

from typing import Optional

from vibetools.interfaces.llm import LLMInterface
from vibetools.defaults import get_default_provider
from vibetools.prompts.edit_prompts import (
    SYSTEM_PROMPT,
    user_prompt,
    custom_system_prompt,
)
from vibetools.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def ai_edit(
    original_content: str,
    instruction: str,
    llm_provider: Optional[LLMInterface] = None,
    expect_edit: bool = False,
    system_prompt: Optional[str] = None,
) -> str:
    """Generate and apply AI edit using function calling.

    Args:
        original_content: Original content to modify
        instruction: Instruction for what changes to make
        llm_provider: LLM interface for generating edits (uses global default if None)
        expect_edit: Whether to expect content changes (default: False)
        system_prompt: Custom system prompt (default: use built-in prompt)

    Returns:
        Modified content after applying AI-generated edit

    Raises:
        ValueError: If edit cannot be applied
    """
    # Use default provider if none provided
    provider = llm_provider or get_default_provider()
    
    logger.info("Starting AI edit generation and application")

    prompt = user_prompt(instruction, original_content)

    current_content = original_content
    failed_attempts = 0

    def edit_content_tool(search: str, replace: str, replace_all: bool = False) -> str:
        """Apply search-replace edit to content.

        Args:
            search: The exact text to find and replace in the content
            replace: The new text to replace the found text with
            replace_all: Whether to replace all occurrences (True) or just the first match (False)
        """
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

    def create_content_tool(content: str) -> str:
        """Create new content when starting from empty document.

        Args:
            content: The complete content to create
        """
        nonlocal current_content, failed_attempts
        
        # Safety check: warn if overwriting non-empty content
        if current_content.strip() and failed_attempts == 0:
            failed_attempts += 1
            message = f"WARNING: Document contains {len(current_content)} characters of existing content. Using create_content_tool will overwrite ALL existing content. If you want to overwrite, call create_content_tool again. If you want to modify existing content, use edit_content_tool instead."
            logger.warning("Create content tool: warning about overwriting content")
            return message
        
        current_content = content
        failed_attempts = 0
        logger.info(f"Content created: {len(content)} characters")
        return "OK: Content created successfully"

    if system_prompt is None:
        final_system_prompt = SYSTEM_PROMPT
    else:
        final_system_prompt = custom_system_prompt(system_prompt)

    provider.generate_with_tools(
        prompt=prompt,
        functions=[edit_content_tool, create_content_tool],
        system_prompt=final_system_prompt,
    )

    if expect_edit and current_content == original_content:
        logger.warning("No changes detected but edit was expected")
        raise ValueError("No changes were applied to the content")

    logger.info(f"AI edit completed: {len(current_content)} chars")
    return current_content

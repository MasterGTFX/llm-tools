"""LLM-powered filtering utilities for lists using natural language instructions."""

from typing import Any, Optional, Union

from llmtools.interfaces.llm import LLMInterface
from llmtools.defaults import get_default_provider
from llmtools.prompts.filter_prompts import (
    DOUBLE_CHECK_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    VERIFICATION_PROMPT,
    history_assistant_prompt,
    user_prompt,
)
from llmtools.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def llm_filter(
    items: list[Any],
    instruction: str,
    llm_provider: Optional[LLMInterface] = None,
    double_check: Union[bool, LLMInterface, None] = False,
) -> list[Any]:
    """Filter a list of items using natural language instructions.

    Args:
        items: List of items to filter
        instruction: Natural language instruction for filtering
        llm_provider: LLM interface for generating filtering decisions (uses global default if None)
        double_check: Verification mode - False: no verification, True: use primary provider,
                     LLMInterface: use specified provider for verification (default: False)

    Returns:
        Filtered list of items

    Raises:
        ValueError: If filtering cannot be completed
    """
    # Use default provider if none provided
    provider = llm_provider or get_default_provider()
    
    logger.info(f"Starting LLM filter with {len(items)} items")

    if not items:
        logger.info("Empty items list, returning empty list")
        return []

    # Create numbered item list for LLM
    item_map = dict(enumerate(items))
    remaining_items = item_map.copy()
    removed_items = {}

    # Format items for LLM presentation
    def format_items(item_dict: dict[int, Any]) -> str:
        return "\n".join(f"{i}: {item}" for i, item in item_dict.items())

    prompt = user_prompt(instruction, format_items(item_map))

    def remove_items(item_ids: list[int]) -> str:
        """Remove multiple items by their IDs.

        Args:
            item_ids: List of numeric IDs of items to remove from the list
        """
        nonlocal remaining_items, removed_items
        
        results = []
        success_count = 0

        for item_id in item_ids:
            if item_id not in remaining_items:
                if item_id in removed_items:
                    result = f"ERROR: Item {item_id} already removed: {removed_items[item_id]}"
                else:
                    result = f"ERROR: Item ID {item_id} does not exist"
            else:
                item_content = remaining_items[item_id]
                removed_items[item_id] = remaining_items.pop(item_id)
                logger.info(f"Removed item {item_id}: {item_content}")
                result = f"OK: Removed item {item_id}: {item_content}"
                success_count += 1
            
            results.append(result)

        summary = (
            f"Processed {len(item_ids)} items: {success_count} removed successfully"
        )
        if success_count < len(item_ids):
            summary += f", {len(item_ids) - success_count} failed"

        return f"{summary}\n" + "\n".join(results)

    def restore_items(item_ids: list[int]) -> str:
        """Restore previously removed items back to the filtered list.

        Args:
            item_ids: List of numeric IDs of previously removed items to restore
        """
        nonlocal remaining_items, removed_items
        
        results = []
        success_count = 0

        for item_id in item_ids:
            if item_id not in removed_items:
                if item_id in remaining_items:
                    result = f"ERROR: Item {item_id} is not removed: {remaining_items[item_id]}"
                else:
                    result = f"ERROR: Item ID {item_id} does not exist"
            else:
                item_content = removed_items[item_id]
                remaining_items[item_id] = removed_items.pop(item_id)
                logger.info(f"Restored item {item_id}: {item_content}")
                result = f"OK: Restored item {item_id}: {item_content}"
                success_count += 1
            
            results.append(result)

        summary = (
            f"Processed {len(item_ids)} items: {success_count} restored successfully"
        )
        if success_count < len(item_ids):
            summary += f", {len(item_ids) - success_count} failed"

        return f"{summary}\n" + "\n".join(results)

    # Use system prompt from prompts module

    try:
        provider.generate_with_tools(
            prompt=prompt,
            functions=[remove_items, restore_items],
            system_prompt=SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        raise ValueError(f"Filtering failed: {e}") from e

    # Double-check verification if requested
    if double_check and (removed_items or len(remaining_items) != len(items)):
        logger.info("Performing double-check verification")

        # Determine which provider to use for verification
        if double_check is True:
            verification_provider = llm_provider
        elif isinstance(double_check, LLMInterface):
            verification_provider = double_check
        else:
            # This shouldn't happen given the type hints, but handle gracefully
            verification_provider = llm_provider

        # Create conversation history from the original filtering
        history = [
            {
                "role": "user",
                "content": user_prompt(instruction, format_items(item_map)),
            },
            {
                "role": "assistant",
                "content": history_assistant_prompt(
                    len(remaining_items),
                    format_items(remaining_items),
                    len(removed_items),
                    format_items(removed_items),
                ),
            },
        ]

        # Use verification prompt from prompts module

        try:
            verification_provider.generate_with_tools(
                prompt=VERIFICATION_PROMPT,
                functions=[
                    remove_items,
                    restore_items,
                ],  # All functions available
                history=history,
                system_prompt=DOUBLE_CHECK_SYSTEM_PROMPT,
                tool_choice="auto"
            )
        except Exception as e:
            logger.warning(f"Double-check verification failed: {e}")

    # Return filtered results in original order
    result = [remaining_items[i] for i in sorted(remaining_items.keys())]
    logger.info(
        f"LLM filter completed: {len(result)} items remaining from {len(items)} original"
    )

    return result

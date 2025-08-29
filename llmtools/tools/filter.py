"""LLM-powered filtering utilities for lists using natural language instructions."""

from typing import Any

from llmtools.interfaces.llm import LLMInterface
from llmtools.prompts.filter_prompts import (
    SYSTEM_PROMPT,
    user_prompt,
    VERIFICATION_PROMPT,
    DOUBLE_CHECK_SYSTEM_PROMPT,
    history_assistant_prompt,
)
from llmtools.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def llm_filter(
    items: list[Any],
    instruction: str,
    llm_provider: LLMInterface,
    double_check: bool = False,
) -> list[Any]:
    """Filter a list of items using natural language instructions.

    Args:
        items: List of items to filter
        instruction: Natural language instruction for filtering
        llm_provider: LLM interface for generating filtering decisions
        double_check: Whether to verify final results with LLM (default: False)

    Returns:
        Filtered list of items

    Raises:
        ValueError: If filtering cannot be completed
    """
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

    def remove_item(item_id: int) -> str:
        """Remove a single item by ID.

        Args:
            item_id: The numeric ID of the item to remove from the list
        """
        nonlocal remaining_items, removed_items

        if item_id not in remaining_items:
            if item_id in removed_items:
                return f"ERROR: Item {item_id} already removed"
            else:
                return f"ERROR: Item ID {item_id} does not exist"

        removed_items[item_id] = remaining_items.pop(item_id)
        logger.info(f"Removed item {item_id}")
        return f"OK: Removed item {item_id}"

    def remove_items(item_ids: list[int]) -> str:
        """Remove multiple items by their IDs.

        Args:
            item_ids: List of numeric IDs of items to remove from the list
        """
        results = []
        success_count = 0

        for item_id in item_ids:
            result = remove_item(item_id)
            results.append(result)
            if result.startswith("OK:"):
                success_count += 1

        summary = (
            f"Processed {len(item_ids)} items: {success_count} removed successfully"
        )
        if success_count < len(item_ids):
            summary += f", {len(item_ids) - success_count} failed"

        return f"{summary}\n" + "\n".join(results)

    def restore_item(item_id: int) -> str:
        """Restore a previously removed item back to the filtered list.

        Args:
            item_id: The numeric ID of the previously removed item to restore
        """
        nonlocal remaining_items, removed_items

        if item_id not in removed_items:
            if item_id in remaining_items:
                return f"ERROR: Item {item_id} is not removed"
            else:
                return f"ERROR: Item ID {item_id} does not exist"

        remaining_items[item_id] = removed_items.pop(item_id)
        logger.info(f"Restored item {item_id}")
        return f"OK: Restored item {item_id}"

    # Use system prompt from prompts module

    try:
        llm_provider.generate_with_tools(
            prompt=prompt,
            functions=[remove_item, remove_items, restore_item],
            system_prompt=SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error(f"Error during filtering: {e}")
        raise ValueError(f"Filtering failed: {e}") from e

    # Double-check verification if requested
    if double_check and (removed_items or len(remaining_items) != len(items)):
        logger.info("Performing double-check verification")

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
            llm_provider.generate_with_tools(
                prompt=VERIFICATION_PROMPT,
                functions=[
                    remove_item,
                    remove_items,
                    restore_item,
                ],  # All functions available
                history=history,
                system_prompt=DOUBLE_CHECK_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.warning(f"Double-check verification failed: {e}")

    # Return filtered results in original order
    result = [remaining_items[i] for i in sorted(remaining_items.keys())]
    logger.info(
        f"LLM filter completed: {len(result)} items remaining from {len(items)} original"
    )

    return result

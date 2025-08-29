"""LLM-powered filtering utilities for lists using natural language instructions."""

from typing import Any

from llmtools.interfaces.llm import LLMInterface
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

    user_prompt = f"""<filtering_instruction>
{instruction}
</filtering_instruction>

<items_to_filter>
{format_items(item_map)}
</items_to_filter>

Use the provided tools to remove items that should be filtered out according to the instruction."""

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

    system_prompt = """You are an expert data filter. Use the provided tools to remove items that should be filtered out according to the given instruction.

Guidelines:
- Use remove_item() to remove individual items by ID
- Use remove_items() to remove multiple items at once
- Use restore_item() if you need to undo a removal
- Only remove items that clearly match the filtering criteria
- When in doubt, keep the item (don't remove it)"""

    try:
        llm_provider.generate_with_tools(
            prompt=user_prompt,
            functions=[remove_item, remove_items, restore_item],
            system_prompt=system_prompt,
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
                "content": f"Filter these items: {instruction}\n\nItems:\n{format_items(item_map)}",
            },
            {
                "role": "assistant",
                "content": f"I've filtered the items as requested. Here are the results:\n\nREMAINING ITEMS ({len(remaining_items)}):\n{format_items(remaining_items)}\n\nREMOVED ITEMS ({len(removed_items)}):\n{format_items(removed_items)}",
            },
        ]

        verification_prompt = "Please double-check this filtering result against the original instruction. Review both the remaining and removed items carefully:\n\n1. Are there any remaining items that should actually be removed?\n2. Are there any removed items that should actually be kept?\n\nMake any necessary corrections using the available tools (only when needed)."

        try:
            llm_provider.generate_with_tools(
                prompt=verification_prompt,
                functions=[
                    remove_item,
                    remove_items,
                    restore_item,
                ],  # All functions available
                history=history,
                system_prompt="You are double-checking your filtering work. Review the results carefully and make corrections if needed. Only make changes if you're confident they improve the accuracy of the filtering.",
            )
        except Exception as e:
            logger.warning(f"Double-check verification failed: {e}")

    # Return filtered results in original order
    result = [remaining_items[i] for i in sorted(remaining_items.keys())]
    logger.info(
        f"LLM filter completed: {len(result)} items remaining from {len(items)} original"
    )

    return result

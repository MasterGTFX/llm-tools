"""LLM-powered sorting utilities for lists using natural language instructions."""

from typing import Any, Union

from llmtools.interfaces.llm import LLMInterface
from llmtools.prompts.sorter_prompts import (
    DOUBLE_CHECK_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    VERIFICATION_PROMPT,
    history_assistant_prompt,
    user_prompt,
)
from llmtools.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def llm_sorter(
    items: list[Any],
    instruction: str,
    llm_provider: LLMInterface,
    double_check: Union[bool, LLMInterface, None] = False,
) -> list[Any]:
    """Sort a list of items using natural language instructions.

    Args:
        items: List of items to sort
        instruction: Natural language instruction for sorting
        llm_provider: LLM interface for generating sorting decisions
        double_check: Verification mode - False: no verification, True: use primary provider,
                     LLMInterface: use specified provider for verification (default: False)

    Returns:
        Sorted list of items

    Raises:
        ValueError: If sorting cannot be completed
    """
    logger.info(f"Starting LLM sorter with {len(items)} items")

    if not items:
        logger.info("Empty items list, returning empty list")
        return []

    # Create persistent ID mapping and current order tracking
    item_map = dict(enumerate(items))
    current_order = list(range(len(items)))

    # Format items for LLM presentation
    def format_items_with_order() -> str:
        return "\n".join(f"{item_id}: {item_map[item_id]}" for item_id in current_order)

    prompt = user_prompt(instruction, format_items_with_order())

    def move_items_to(moves: list[tuple[int, int]]) -> str:
        """Move items to absolute positions.

        Args:
            moves: List of tuples (item_id, target_position) where target_position is 0-indexed
        """
        nonlocal current_order

        if not moves:
            return "ERROR: No moves specified"

        results = []
        valid_moves = []

        # Validate all moves first
        for item_id, position in moves:
            if item_id not in item_map:
                results.append(f"ERROR: Item ID {item_id} does not exist")
                continue
            if not (0 <= position < len(items)):
                results.append(
                    f"ERROR: Position {position} out of bounds (0-{len(items) - 1}) for item {item_id}: {item_map[item_id]}"
                )
                continue
            valid_moves.append((item_id, position))

        if not valid_moves:
            return "\n".join(results)

        # Create new order by processing moves
        new_order = current_order.copy()

        # Remove items that will be moved
        items_to_move = dict(valid_moves)
        new_order = [item_id for item_id in new_order if item_id not in items_to_move]

        # Insert items at their target positions
        for item_id, position in sorted(valid_moves, key=lambda x: x[1]):
            # Adjust position if it's beyond current length
            insert_pos = min(position, len(new_order))
            new_order.insert(insert_pos, item_id)

        current_order = new_order
        success_count = len(valid_moves)

        for item_id, position in valid_moves:
            item_content = item_map[item_id]
            results.append(f"OK: Moved item {item_id} to position {position}: {item_content}")
            logger.info(f"Moved item {item_id} to position {position}: {item_content}")

        summary = f"Successfully moved {success_count} items"
        if len(moves) > success_count:
            summary += f", {len(moves) - success_count} failed"

        return f"{summary}\n" + "\n".join(results)

    def move_items_by(moves: list[tuple[int, int]]) -> str:
        """Move items by relative offset.

        Args:
            moves: List of tuples (item_id, diff) where diff is position change (positive = forward, negative = backward)
        """
        nonlocal current_order

        if not moves:
            return "ERROR: No moves specified"

        results = []
        valid_moves = []

        # Validate all moves and calculate target positions
        for item_id, diff in moves:
            if item_id not in item_map:
                results.append(f"ERROR: Item ID {item_id} does not exist")
                continue

            try:
                current_pos = current_order.index(item_id)
                target_pos = max(0, min(current_pos + diff, len(items) - 1))
                valid_moves.append((item_id, current_pos, target_pos, diff))
                item_content = item_map[item_id]
                results.append(
                    f"OK: Will move item {item_id} by {diff:+d} (position {current_pos} → {target_pos}): {item_content}"
                )
                logger.info(f"Moving item {item_id} by {diff:+d} positions: {item_content}")
            except ValueError:
                results.append(f"ERROR: Item ID {item_id} not found in current order: {item_map.get(item_id, 'unknown')}")

        if not valid_moves:
            return "\n".join(results)

        # Apply moves by converting to absolute positions and using move_items_to logic
        absolute_moves = [
            (item_id, target_pos) for item_id, _, target_pos, _ in valid_moves
        ]

        # Create new order
        new_order = current_order.copy()
        items_to_move = dict(absolute_moves)
        new_order = [item_id for item_id in new_order if item_id not in items_to_move]

        # Insert items at their target positions
        for item_id, target_pos in sorted(absolute_moves, key=lambda x: x[1]):
            insert_pos = min(target_pos, len(new_order))
            new_order.insert(insert_pos, item_id)

        current_order = new_order

        summary = f"Successfully moved {len(valid_moves)} items by relative offset"
        return f"{summary}\n" + "\n".join(results)

    def show_modified_order() -> str:
        """Display current item order with persistent IDs. Use this to check your progress after making moves.

        Returns:
            Current order formatted as "id: content" with each item on a separate line
        """
        order_display = format_items_with_order()
        logger.info("Displayed modified order")
        return f"Current order:\n{order_display}"

    def set_complete_order(order: list[int]) -> str:
        """Set the complete order by providing item IDs in desired sequence.

        Args:
            order: List of item IDs in the desired final order. Missing IDs will be appended automatically.

        Returns:
            Status message indicating success, warnings about missing/invalid IDs
        """
        nonlocal current_order

        if not order:
            return "ERROR: Empty order provided"

        # Validate IDs and track which ones are provided
        valid_ids = []
        invalid_ids = []
        seen_ids = set()

        for item_id in order:
            if item_id in item_map:
                if item_id not in seen_ids:
                    valid_ids.append(item_id)
                    seen_ids.add(item_id)
                # Silently ignore duplicates
            else:
                invalid_ids.append(item_id)

        # Find missing IDs (ones that exist but weren't specified)
        missing_ids = [item_id for item_id in current_order if item_id not in seen_ids]

        # Build the new order: specified valid IDs + missing IDs in original relative order
        current_order = valid_ids + missing_ids

        # Prepare result message
        result = f"OK: Set complete order with {len(valid_ids)} specified items"
        if missing_ids:
            result += f", {len(missing_ids)} unspecified items added at end"
        if invalid_ids:
            result += f", {len(invalid_ids)} invalid IDs ignored: {invalid_ids}"

        logger.info(f"Set complete order: {len(valid_ids)} positioned, {len(missing_ids)} appended, {len(invalid_ids)} invalid")
        return result

    # Use system prompt from prompts module
    try:
        llm_provider.generate_with_tools(
            prompt=prompt,
            functions=[move_items_to, move_items_by, show_modified_order, set_complete_order],
            system_prompt=SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error(f"Error during sorting: {e}")
        raise ValueError(f"Sorting failed: {e}") from e

    # Double-check verification if requested
    if double_check:
        logger.info("Performing double-check verification")

        # Determine which provider to use for verification
        if double_check is True:
            verification_provider = llm_provider
        elif isinstance(double_check, LLMInterface):
            verification_provider = double_check
        else:
            # This shouldn't happen given the type hints, but handle gracefully
            verification_provider = llm_provider

        final_order = format_items_with_order()

        # Create conversation history from the original sorting
        history = [
            {
                "role": "user",
                "content": user_prompt(instruction, format_items_with_order()),
            },
            {
                "role": "assistant",
                "content": history_assistant_prompt(final_order),
            },
        ]

        # Use verification prompt from prompts module
        try:
            verification_provider.generate_with_tools(
                prompt=VERIFICATION_PROMPT,
                functions=[
                    move_items_to,
                    move_items_by,
                    show_modified_order,
                    # set_complete_order -> is intentionally omitted to prevent wholesale reordering
                ],  # All functions available
                history=history,
                system_prompt=DOUBLE_CHECK_SYSTEM_PROMPT,
                tool_choice="auto"
            )
        except Exception as e:
            logger.warning(f"Double-check verification failed: {e}")

    # Return sorted results maintaining original order of IDs but in new sequence
    result = [item_map[item_id] for item_id in current_order]
    logger.info(
        f"LLM sorter completed: {len(result)} items sorted from original {len(items)}"
    )

    return result

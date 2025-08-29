"""LLM-powered sorting utilities for lists using natural language instructions."""

from typing import Any

from llmtools.interfaces.llm import LLMInterface
from llmtools.prompts.sorter_prompts import (
    SYSTEM_PROMPT,
    user_prompt,
    VERIFICATION_PROMPT,
    DOUBLE_CHECK_SYSTEM_PROMPT,
    history_assistant_prompt,
)
from llmtools.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def llm_sorter(
    items: list[Any],
    instruction: str,
    llm_provider: LLMInterface,
    double_check: bool = False,
) -> list[Any]:
    """Sort a list of items using natural language instructions.

    Args:
        items: List of items to sort
        instruction: Natural language instruction for sorting
        llm_provider: LLM interface for generating sorting decisions
        double_check: Whether to verify final results with LLM (default: False)

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
                results.append(f"ERROR: Position {position} out of bounds (0-{len(items)-1})")
                continue
            valid_moves.append((item_id, position))

        if not valid_moves:
            return "\n".join(results)

        # Create new order by processing moves
        new_order = current_order.copy()
        
        # Remove items that will be moved
        items_to_move = {item_id: position for item_id, position in valid_moves}
        new_order = [item_id for item_id in new_order if item_id not in items_to_move]
        
        # Insert items at their target positions
        for item_id, position in sorted(valid_moves, key=lambda x: x[1]):
            # Adjust position if it's beyond current length
            insert_pos = min(position, len(new_order))
            new_order.insert(insert_pos, item_id)
        
        current_order = new_order
        success_count = len(valid_moves)
        
        for item_id, position in valid_moves:
            results.append(f"OK: Moved item {item_id} to position {position}")
            logger.info(f"Moved item {item_id} to position {position}")

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
                results.append(f"OK: Will move item {item_id} by {diff:+d} (position {current_pos} → {target_pos})")
                logger.info(f"Moving item {item_id} by {diff:+d} positions")
            except ValueError:
                results.append(f"ERROR: Item ID {item_id} not found in current order")

        if not valid_moves:
            return "\n".join(results)

        # Apply moves by converting to absolute positions and using move_items_to logic
        absolute_moves = [(item_id, target_pos) for item_id, _, target_pos, _ in valid_moves]
        
        # Create new order
        new_order = current_order.copy()
        items_to_move = {item_id: target_pos for item_id, target_pos in absolute_moves}
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

    # Use system prompt from prompts module
    try:
        llm_provider.generate_with_tools(
            prompt=prompt,
            functions=[move_items_to, move_items_by, show_modified_order],
            system_prompt=SYSTEM_PROMPT,
        )
    except Exception as e:
        logger.error(f"Error during sorting: {e}")
        raise ValueError(f"Sorting failed: {e}") from e

    # Double-check verification if requested
    if double_check:
        logger.info("Performing double-check verification")

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
            llm_provider.generate_with_tools(
                prompt=VERIFICATION_PROMPT,
                functions=[
                    move_items_to,
                    move_items_by,
                    show_modified_order,
                ],  # All functions available
                history=history,
                system_prompt=DOUBLE_CHECK_SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.warning(f"Double-check verification failed: {e}")

    # Return sorted results maintaining original order of IDs but in new sequence
    result = [item_map[item_id] for item_id in current_order]
    logger.info(
        f"LLM sorter completed: {len(result)} items sorted from original {len(items)}"
    )

    return result
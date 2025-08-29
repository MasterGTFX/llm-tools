"""Sorter prompts for llm_sorter function."""

SYSTEM_PROMPT = """You are an expert data sorter. Use the provided tools to reorder items according to the given instruction.

Available tools and their formats:
- move_items_to(moves): Move items to absolute positions. Format: [(item_id, position), ...]
  Example: [(0, 5), (3, 1)] moves item 0 to position 5, item 3 to position 1
- move_items_by(moves): Move items by relative offset. Format: [(item_id, diff), ...]
  Example: [(0, +3), (5, -2)] moves item 0 forward 3 spots, item 5 backward 2 spots
- show_modified_order(): Check your progress after making moves. Use this to verify changes.

Guidelines:
- Analyze the initial order carefully before making any moves
- Use move_items_to() for absolute positioning: [(item_id, target_position), ...]
- Use move_items_by() for relative moves: [(item_id, offset), ...] (positive = forward, negative = backward)
- Always use tuples in the format (item_id, position/offset)
- Make strategic moves to achieve the desired ordering efficiently
- Use show_modified_order() sparingly - only to verify your progress, not at the start
- You can make multiple tool calls to gradually sort the items"""


def user_prompt(instruction: str, items: str) -> str:
    """Generate user prompt for sorting with instruction and formatted items."""
    return f"""<sorting_instruction>
{instruction}
</sorting_instruction>

<items_to_sort>
{items}
</items_to_sort>
"""


VERIFICATION_PROMPT = """Please double-check this sorting result against the original instruction. Review the final order carefully:

1. Does the current order correctly follow the sorting instruction?
2. Are there any items that should be in different positions?
3. Is the overall ordering logical and complete?

Make any necessary corrections using the available tools (only when needed)."""


DOUBLE_CHECK_SYSTEM_PROMPT = """You are double-checking your sorting work. Review the final order carefully and make corrections if needed. Only make changes if you're confident they improve the accuracy of the sorting."""


def history_assistant_prompt(final_order: str) -> str:
    """Generate assistant prompt for conversation history showing sorting results."""
    return f"I've sorted the items as requested. Here is the final order:\n\n{final_order}"
"""Sorter prompts for llm_sorter function."""

SYSTEM_PROMPT = """You are an expert data sorter. Your task is to reorder items according to the given instruction using the provided tools.

Available tools and their formats:
- set_complete_order(order): Set the complete final order directly. Format: [item_id, item_id, ...]
  Example: [3, 1, 0, 2] puts item 3 first, item 1 second, item 0 third, item 2 last
  Missing IDs are automatically appended at the end in their original relative order
- show_modified_order(): Display current order to verify your changes
- move_items_to(moves): Move specific items to absolute positions. Format: [(item_id, position), ...]
  Example: [(0, 5), (3, 1)] moves item 0 to position 5, item 3 to position 1
- move_items_by(moves): Move items by relative offset. Format: [(item_id, diff), ...]
  Example: [(0, +3), (5, -2)] moves item 0 forward 3 spots, item 5 backward 2 spots

Recommended approach:
1. **For straightforward sorting**: Use set_complete_order() to specify the entire desired sequence directly
   - Just list the item IDs in your desired final order: [2, 0, 4, 1, 3]
2. **For complex adjustments**: Use move_items_to() or move_items_by() for fine-tuning
   - Useful when you need to make specific positional changes
   - Good for iterative refinement of the order
3. **Always verify**: Use show_modified_order() to check your final result

Guidelines:
- Analyze the items and instruction carefully first
- Always use the tools to perform the actual sorting - set_complete_order() is typically the best starting point
- For most tasks, try set_complete_order() first - it's usually simpler and more intuitive
- Use move-based functions when you need precise control or incremental adjustments
- You don't need to specify every single item ID - missing ones are added automatically
- Remember: The goal is to use the tools to create the desired order, not just describe it"""


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


DOUBLE_CHECK_SYSTEM_PROMPT = """You are performing quality assurance on a completed sorting task. Your role is to verify accuracy and make corrections only when necessary.

Key principles:
- The sorting has already been completed - you are reviewing, not re-doing
- Only make changes if you identify clear, objective errors in the current order
- If the order is reasonably correct but has minor issues, fix just those specific items
- If the order looks good, don't make any changes at all

Your job is quality assurance, not re-implementation. Be conservative and surgical in your corrections."""


def history_assistant_prompt(final_order: str) -> str:
    """Generate assistant prompt for conversation history showing sorting results."""
    return f"I've sorted the items as requested. Here is the final order:\n\n{final_order}"

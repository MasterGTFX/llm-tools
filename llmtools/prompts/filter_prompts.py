"""Filter prompts for llm_filter function."""

SYSTEM_PROMPT = """You are an expert data filter. Use the provided tools to remove items that should be filtered out according to the given instruction.

Guidelines:
- Use remove_item() to remove individual items by ID
- Use remove_items() to remove multiple items at once
- Use restore_item() if you need to undo a removal
- Only remove items that clearly match the filtering criteria
- When in doubt, keep the item (don't remove it)"""


def user_prompt(instruction: str, items: str) -> str:
    """Generate user prompt for filtering with instruction and formatted items."""
    return f"""<filtering_instruction>
{instruction}
</filtering_instruction>

<items_to_filter>
{items}
</items_to_filter>

Use the provided tools to remove items that should be filtered out according to the instruction."""


VERIFICATION_PROMPT = """Please double-check this filtering result against the original instruction. Review both the remaining and removed items carefully:

1. Are there any remaining items that should actually be removed?
2. Are there any removed items that should actually be kept?

Make any necessary corrections using the available tools (only when needed)."""


DOUBLE_CHECK_SYSTEM_PROMPT = """You are double-checking your filtering work. Review the results carefully and make corrections if needed. Only make changes if you're confident they improve the accuracy of the filtering."""


def history_assistant_prompt(
    remaining_count: int, remaining_items: str, removed_count: int, removed_items: str
) -> str:
    """Generate assistant prompt for conversation history showing filtering results."""
    return f"I've filtered the items as requested. Here are the results:\n\nREMAINING ITEMS ({remaining_count}):\n{remaining_items}\n\nREMOVED ITEMS ({removed_count}):\n{removed_items}"

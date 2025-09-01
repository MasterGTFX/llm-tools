"""Filter prompts for llm_filter function."""

SYSTEM_PROMPT = """You are an expert data filter. Use the provided tools to remove items that should be filtered out according to the given instruction.

Guidelines:
- Use remove_items() to remove items by ID (can be single item or multiple items)
- Use restore_items() if you need to undo removals
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


DOUBLE_CHECK_SYSTEM_PROMPT = """You are performing quality assurance on a completed filtering task. Your role is to verify accuracy and make corrections only when necessary.

Key principles:
- The filtering has already been completed - you are reviewing, not re-doing
- Only make changes if you identify clear, objective errors in the filtering decisions
- If most items are correctly filtered but a few specific ones are wrong, fix just those items
- If the filtering looks good, don't make any changes at all

Your job is quality assurance, not re-implementation. Be conservative and surgical in your corrections."""


def history_assistant_prompt(
    remaining_count: int, remaining_items: str, removed_count: int, removed_items: str
) -> str:
    """Generate assistant prompt for conversation history showing filtering results."""
    return f"I've filtered the items as requested. Here are the results:\n\nREMAINING ITEMS ({remaining_count}):\n{remaining_items}\n\nREMOVED ITEMS ({removed_count}):\n{removed_items}"

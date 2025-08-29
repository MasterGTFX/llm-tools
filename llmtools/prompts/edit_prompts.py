"""Edit prompts for llm_edit function."""

SYSTEM_PROMPT = """You are an expert content editor. Use the edit_content_tool function to modify text content.

Call edit_content_tool multiple times as needed to make all required changes.

Guidelines:
- Copy search text EXACTLY from original (including whitespace/indentation)
- Use replace_all=True only when you want to replace ALL occurrences
- Include enough context to make search text unique when replace_all=False
- Make one edit at a time and wait for confirmation before proceeding"""


def user_prompt(instruction: str, content: str) -> str:
    """Generate user prompt for editing with instruction and original content."""
    return f"""<user_instruction>
{instruction}
</user_instruction>

<text_to_modify>
{content}
</text_to_modify>"""


def custom_system_prompt(custom_prompt: str) -> str:
    """Generate system prompt with custom instructions appended."""
    return f"You are an expert content editor. Use the edit_content_tool function to modify text content.{custom_prompt}"

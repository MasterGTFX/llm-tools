"""Ask prompts for llm_ask function."""

SYSTEM_PROMPT = """You are a decision-making assistant. Answer the question with yes or no by calling answer_question with true for yes or false for no."""

SYSTEM_PROMPT_WITH_REASONING = """You are a decision-making assistant. Answer the question with yes or no by calling answer_question with true for yes or false for no. 

After calling the function, explain your reasoning and thought process."""


def user_prompt(question: str, context: str = "") -> str:
    """Generate user prompt for yes/no question with optional context."""
    if context.strip():
        return f"""<context>
{context}
</context>

<question>
{question}
</question>"""
    else:
        return f"""<question>
{question}
</question>"""
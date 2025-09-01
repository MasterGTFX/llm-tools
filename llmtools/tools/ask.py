"""LLM-powered yes/no question answering using natural language instructions."""

from typing import Optional, Union

from llmtools.interfaces.llm import LLMInterface
from llmtools.defaults import get_default_provider
from llmtools.prompts.ask_prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_WITH_REASONING,
    user_prompt,
)
from llmtools.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def llm_ask(
    question: str,
    llm_provider: Optional[LLMInterface] = None,
    context: str = "",
    reasoning: bool = False,
) -> Union[bool, tuple[bool, str]]:
    """Ask the LLM a yes/no question and get a boolean answer.

    Args:
        question: The yes/no question to ask
        llm_provider: LLM interface for generating the answer (uses global default if None)
        context: Optional context information to inform the decision
        reasoning: Whether to return reasoning along with the answer (default: False)

    Returns:
        Boolean answer if reasoning=False, or (bool, str) tuple if reasoning=True

    Raises:
        ValueError: If the LLM cannot provide a clear yes/no answer
    """
    # Use default provider if none provided
    provider = llm_provider or get_default_provider()
    
    logger.info(f"Starting LLM ask: {question[:50]}...")

    prompt = user_prompt(question, context)
    answer_given = False
    final_answer = None

    def answer_question(answer: bool) -> str:
        """Answer the question with yes (true) or no (false).

        Args:
            answer: True for yes, False for no
        """
        nonlocal answer_given, final_answer
        
        if answer_given:
            logger.warning("Multiple answers provided, using first answer")
            return "WARNING: Answer already provided. Using first answer."
        
        answer_given = True
        final_answer = answer
        answer_text = "yes" if answer else "no"
        logger.info(f"LLM answered: {answer_text}")
        return f"OK: Answered {answer_text}"

    # Choose system prompt based on reasoning requirement
    system_prompt = SYSTEM_PROMPT_WITH_REASONING if reasoning else SYSTEM_PROMPT

    try:
        response = provider.generate_with_tools(
            prompt=prompt,
            functions=[answer_question],
            system_prompt=system_prompt,
        )
    except Exception as e:
        logger.error(f"Error during question answering: {e}")
        raise ValueError(f"Failed to get answer: {e}") from e

    if not answer_given or final_answer is None:
        logger.error("LLM did not provide an answer")
        raise ValueError("LLM failed to provide a yes/no answer")

    if reasoning:
        reasoning_text = response.strip() if response else ""
        logger.info(f"LLM ask completed with reasoning: {len(reasoning_text)} chars")
        return (final_answer, reasoning_text)
    else:
        logger.info("LLM ask completed")
        return final_answer
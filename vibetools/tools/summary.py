"""AI-powered document summarization using iterative summary updates."""

from typing import Optional, Union

from vibetools.interfaces.llm import LLMInterface
from vibetools.defaults import get_default_provider
from vibetools.tools.edit import ai_edit
from vibetools.prompts.summary_prompts import (
    SYSTEM_PROMPT,
    initial_summary_prompt,
    update_instruction_template,
)
from vibetools.utils.logger_config import setup_logger

logger = setup_logger(__name__)


def ai_summary(
    documents: list[str],
    instruction: str,
    llm_provider: Optional[LLMInterface] = None,
    initial_summary: Optional[str] = None,
    return_all: bool = False,
) -> Union[str, list[str]]:
    """Create and iteratively update a summary from multiple documents.

    Args:
        documents: List of document strings to summarize
        instruction: Natural language instruction for summarization approach
        llm_provider: LLM interface for generating summary updates (uses global default if None)
        initial_summary: Optional starting summary (if None, starts empty)
        return_all: If True, returns list of all summary versions, else final summary

    Returns:
        Final consolidated summary as string, or list of all versions if return_all_versions=True

    Raises:
        ValueError: If summarization cannot be completed
    """
    # Use default provider if none provided
    provider = llm_provider or get_default_provider()
    
    logger.info(f"Starting AI summary with {len(documents)} documents")

    # Handle empty documents case - return initial summary if provided
    if not documents:
        logger.info("Empty documents list")
        result = initial_summary or ""
        return [result] if return_all else result

    # Initialize summary and version tracking
    summary = initial_summary or ""
    versions = [summary] if summary else []

    # Process each document
    for i, document in enumerate(documents):
        logger.info(f"Processing document {i+1}/{len(documents)}")
        
        # Choose prompt based on whether summary exists
        prompt = (initial_summary_prompt(instruction, document) if not summary 
                 else update_instruction_template(instruction, document))
        
        try:
            summary = ai_edit(
                original_content=summary,
                instruction=prompt,
                llm_provider=provider,
                system_prompt=SYSTEM_PROMPT,
            )
            versions.append(summary)
            logger.info(f"Successfully processed document {i+1}")
        except Exception as e:
            logger.error(f"Error processing document {i+1}: {e}")
            raise ValueError(f"Failed to process document {i+1}: {e}") from e

    logger.info(f"AI summary completed: {len(summary)} characters, {len(versions)} versions")
    return versions if return_all else summary
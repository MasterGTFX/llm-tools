"""Sorter implementation for LLM-based list sorting and filtering."""

import json
from typing import Any, Literal, Optional, Union

from llmtools.config import LLMConfig, SorterConfig
from llmtools.interfaces.llm import LLMInterface
from llmtools.utils.logging import setup_logger


class Sorter:
    """Sort or filter a Python list based on an instruction using an LLM.

    Supports two modes:
    - Strict mode: Output has the same length and content as input, only re-ordered
    - Filter mode: Output is a subset of items that satisfy certain conditions
    """

    def __init__(
        self,
        mode: Literal["strict", "filter"] = "strict",
        config: Optional[Union[dict[str, Any], SorterConfig]] = None,
        llm_provider: Optional[LLMInterface] = None,
    ):
        """Initialize Sorter.

        Args:
            mode: Sorting mode - "strict" or "filter"
            config: Configuration dictionary or SorterConfig object
            llm_provider: LLM provider instance (if None, will create from config)
        """
        # Handle config initialization
        if isinstance(config, dict):
            # Convert dict config to proper config objects
            llm_config = LLMConfig(**config)
            self.config = SorterConfig(
                llm=llm_config, mode=mode, max_retries=3, validate_output=True
            )
        elif isinstance(config, SorterConfig):
            self.config = config
            self.config.mode = mode  # Override mode if provided
        else:
            # Default config
            default_llm = LLMConfig(
                provider="openai",
                model=None,
                api_key=None,
                base_url=None,
                temperature=0.7,
                max_tokens=None,
                timeout=30,
            )
            self.config = SorterConfig(
                llm=default_llm, mode=mode, max_retries=3, validate_output=True
            )

        self.llm_provider = llm_provider

        # Set up logging
        self.logger = setup_logger(
            __name__,
            level=self.config.logging.level,
            format_string=self.config.logging.format,
            handler_type=self.config.logging.handler_type,
        )
        self.logger.debug(f"Sorter initialized in {self.config.mode} mode")

    def sort(self, items: list[Any], instruction: str) -> list[Any]:
        """Sort or filter items based on the given instruction.

        Args:
            items: List of items to sort/filter
            instruction: Natural language instruction for sorting/filtering

        Returns:
            Sorted/filtered list based on the mode and instruction
        """
        self.logger.info(
            f"Starting {self.config.mode} operation with {len(items)} items: '{instruction[:50]}{'...' if len(instruction) > 50 else ''}'"
        )

        if not items:
            self.logger.info("Empty input list, returning empty result")
            return []

        if not self.llm_provider:
            self.logger.error("LLM provider not configured")
            raise ValueError("LLM provider not configured. Cannot sort items.")

        self.logger.debug(f"Input items: {items[:5]}{'...' if len(items) > 5 else ''}")

        # Create the sorting schema for structured output
        if self.config.mode == "strict":
            schema = {
                "type": "object",
                "properties": {
                    "sorted_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"All {len(items)} items reordered according to instruction",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the sorting logic",
                    },
                },
                "required": ["sorted_items", "reasoning"],
            }
        else:  # filter mode
            schema = {
                "type": "object",
                "properties": {
                    "filtered_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Subset of items that match the filtering criteria",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of the filtering logic",
                    },
                },
                "required": ["filtered_items", "reasoning"],
            }

        # Convert items to strings for LLM processing
        str_items = [str(item) for item in items]
        items_json = json.dumps(str_items, indent=2)

        system_prompt = self._get_system_prompt()
        prompt = f"""
        Items to process:
        {items_json}

        Instruction: {instruction}

        Mode: {self.config.mode}
        {"Requirement: Return ALL items in the new order - no items should be missing." if self.config.mode == "strict" else "Requirement: Return only items that satisfy the criteria."}
        """

        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.debug(
                    f"Attempt {attempt + 1} of {self.config.max_retries + 1}"
                )

                # Get structured response from LLM
                response = self.llm_provider.generate_structured(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=system_prompt,
                    temperature=self.config.llm.temperature,
                )
                self.logger.debug(f"Received LLM response: {response}")

                # Extract the result
                if self.config.mode == "strict":
                    result_items = response.get("sorted_items", [])
                else:
                    result_items = response.get("filtered_items", [])

                # Validate output if enabled
                if self.config.validate_output:
                    self.logger.debug("Validating output")
                    self._validate_result(items, result_items, str_items)
                    self.logger.debug("Output validation passed")

                # Convert back to original types if possible
                final_result = self._restore_types(result_items, items)
                self.logger.info(
                    f"Successfully completed {self.config.mode} operation. Output: {len(final_result)} items"
                )
                self.logger.debug(
                    f"Final result: {final_result[:5]}{'...' if len(final_result) > 5 else ''}"
                )
                return final_result

            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries:
                    self.logger.error(
                        f"All {self.config.max_retries + 1} attempts failed"
                    )
                    raise RuntimeError(
                        f"Failed to sort items after {self.config.max_retries + 1} attempts: {e}"
                    ) from e

        # Fallback - return original list
        self.logger.warning("Returning original list as fallback")
        return items

    def _get_system_prompt(self) -> str:
        """Get the system prompt based on the sorting mode."""
        if self.config.mode == "strict":
            return (
                "You are a precise list sorter. You must return ALL items from the input "
                "in a new order based on the given instruction. Never add, remove, or "
                "modify items - only reorder them. The output must contain exactly the "
                "same items as the input, just in a different sequence."
            )
        else:
            return (
                "You are a list filter. Return only the items from the input that "
                "satisfy the given criteria. You may return fewer items than the input, "
                "but never add new items or modify existing ones. Only include items "
                "that clearly match the filtering instruction."
            )

    def _validate_result(
        self, original_items: list[Any], result_items: list[str], str_items: list[str]
    ) -> None:
        """Validate the sorting/filtering result.

        Args:
            original_items: Original input items
            result_items: LLM output items as strings
            str_items: Original items converted to strings
        """
        if self.config.mode == "strict":
            # Check that all items are present and no extras
            if len(result_items) != len(original_items):
                raise ValueError(
                    f"Strict mode violation: expected {len(original_items)} items, "
                    f"got {len(result_items)}"
                )

            # Check that all items from input are present
            result_set = set(result_items)
            input_set = set(str_items)
            if result_set != input_set:
                missing = input_set - result_set
                extra = result_set - input_set
                raise ValueError(
                    f"Strict mode violation: missing items {missing}, extra items {extra}"
                )
        else:
            # Filter mode: check that result is subset of input
            result_set = set(result_items)
            input_set = set(str_items)
            if not result_set.issubset(input_set):
                extra = result_set - input_set
                raise ValueError(
                    f"Filter mode violation: extra items not in input: {extra}"
                )

    def _restore_types(
        self, str_result: list[str], original_items: list[Any]
    ) -> list[Any]:
        """Attempt to restore original types from string results.

        Args:
            str_result: String results from LLM
            original_items: Original items with their types

        Returns:
            List with original types restored where possible
        """
        # Create mapping from string representation to original item
        str_to_original = {}
        for item in original_items:
            str_to_original[str(item)] = item

        # Restore types
        restored = []
        for str_item in str_result:
            if str_item in str_to_original:
                restored.append(str_to_original[str_item])
            else:
                # Fallback to string if we can't find the original
                restored.append(str_item)

        return restored

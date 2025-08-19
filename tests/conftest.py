"""Pytest configuration and fixtures for llmtools tests."""

from typing import Any, Optional, Union

import pytest

from llmtools.interfaces.llm import LLMInterface


class TestLLM(LLMInterface):
    """Simple test LLM implementation."""

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        **kwargs: Any,
    ) -> str:
        return "Test knowledge base content"

    def generate_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if "sorted_items" in str(schema):
            return {"sorted_items": ["apple", "banana", "pear"], "reasoning": "Alphabetical"}
        return {"filtered_items": ["apple"], "reasoning": "Contains 'a'"}

    def generate_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        history: Optional[list[dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Union[str, dict[str, Any]]:
        return "Test tool response"

    def configure(self, config: dict[str, Any]) -> None:
        pass


@pytest.fixture
def test_llm():
    """Provide a simple test LLM."""
    return TestLLM()


@pytest.fixture
def sample_docs(tmp_path):
    """Create test documents."""
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("Document one content")
    doc2 = tmp_path / "doc2.txt"
    doc2.write_text("Document two content")
    return [str(doc1), str(doc2)]


@pytest.fixture
def output_dir(tmp_path):
    """Provide test output directory."""
    return tmp_path / "output"

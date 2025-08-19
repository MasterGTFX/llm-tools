"""Pytest configuration and fixtures for llmtools tests."""

from typing import Any, Dict, List, Union

import pytest

from llmtools.interfaces.llm import LLMInterface


class MockLLMProvider(LLMInterface):
    """Mock LLM provider for testing."""

    def __init__(self):
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        **kwargs: Any,
    ) -> str:
        """Mock generate method that returns predictable responses."""
        self.call_history.append(
            {
                "method": "generate",
                "prompt": prompt,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )

        # Return predictable responses based on prompt content
        if "sort" in prompt.lower() and "alphabetically" in prompt.lower():
            return "apple\nbanana\norange\npear"
        elif "knowledge base" in prompt.lower():
            return "This is a comprehensive knowledge base containing information from the provided documents."
        else:
            return "Mock LLM response"

    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: str = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Mock structured generation."""
        self.call_history.append(
            {
                "method": "generate_structured",
                "prompt": prompt,
                "schema": schema,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )

        # Return mock structured responses based on schema
        if "sorted_items" in str(schema):
            # Mock sorting response
            return {
                "sorted_items": ["apple", "banana", "orange", "pear"],
                "reasoning": "Sorted alphabetically",
            }
        elif "filtered_items" in str(schema):
            # Mock filtering response
            return {
                "filtered_items": ["apple", "banana", "orange"],
                "reasoning": "Items containing letter 'a'",
            }
        else:
            return {"result": "mock structured response"}

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: str = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Mock tool calling."""
        self.call_history.append(
            {
                "method": "generate_with_tools",
                "prompt": prompt,
                "tools": tools,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        return "Mock tool response"


@pytest.fixture
def mock_llm():
    """Provide a mock LLM provider for testing."""
    return MockLLMProvider()


@pytest.fixture
def sample_documents(tmp_path):
    """Create sample documents for testing."""
    doc1 = tmp_path / "doc1.txt"
    doc1.write_text("This is the first document with some content.")

    doc2 = tmp_path / "doc2.txt"
    doc2.write_text("This is the second document with different information.")

    return [str(doc1), str(doc2)]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary output directory for testing."""
    return tmp_path / "output"

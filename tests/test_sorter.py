"""Tests for the Sorter component."""

import pytest

from llmtools.sorter import Sorter


def test_sorter_initialization():
    """Test Sorter initialization with different configurations."""
    # Default initialization
    sorter = Sorter(mode="strict")
    assert sorter.config.mode == "strict"

    # With dict config
    config = {"provider": "openai", "model": "gpt-4"}
    sorter = Sorter(mode="filter", config=config)
    assert sorter.config.mode == "filter"
    assert sorter.config.llm.provider == "openai"


def test_sorter_strict_mode(mock_llm):
    """Test Sorter in strict mode."""
    sorter = Sorter(mode="strict", llm_provider=mock_llm)
    items = ["apple", "banana", "pear", "orange"]

    result = sorter.sort(items, "Sort fruits alphabetically")

    # Check that all items are returned
    assert len(result) == len(items)
    assert set(result) == set(items)

    # Verify LLM was called
    assert len(mock_llm.call_history) == 1
    assert mock_llm.call_history[0]["method"] == "generate_structured"


def test_sorter_filter_mode(mock_llm):
    """Test Sorter in filter mode."""
    sorter = Sorter(mode="filter", llm_provider=mock_llm)
    items = ["apple", "banana", "pear", "orange"]

    result = sorter.sort(items, "Keep only fruits with 'a'")

    # Check that result is subset of input
    assert all(item in items for item in result)

    # Verify LLM was called
    assert len(mock_llm.call_history) == 1
    assert mock_llm.call_history[0]["method"] == "generate_structured"


def test_sorter_empty_list(mock_llm):
    """Test Sorter with empty input list."""
    sorter = Sorter(mode="strict", llm_provider=mock_llm)
    result = sorter.sort([], "Sort empty list")
    assert result == []


def test_sorter_no_llm_provider():
    """Test Sorter without LLM provider raises error."""
    sorter = Sorter(mode="strict")
    with pytest.raises(ValueError, match="LLM provider not configured"):
        sorter.sort(["item"], "Sort items")

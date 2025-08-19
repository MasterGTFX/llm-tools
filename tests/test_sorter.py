"""Tests for Sorter."""

import pytest

from llmtools.sorter import Sorter


def test_initialization():
    """Test Sorter initialization."""
    sorter = Sorter(mode="strict")
    assert sorter.config.mode == "strict"

    config = {"provider": "openai", "model": "gpt-4"}
    sorter = Sorter(mode="filter", config=config)
    assert sorter.config.mode == "filter"
    assert sorter.config.llm.provider == "openai"


def test_strict_mode(test_llm):
    """Test strict mode sorting."""
    sorter = Sorter(mode="strict", llm_provider=test_llm)
    items = ["apple", "banana", "pear"]

    result = sorter.sort(items, "Sort alphabetically")

    assert len(result) == len(items)
    assert set(result) == set(items)


def test_filter_mode(test_llm):
    """Test filter mode sorting."""
    sorter = Sorter(mode="filter", llm_provider=test_llm)
    items = ["apple", "banana", "pear"]

    result = sorter.sort(items, "Keep fruits with 'a'")

    assert all(item in items for item in result)


def test_empty_list(test_llm):
    """Test sorting empty list."""
    sorter = Sorter(mode="strict", llm_provider=test_llm)
    result = sorter.sort([], "Sort empty list")
    assert result == []


def test_no_llm_provider():
    """Test sorting without LLM raises error."""
    sorter = Sorter(mode="strict")
    with pytest.raises(ValueError, match="LLM provider not configured"):
        sorter.sort(["item"], "Sort items")

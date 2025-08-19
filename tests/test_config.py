"""Tests for config models."""

from pathlib import Path

import pytest

from llmtools.config import KnowledgeBaseConfig, LLMConfig, SorterConfig


def test_llm_config():
    """Test LLMConfig model."""
    config = LLMConfig(provider="openai", model="gpt-4", temperature=0.5)

    assert config.provider == "openai"
    assert config.model == "gpt-4"
    assert config.temperature == 0.5
    assert config.timeout == 30


def test_llm_config_validation():
    """Test LLMConfig validation."""
    with pytest.raises(ValueError):
        LLMConfig(provider="openai", temperature=3.0)

    with pytest.raises(ValueError):
        LLMConfig(provider="openai", timeout=-1)


def test_knowledge_base_config():
    """Test KnowledgeBaseConfig model."""
    llm_config = LLMConfig(provider="openai")
    config = KnowledgeBaseConfig(
        llm=llm_config, instruction="Custom instruction", output_dir="/tmp/test"
    )

    assert config.llm.provider == "openai"
    assert config.instruction == "Custom instruction"
    assert config.output_dir == Path("/tmp/test")


def test_sorter_config():
    """Test SorterConfig model."""
    llm_config = LLMConfig(provider="anthropic")
    config = SorterConfig(llm=llm_config, mode="filter", max_retries=5)

    assert config.llm.provider == "anthropic"
    assert config.mode == "filter"
    assert config.max_retries == 5

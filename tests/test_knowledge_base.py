"""Tests for KnowledgeBase."""

import pytest
from llmtools.knowledge_base import KnowledgeBase


def test_initialization(output_dir):
    """Test basic initialization."""
    kb = KnowledgeBase(
        config={"provider": "openai"},
        instruction="Test instruction",
        output_dir=output_dir,
    )
    assert kb.config.instruction == "Test instruction"
    assert kb.config.output_dir == output_dir


def test_add_documents(test_llm, sample_docs, output_dir):
    """Test adding documents."""
    kb = KnowledgeBase(
        config={"provider": "openai"}, 
        output_dir=output_dir, 
        llm_provider=test_llm
    )
    
    kb.add_documents(sample_docs)
    assert len(kb.documents) == 2


def test_add_content(test_llm, output_dir):
    """Test adding content directly."""
    kb = KnowledgeBase(
        config={"provider": "openai"}, 
        output_dir=output_dir, 
        llm_provider=test_llm
    )
    
    kb.add_document_content("Test content")
    assert len(kb.documents) == 1
    assert kb.documents[0] == "Test content"


def test_process(test_llm, sample_docs, output_dir):
    """Test processing documents."""
    kb = KnowledgeBase(
        config={"provider": "openai"}, 
        output_dir=output_dir, 
        llm_provider=test_llm
    )
    
    kb.add_documents(sample_docs)
    versions = kb.process()
    
    assert len(versions) == 1
    assert len(kb.versions) == 1
    
    history_dir = output_dir / ".history"
    assert history_dir.exists()
    assert (history_dir / "v000.txt").exists()
    assert (history_dir / "metadata.json").exists()


def test_query(test_llm, sample_docs, output_dir):
    """Test querying knowledge base."""
    kb = KnowledgeBase(
        config={"provider": "openai"}, 
        output_dir=output_dir, 
        llm_provider=test_llm
    )
    
    kb.add_documents(sample_docs)
    kb.process()
    
    response = kb.query("What is in the knowledge base?")
    assert isinstance(response, str)
    assert len(response) > 0


def test_query_no_versions(test_llm, output_dir):
    """Test querying without versions raises error."""
    kb = KnowledgeBase(
        config={"provider": "openai"}, 
        output_dir=output_dir, 
        llm_provider=test_llm
    )
    
    with pytest.raises(ValueError, match="No knowledge base versions available"):
        kb.query("Test query")


def test_process_no_llm():
    """Test processing without LLM raises error."""
    kb = KnowledgeBase(config={"provider": "openai"})
    kb.add_document_content("Test content")
    
    with pytest.raises(ValueError, match="LLM provider not configured"):
        kb.process()

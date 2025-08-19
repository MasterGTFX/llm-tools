"""Tests for the KnowledgeBase component."""


import pytest

from llmtools.knowledge_base import KnowledgeBase


def test_knowledge_base_initialization(temp_output_dir):
    """Test KnowledgeBase initialization."""
    # Default initialization
    kb = KnowledgeBase(
        config={"provider": "openai"},
        instruction="Test instruction",
        output_dir=temp_output_dir,
    )
    assert kb.config.instruction == "Test instruction"
    assert kb.config.output_dir == temp_output_dir


def test_add_documents(mock_llm, sample_documents, temp_output_dir):
    """Test adding documents to knowledge base."""
    kb = KnowledgeBase(
        config={"provider": "openai"}, output_dir=temp_output_dir, llm_provider=mock_llm
    )

    # Add documents
    kb.add_documents(sample_documents)
    assert len(kb.documents) == 2


def test_add_document_content(mock_llm, temp_output_dir):
    """Test adding document content directly."""
    kb = KnowledgeBase(
        config={"provider": "openai"}, output_dir=temp_output_dir, llm_provider=mock_llm
    )

    kb.add_document_content("Test content")
    assert len(kb.documents) == 1
    assert kb.documents[0] == "Test content"


def test_process_documents(mock_llm, sample_documents, temp_output_dir):
    """Test processing documents into knowledge base."""
    kb = KnowledgeBase(
        config={"provider": "openai"}, output_dir=temp_output_dir, llm_provider=mock_llm
    )

    kb.add_documents(sample_documents)
    versions = kb.process()

    # Check that version was created
    assert len(versions) == 1
    assert len(kb.versions) == 1

    # Check that files were created
    history_dir = temp_output_dir / ".history"
    assert history_dir.exists()
    assert (history_dir / "v000.txt").exists()
    assert (history_dir / "metadata.json").exists()


def test_query_knowledge_base(mock_llm, sample_documents, temp_output_dir):
    """Test querying the knowledge base."""
    kb = KnowledgeBase(
        config={"provider": "openai"}, output_dir=temp_output_dir, llm_provider=mock_llm
    )

    kb.add_documents(sample_documents)
    kb.process()

    response = kb.query("What is in the knowledge base?")
    assert isinstance(response, str)
    assert len(response) > 0


def test_query_without_versions(mock_llm, temp_output_dir):
    """Test querying without processed versions raises error."""
    kb = KnowledgeBase(
        config={"provider": "openai"}, output_dir=temp_output_dir, llm_provider=mock_llm
    )

    with pytest.raises(ValueError, match="No knowledge base versions available"):
        kb.query("Test query")


def test_process_without_llm():
    """Test processing without LLM provider raises error."""
    kb = KnowledgeBase(config={"provider": "openai"})
    kb.add_document_content("Test content")

    with pytest.raises(ValueError, match="LLM provider not configured"):
        kb.process()

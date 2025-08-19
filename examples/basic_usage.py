"""
Basic usage examples for llmtools components.

This script demonstrates how to use the KnowledgeBase and Sorter components
with mock LLM providers (since real LLM providers require API keys).
"""

from pathlib import Path
from llmtools import KnowledgeBase, Sorter
from tests.conftest import MockLLMProvider


def main():
    """Run basic usage examples."""
    print("=== LLMTools Basic Usage Examples ===\n")
    
    # Initialize mock LLM provider
    llm = MockLLMProvider()
    
    # Example 1: Sorter in strict mode
    print("1. Sorter - Strict Mode (reorder all items)")
    sorter = Sorter(mode="strict", llm_provider=llm)
    fruits = ["banana", "apple", "orange", "pear"]
    
    sorted_fruits = sorter.sort(fruits, "Sort fruits alphabetically")
    print(f"Original: {fruits}")
    print(f"Sorted:   {sorted_fruits}")
    print()
    
    # Example 2: Sorter in filter mode
    print("2. Sorter - Filter Mode (subset only)")
    filterer = Sorter(mode="filter", llm_provider=llm)
    
    filtered_fruits = filterer.sort(fruits, "Keep only fruits with 'a'")
    print(f"Original: {fruits}")
    print(f"Filtered: {filtered_fruits}")
    print()
    
    # Example 3: Knowledge Base
    print("3. Knowledge Base - Document Processing")
    
    # Create temporary documents
    temp_dir = Path("temp_examples")
    temp_dir.mkdir(exist_ok=True)
    
    doc1 = temp_dir / "doc1.txt"
    doc2 = temp_dir / "doc2.txt"
    
    doc1.write_text("Python is a high-level programming language known for its simplicity and readability.")
    doc2.write_text("Machine learning is a subset of AI that enables computers to learn from data.")
    
    # Initialize knowledge base
    kb = KnowledgeBase(
        config={"provider": "mock"},
        instruction="Create a comprehensive knowledge base about programming and AI",
        output_dir=temp_dir / "kb_output",
        llm_provider=llm
    )
    
    # Add documents and process
    kb.add_documents([str(doc1), str(doc2)])
    versions = kb.process()
    
    print(f"Created {len(versions)} knowledge base version(s)")
    print(f"Knowledge base saved to: {kb.output_dir}")
    
    # Query the knowledge base
    response = kb.query("What topics are covered in the knowledge base?")
    print(f"Query response: {response}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n=== Examples completed successfully! ===")


if __name__ == "__main__":
    main()
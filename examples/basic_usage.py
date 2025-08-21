"""
Basic usage examples for llmtools components.

This script demonstrates how to use the KnowledgeBase and Sorter components
with mock LLM providers (since real LLM providers require API keys).
"""

from pathlib import Path

from llmtools import KnowledgeBase, Sorter


def main() -> None:
    """Run basic usage examples."""
    print("=== LLMTools Basic Usage Examples ===\n")

    # Initialize mock LLM provider
    from typing import Any, Callable, Optional

    from llmtools.interfaces.llm import LLMInterface

    class MockLLMProvider(LLMInterface):
        def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            history: Optional[list[dict[str, str]]] = None,
            **kwargs: Any,
        ) -> str:
            return "Mock response"

        def generate_structured(
            self,
            prompt: str,
            schema: dict[str, Any],
            system_prompt: Optional[str] = None,
            history: Optional[list[dict[str, str]]] = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            # Return a more realistic response based on the schema
            properties = schema.get("properties", {})
            if "sorted_items" in properties:
                # For Sorter strict mode
                return {
                    "sorted_items": ["apple", "banana", "orange", "pear"],
                    "reasoning": "Sorted alphabetically",
                }
            elif "filtered_items" in properties:
                # For Sorter filter mode
                return {
                    "filtered_items": ["apple", "banana"],
                    "reasoning": "Filtered items containing 'a'",
                }
            return {"mock": "structured_response"}

        def generate_model(
            self,
            prompt: str,
            model_class: type,
            system_prompt: Optional[str] = None,
            history: Optional[list[dict[str, str]]] = None,
            **kwargs: Any,
        ) -> Any:
            return model_class()

        def generate_with_tools(
            self,
            prompt: str,
            functions: Optional[list[Callable[..., Any]]] = None,
            function_map: Optional[dict[Callable[..., Any], dict[str, Any]]] = None,
            system_prompt: Optional[str] = None,
            history: Optional[list[dict[str, str]]] = None,
            max_tool_iterations: int = 10,
            handle_tool_errors: bool = True,
            tool_timeout: Optional[float] = None,
            **kwargs: Any,
        ) -> str:
            return "Mock tool response"

        def configure(self, config: dict[str, Any]) -> None:
            pass

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

    doc1.write_text(
        "Python is a high-level programming language known for its simplicity and readability."
    )
    doc2.write_text(
        "Machine learning is a subset of AI that enables computers to learn from data."
    )

    # Initialize knowledge base
    kb = KnowledgeBase(
        config={"provider": "mock"},
        instruction="Create a comprehensive knowledge base about programming and AI",
        output_dir=temp_dir / "kb_output",
        llm_provider=llm,
    )

    # Add documents and process
    kb.add_documents([str(doc1), str(doc2)])
    versions = kb.process()

    print(f"Created {len(versions)} knowledge base version(s)")
    print(f"Knowledge base saved to: {temp_dir / 'kb_output'}")

    # Query the knowledge base
    response = kb.query("What topics are covered in the knowledge base?")
    print(f"Query response: {response}")

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n=== Examples completed successfully! ===")


if __name__ == "__main__":
    main()

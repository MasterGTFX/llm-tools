"""Simple ask usage examples."""

from llmtools import OpenAIProvider, llm_ask

llm = OpenAIProvider(model="gpt-4o-mini")


def simple_example() -> None:
    print("=== Simple Example: Basic yes/no question ===")
    
    question = "Is 5 greater than 3?"
    print(f"Question: {question}")
    answer = llm_ask(question, llm)
    print(f"Answer: {answer}")


def complex_example() -> None:
    print("=== Complex Example: Decision with context and reasoning ===")
    
    question = "Should we deploy this feature?"
    context = "System load: 85%, Tests: passing, Team available for monitoring"
    print(f"Question: {question}")
    print(f"Context: {context}")
    
    answer, reasoning = llm_ask(question, llm, context=context, reasoning=True)
    print(f"Answer: {answer}")
    print(f"Reasoning: {reasoning}")


if __name__ == "__main__":
    simple_example()
    print("\n" + "=" * 30 + "\n")
    complex_example()
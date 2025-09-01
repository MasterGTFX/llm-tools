"""Simple ask usage examples."""

import vibetools
from vibetools import ai_ask

# Global configuration
vibetools.configure(model="gpt-4o-mini", temperature=0.7)
llm = vibetools.get_provider(vibetools.OpenAIProvider)


def simple_example() -> None:
    print("=== Simple Example: Basic yes/no question ===")

    question = "Is 5 greater than 3?"
    print(f"Q: {question}")

    answer = ai_ask(question, llm)
    print(f"A: {answer}")


def complex_example() -> None:
    print("=== Complex Example: Decision with context and reasoning ===")

    question = "Should we deploy this feature?"
    context = "System load: 85%, Tests: passing, Team available for monitoring"
    print(f"Q: {question}")
    print(f"Context: {context}")

    answer, reasoning = ai_ask(question, llm, context=context, reasoning=True)
    print(f"A: {answer}")
    print(f"Reasoning: {reasoning}")


if __name__ == "__main__":
    simple_example()
    print("\n" + "=" * 30 + "\n")
    complex_example()
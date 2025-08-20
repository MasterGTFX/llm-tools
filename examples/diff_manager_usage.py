"""Simple diff manager usage examples."""

import os

from llmtools import OpenAIProvider
from llmtools.utils.diff_manager import apply_llm_diff

# Enable llmtools logging to see the reasoning messages
os.environ["LLMTOOLS_LOG_LEVEL"] = "INFO"


def simple_example() -> None:
    print("=== Simple Example: Change print message ===")
    llm = OpenAIProvider(model="z-ai/glm-4.5-air")

    original = """def hello():
    print("Hello")

hello()"""
    print("Original:\n", original)

    modified = apply_llm_diff(
        original_content=original,
        prompt="Change the message to say 'Hi there!'",
        llm_provider=llm,
    )

    print("Modified:\n", modified)


def complex_example() -> None:
    print("=== Complex Example: Add method and update usage ===")
    llm = OpenAIProvider(model="z-ai/glm-4.5-air")

    original = """class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

calc = Calculator()
result = calc.add(5, 3)
print(result)"""
    print("Original:\n", original)

    modified = apply_llm_diff(
        original_content=original,
        prompt="Add a subtract method and use it in the example",
        llm_provider=llm,
    )
    print("Modified:\n", modified)


if __name__ == "__main__":
    simple_example()
    print("\n" + "=" * 30 + "\n")
    complex_example()

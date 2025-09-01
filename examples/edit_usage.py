"""Simple edit usage examples."""

from vibetools import OpenAIProvider, ai_edit

llm = OpenAIProvider(model="openai/gpt-4.1-mini", base_url="https://openrouter.ai/api/v1")


def simple_example() -> None:
    print("=== Simple Example: Change print message ===")

    original = """def hello():
    print("Hello")

hello()"""
    print(f"Before:\n{original}")

    modified = ai_edit(
        original_content=original,
        instruction="Change the message to say 'Hi there!'",
        llm_provider=llm,
    )

    print(f"After:\n{modified}")


def complex_example() -> None:
    print("=== Complex Example: Add method and update usage ===")

    original = """class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b

calc = Calculator()
result = calc.add(5, 3)
print(result)"""
    print(f"Before:\n{original}")

    modified = ai_edit(
        original_content=original,
        instruction="Add a subtract method and use it in the example",
        llm_provider=llm,
    )
    print(f"After:\n{modified}")


if __name__ == "__main__":
    simple_example()
    print("\n" + "=" * 30 + "\n")
    complex_example()

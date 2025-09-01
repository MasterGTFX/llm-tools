"""Simple filter usage examples."""

from vibetools import OpenAIProvider, ai_filter

llm = OpenAIProvider(model="gpt-5-mini")


def simple_example() -> None:
    """Basic filtering example."""
    print("=== Simple Example: Filter web development languages ===")

    languages = ["Python", "JavaScript", "Java", "C++", "Ruby", "PHP", "Go", "Rust"]
    instruction = "Keep only programming languages commonly used for web development"
    
    print(f"Input: {languages}")
    print(f"Instruction: {instruction}")

    filtered = ai_filter(
        items=languages,
        instruction=instruction,
        llm_provider=llm,
    )

    print(f"Output: {filtered}")


def complex_example() -> None:
    """Complex filtering with double-check."""
    print("=== Complex Example: Filter healthy foods (with double-check) ===")

    foods = ["Apple pie", "Grilled salmon", "Cookies", "Caesar salad", "Pizza", "Fruit salad", "Fried chicken", "Quinoa bowl"]
    instruction = "Keep only healthy, low-calorie food options"
    
    print(f"Input: {foods}")
    print(f"Instruction: {instruction}")

    filtered = ai_filter(
        items=foods,
        instruction=instruction,
        llm_provider=llm,
        double_check=True,
    )

    print(f"Output: {filtered}")


if __name__ == "__main__":
    simple_example()
    print("\n" + "=" * 30 + "\n")
    complex_example()

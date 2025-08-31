"""Simple filter usage examples."""

from llmtools import OpenAIProvider, llm_filter

llm = OpenAIProvider(model="gpt-5-mini")


def easy_example() -> None:
    """Basic filtering without double-check."""
    print("=== Easy Example: Filter programming languages: commonly used for web development ===")

    languages = ["Python", "JavaScript", "Java", "C++", "Ruby", "PHP", "Go", "Rust"]

    print("Original list:")
    for i, lang in enumerate(languages):
        print(f"  {i}: {lang}")

    # Filter to keep only languages commonly used for web development
    filtered = llm_filter(
        items=languages,
        instruction="Keep only programming languages commonly used for web development",
        llm_provider=llm,
    )

    print(f"\nFiltered list ({len(filtered)} items):")
    for lang in filtered:
        print(f"  - {lang}")


def harder_example() -> None:
    """Complex filtering with double-check verification."""
    print("=== Harder Example: Filter foods to keep healty, low-calorie options (with double-check) ===")

    foods = [
        "Apple pie with ice cream",
        "Grilled salmon with vegetables",
        "Chocolate chip cookies",
        "Caesar salad with grilled chicken",
        "Pepperoni pizza",
        "Fresh fruit salad",
        "Fried chicken and fries",
        "Quinoa bowl with roasted vegetables",
        "Cheeseburger and onion rings",
        "Greek yogurt with berries",
        "Pasta carbonara",
        "Spinach and kale smoothie",
        "Bacon and eggs",
        "Avocado toast with tomatoes",
    ]

    print("Original food list:")
    for i, food in enumerate(foods):
        print(f"  {i}: {food}")

    # Filter to keep only healthy, low-calorie options
    # Use double-check to ensure accuracy
    filtered = llm_filter(
        items=foods,
        instruction="Keep only healthy, low-calorie food options suitable for weight loss",
        llm_provider=llm,
        double_check=True,  # Enable verification step
    )

    print(f"\nFiltered healthy foods ({len(filtered)} items):")
    for food in filtered:
        print(f"  - {food}")


def custom_llm_example() -> None:
    """Example using custom LLM provider."""
    print("=== Custom LLM Example: Filter movies to keep only family-friendly ===")

    # Use a different model
    custom_llm = OpenAIProvider(model="gpt-4o-mini")

    movies = [
        "The Godfather",
        "Toy Story",
        "Pulp Fiction",
        "Finding Nemo",
        "The Dark Knight",
        "Shrek",
        "Goodfellas",
        "Monsters, Inc.",
        "Scarface",
        "The Incredibles",
    ]

    print("Original movies:")
    for i, movie in enumerate(movies):
        print(f"  {i}: {movie}")

    # Filter for family-friendly movies
    filtered = llm_filter(
        items=movies,
        instruction="Keep only family-friendly animated movies suitable for children",
        llm_provider=custom_llm,
    )

    print(f"\nFamily-friendly movies ({len(filtered)} items):")
    for movie in filtered:
        print(f"  - {movie}")


if __name__ == "__main__":
    easy_example()
    print("\n" + "=" * 50 + "\n")
    harder_example()
    print("\n" + "=" * 50 + "\n")
    custom_llm_example()

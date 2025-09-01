"""Simple sorter usage examples."""

from vibetools import OpenAIProvider, ai_sort

llm = OpenAIProvider(model="gpt-5-nano")
smarter_llm = OpenAIProvider(model="gpt-5-mini")


def simple_example() -> None:
    """Basic sorting example."""
    print("=== Simple Example: Sort movies by release year ===")

    movies = ["The Matrix (1999)", "Casablanca (1942)", "Avatar (2009)", "Titanic (1997)", "The Godfather (1972)"]
    instruction = "Sort these movies chronologically by release year, earliest first"
    
    print(f"Input: {movies}")
    print(f"Instruction: {instruction}")

    sorted_movies = ai_sort(
        items=movies,
        instruction=instruction,
        llm_provider=llm,
    )

    print(f"Output: {sorted_movies}")


def complex_example() -> None:
    """Complex sorting with double-check."""
    print("=== Complex Example: Sort activities by stress relief (with double-check) ===")

    activities = ["Social media", "Meditation", "Netflix", "Nature walk", "Coffee & emails", "Reading book"]
    instruction = "Sort by stress relief effectiveness, most effective first"
    
    print(f"Input: {activities}")
    print(f"Instruction: {instruction}")

    sorted_activities = ai_sort(
        items=activities,
        instruction=instruction,
        llm_provider=llm,
        double_check=smarter_llm,
    )

    print(f"Output: {sorted_activities}")


if __name__ == "__main__":
    simple_example()
    print("\n" + "=" * 30 + "\n")
    complex_example()
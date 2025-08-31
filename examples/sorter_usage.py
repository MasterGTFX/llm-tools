"""Simple sorter usage examples."""

from llmtools import OpenAIProvider, llm_sorter

llm = OpenAIProvider(model="gpt-5-nano")
smarter_llm = OpenAIProvider(model="gpt-5-mini")


def easy_example() -> None:
    """Basic sorting - objective ordering that's easy to verify."""
    print("=== Easy Example: Sort movies by release year ===")

    movies = [
        "The Matrix (1999)", "Casablanca (1942)", "Avatar (2009)", 
        "Titanic (1997)", "The Godfather (1972)", "Jaws (1975)",
        "Star Wars (1977)", "E.T. (1982)", "Jurassic Park (1993)",
        "The Dark Knight (2008)"
    ]

    print("Original list:")
    for i, movie in enumerate(movies):
        print(f"  {i}: {movie}")

    # Sort movies chronologically by release year
    # Easy to verify the results but hard to program without parsing
    sorted_movies = llm_sorter(
        items=movies,
        instruction="Sort these movies chronologically by release year, earliest first",
        llm_provider=llm,
    )

    print(f"\nSorted chronologically ({len(sorted_movies)} items):")
    for i, movie in enumerate(sorted_movies):
        print(f"  {i}: {movie}")


def harder_example() -> None:
    """Complex subjective sorting requiring LLM reasoning with double-check."""
    print("=== Harder Example: Sort activities by stress relief effectiveness ===")

    activities = [
        "Arguing with strangers on social media",
        "Deep breathing meditation", 
        "Binge-watching Netflix",
        "Going for a nature walk",
        "Drinking coffee while checking work emails",
        "Playing with a pet",
        "Listening to heavy metal music",
        "Taking a hot bath with candles",
        "Organizing your closet",
        "Video gaming until 3am",
        "Reading a good book",
        "Doing intense cardio workout"
    ]

    print("Original activity list:")
    for i, activity in enumerate(activities):
        print(f"  {i}: {activity}")

    # Sort by stress relief effectiveness (most effective first)
    # This requires subjective reasoning about psychology and wellness
    sorted_activities = llm_sorter(
        items=activities,
        instruction="Sort these activities by how effective they are for stress relief and mental wellness, most effective first",
        llm_provider=llm,
        double_check=smarter_llm,  # Use a smarter model for verification
    )

    print(f"\nSorted by stress relief effectiveness ({len(sorted_activities)} items):")
    for i, activity in enumerate(sorted_activities):
        print(f"  {i}: {activity}")


if __name__ == "__main__":
    easy_example()
    print("\n" + "=" * 70 + "\n")
    harder_example()
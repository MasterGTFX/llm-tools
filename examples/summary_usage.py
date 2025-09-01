"""Simple summary usage examples."""

from vibetools import OpenAIProvider, ai_summary

llm = OpenAIProvider(model="google/gemini-2.5-flash", base_url="https://openrouter.ai/api/v1")


def simple_example() -> None:
    print("=== Simple Example: Meeting summaries ===")

    documents = [
        "Monday: Discussed project timeline. Q1 deadline confirmed.",
        "Tuesday: Budget approved at $50K. Hired 2 new developers.", 
        "Wednesday: First prototype demo. Positive client feedback."
    ]
    instruction = "Create a weekly project summary"
    
    print(f"Input: {documents}")
    print(f"Instruction: {instruction}")

    summary = ai_summary(
        documents=documents,
        instruction=instruction,
        llm_provider=llm,
    )

    print(f"Output:\n{summary}")


def complex_example() -> None:
    print("=== Complex Example: Track all versions with initial summary) ===")

    initial_summary = "Project started in January with $10K budget."
    documents = [
        "February: Added new team member, budget increased to $15K.",
        "March: First milestone completed ahead of schedule.",
        "April: Client requested additional features, budget now $20K."
    ]
    instruction = "Update the project timeline summary"

    print(f"Initial: {initial_summary}")
    print(f"Input: {documents}")
    print(f"Instruction: {instruction}")

    all_versions = ai_summary(
        documents=documents,
        initial_summary=initial_summary,
        instruction=instruction,
        llm_provider=llm,
        return_all=True,
    )

    print("All versions:")
    for i, version in enumerate(all_versions, 1):
        print(f"V{i}: {version}")
        if i < len(all_versions):
            print("---")


if __name__ == "__main__":
    simple_example()
    print("\n" + "=" * 30 + "\n")
    complex_example()

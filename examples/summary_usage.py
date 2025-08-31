"""Simple summary usage examples."""

from llmtools import OpenAIProvider, llm_summary

llm = OpenAIProvider(model="google/gemini-2.5-flash", base_url="https://openrouter.ai/api/v1")


def basic_example() -> None:
    print("=== Basic Example: Research Study Evolution ===")

    documents = [
        """Initial Study Results: A new research study involving 50 participants found that 
        daily coffee consumption significantly improved memory performance by 15% compared 
        to baseline tests. The study was conducted over 4 weeks at University Hospital.""",

        """Extended Research: The study has been expanded to include 200 participants across 
        multiple age groups. Preliminary findings suggest the memory improvement effect is 
        most pronounced when coffee is consumed in the morning (before 10 AM), with afternoon 
        consumption showing minimal benefits.""",

        """Peer Review Concerns: Independent researchers reviewing the study methodology have 
        identified a significant flaw - the original study lacked a proper placebo control group. 
        Participants knew they were consuming coffee, which may have created expectation bias 
        affecting the results.""",

        """Controlled Follow-up Study: A new double-blind, placebo-controlled study with 150 
        participants has been completed. Results show no statistically significant improvement 
        in memory performance between the coffee group and placebo group (p=0.43). The original 
        findings appear to have been due to placebo effect and study design flaws."""
    ]

    print(f"Processing {len(documents)} research updates...")

    summary = llm_summary(
        documents=documents,
        instruction="Track the evolution of research findings, noting corrections and methodology improvements",
        llm_provider=llm,
    )

    print("Final Research Summary:")
    print(summary)


def with_initial_summary_example() -> None:
    print("=== Example with Initial Summary: Breaking News Evolution ===")

    initial = """BREAKING: Major fire breaks out at downtown office building on Main Street. 
    Firefighters responding, building being evacuated. Cause unknown at this time."""

    new_documents = [
        """Fire Department Update: The downtown office fire has been confirmed to be electrical 
        in origin. Three people have been hospitalized for smoke inhalation. The building's 
        sprinkler system helped contain the blaze to the third floor.""",

        """Investigation Report: Fire investigators have determined the cause was faulty wiring 
        installed during recent renovation work last month. The injury count has risen to 5 people 
        treated for smoke inhalation, with 2 in serious condition at Memorial Hospital.""",

        """Final Update: The electrical contractor responsible for the faulty wiring has been 
        charged with negligence. Final casualty count is 4 people injured (one person was 
        discharged and did not require extended treatment). Building suffered $2.3M in damages."""
    ]

    print("Initial breaking news:", initial)
    print(f"Processing {len(new_documents)} follow-up reports...")

    final_summary = llm_summary(
        documents=new_documents,
        instruction="Write an 'update news article' that incorporates all the initial facts and  new information.",
        llm_provider=llm,
        initial_summary=initial,
    )

    print("Final News Summary:")
    print(final_summary)


def version_tracking_example() -> None:
    print("=== Example with Version Tracking: Product Development Timeline ===")

    documents = [
        """Product Announcement: TechCorp reveals new flagship smartphone 'UltraPhone X' 
        with revolutionary 5-day battery life using new solid-state technology. Planned 
        launch date: Q1 2024. Expected price: $899.""",

        """Development Update: Battery testing phase reveals disappointing results - 
        solid-state batteries only achieving 3-day life in real-world usage. Engineering 
        team reports technical challenges with heat dissipation. Launch pushed to Q2 2024.""",

        """Technology Breakthrough: TechCorp acquires battery startup with proprietary 
        cooling technology. Combined with refined solid-state design, new target is 
        4-day battery life. Marketing adjusting expectations accordingly.""",

        """Final Specifications: UltraPhone X confirmed for April 2024 release with 
        3.5-day battery life, improved camera system (48MP vs planned 32MP), and 
        $799 price point. Pre-orders exceed 100,000 units."""
    ]

    print("Tracking all development phases...")

    all_versions = llm_summary(
        documents=documents,
        instruction="Track product development changes, specification updates, and timeline shifts",
        llm_provider=llm,
        return_all=True,
    )

    for i, version in enumerate(all_versions):
        print(f"\nDevelopment Phase {i + 1}:")
        print(version)
        print("-" * 50)

    print(f"\nFinal Product Summary: {all_versions[-1]}")


if __name__ == "__main__":
    basic_example()
    print("\n" + "=" * 50 + "\n")
    with_initial_summary_example()
    print("\n" + "=" * 50 + "\n")
    version_tracking_example()

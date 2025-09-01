# -*- coding: utf-8 -*-
"""Book List Manager - Chaining sort, filter, and edit operations."""

from vibetools import OpenAIProvider, ai_sort, ai_filter, ai_edit

llm = OpenAIProvider(model="google/gemini-2.5-flash", base_url="https://openrouter.ai/api/v1")


def book_list_manager_example() -> None:
    """Example: Book List Manager - Chain sorting, filtering, and editing."""
    print("=== Book List Manager: Chain Operations Example ===")
    
    # Initial book list with mixed formats and genres
    books = [
        "the martian - andy weir",
        "to kill a mockingbird - harper lee", 
        "dune - frank herbert",
        "1984 - george orwell",
        "the great gatsby - f. scott fitzgerald",
        "foundation - isaac asimov",
        "brave new world - aldous huxley",
        "the hitchhiker's guide to the galaxy - douglas adams",
        "pride and prejudice - jane austen",
        "neuromancer - william gibson",
        "fahrenheit 451 - ray bradbury",
        "the catcher in the rye - j.d. salinger"
    ]
    
    print("Original book list:")
    for i, book in enumerate(books):
        print(f"  {i+1}: {book}")
    
    # Step 1: Sort by popularity
    print("\nStep 1: Sorting by popularity...")
    sorted_books = ai_sort(
        items=books,
        instruction="Sort this list of books by popularity, most popular first",
        llm_provider=llm,
    )
    
    print("Books sorted by popularity:")
    for i, book in enumerate(sorted_books):
        print(f"  {i+1}: {book}")
    
    # Step 2: Filter to only science fiction books
    print("\nStep 2: Filtering for science fiction books...")
    sci_fi_books = ai_filter(
        items=sorted_books,
        instruction="Keep only science fiction books",
        llm_provider=llm,
    )
    
    print("Science fiction books only:")
    for i, book in enumerate(sci_fi_books):
        print(f"  {i+1}: {book}")
    
    # Step 3: Edit to improve formatting and add descriptions
    print("\nStep 3: Reformatting titles and adding descriptions...")
    
    # Convert list to formatted string for editing
    books_text = "\n".join([f"{i+1}. {book}" for i, book in enumerate(sci_fi_books)])
    
    edited_books = ai_edit(
        original_content=books_text,
        instruction="Rewrite each book entry to be in proper title case and add a short one-sentence description of the book after each title. Keep the author format.",
        llm_provider=llm,
    )
    
    print("Final formatted book list with descriptions:")
    print(edited_books)


def shopping_list_helper() -> None:
    """Shopping List Helper - Simple chain without verbose output."""
    print("=== Shopping List Helper ===")
    
    # Initial shopping list with mixed items and prices
    items = [
        "bananas - $2.99",
        "laptop - $899.99", 
        "apples - $1.49",
        "chicken breast - $8.99",
        "carrots - $1.25",
        "headphones - $79.99",
        "spinach - $3.49",
        "coffee - $12.99",
        "tomatoes - $2.79",
        "phone case - $15.99",
        "broccoli - $2.89",
        "pasta - $1.99"
    ]
    
    print("Original items:")
    for item in items:
        print(f"  - {item}")
    
    # Chain operations: sort by healthiness, filter fruits/vegetables, format with emojis
    formatted_list = ai_edit(
        original_content="\n".join(ai_filter(
            items=ai_sort(
                items=items,
                instruction="Sort the items by sugar content, lowest first",
                llm_provider=llm,
            ),
            instruction="Keep only items that are fruits or vegetables",
            llm_provider=llm,
        )),
        instruction="Format them into a nice shopping list with bullet points and emoji icons",
        llm_provider=llm,
    )
    
    print("\nFinal shopping list:")
    print(formatted_list)


if __name__ == "__main__":
    # book_list_manager_example()
    print("\n" + "=" * 50 + "\n")
    shopping_list_helper()
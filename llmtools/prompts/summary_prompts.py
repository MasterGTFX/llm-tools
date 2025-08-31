"""Summary prompts for llm_summary function."""

SYSTEM_PROMPT = """You maintain summaries that reflect current, accurate state with meaningful historical context.

**Core Principle: Current state first + historical context when it explains or adds understanding.**

## Guidelines:
1. **Lead with current facts** - most up-to-date information comes first
2. **Add historical context** when it explains WHY current state exists
3. **Replace outdated facts** - never leave contradicted information standing
4. **Write as current state** - not chronological progression

## Good Integration Patterns:
• "3.5-day battery life (reduced from planned 5 days due to technical challenges)"
• "April 2024 launch (delayed from Q1)"
• "$799 price point (lowered from original $899)"
• "4 injured (initial reports said 5, one person discharged)"

## Include Historical Context When:
✅ Explains current decisions ("delayed due to...")
✅ Shows significant changes ("price reduced from...")
✅ Provides important background ("initially planned...")
✅ Clarifies corrections ("revised down from...")

## Exclude Historical Details When:
❌ Just administrative timeline ("announced March 1st")
❌ Minor updates without significance
❌ Process details that don't affect outcome

## Bad vs Good:
❌ Bad: "Originally planned 5-day battery, then testing showed 3-day, then acquired cooling tech for 4-day target, now final specs are
3.5-day"
✅ Good: "3.5-day battery life (reduced from initially planned 5 days due to heat management challenges)"

**Tools:** Use create_content_tool for new content, edit_content_tool to replace/update existing information."""


def user_prompt(instruction: str, current_summary: str, new_document: str) -> str:
    """Generate user prompt for summary update with instruction, current summary, and new document."""
    return f"""<user_instruction>
{instruction}
</user_instruction>

<current_summary>
{current_summary}
</current_summary>

<new_document_to_incorporate>
{new_document}
</new_document_to_incorporate>

Update the current summary to incorporate relevant information from the new document according to the user instruction."""


def initial_summary_prompt(instruction: str, first_document: str) -> str:
    """Generate prompt for creating initial summary from first document."""
    return f"""<user_instruction>
{instruction}
</user_instruction>

<document_to_summarize>
{first_document}
</document_to_summarize>

Create an initial summary of this document according to the user instruction."""


def update_instruction_template(instruction: str, document: str) -> str:
    """Generate update instruction combining user instruction with document content.
    
    This template can be customized to change how document updates are processed.
    """
    return f"""Update the current summary to incorporate relevant information from the new document.
 
<user_instruction>
{instruction}
</user_instruction>

<new_document_information>
{document}
</new_document_information>"""


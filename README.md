# vibe-tools

A collection of **simple, self-contained LLM functions** for personal and project use.
Just import a function and use it - minimal setup, optional configuration, no heavy dependencies.

Each function is designed to solve a specific problem with a single line of code, using OpenAI by default.

---

## Available Tools

* **`ai_ask(question, context="")`** - Ask yes/no questions and get boolean answers
* **`ai_filter(items, instruction)`** - Filter a list based on natural language criteria
* **`ai_sort(items, instruction)`** - Sort a list using natural language instructions
* **`ai_knowledge_base(documents, instruction)`** - Build a knowledge base from documents (TBA)
* **`ai_edit(text, instruction)`** - Edit or modify text content using AI instructions

---

## Installation

```bash
git clone https://github.com/MasterGTFX/vibe-tools.git
cd vibe-tools
pip install -e .
```

Requires Python 3.9+ and automatically installs OpenAI for immediate use.
Set your `OPENAI_API_KEY` environment variable and you're ready to go.

---

## Usage Examples

### Ask Yes/No Questions

```python
from vibetools import ai_ask

# Simple questions
answer = ai_ask("Is 5 greater than 3?")
print(answer)  # True

# Complex decisions with context
answer, reasoning = ai_ask(
    "Should we deploy this feature?", 
    context="System load: 85%, Tests: passing, Team available",
    reasoning=True
)
print(f"{answer} - {reasoning}")  # True - The system is stable and ready...
```

### Filter Lists

```python
from vibetools import ai_filter

fruits = ["apple", "banana", "pear", "orange", "grape"]
citrus = ai_filter(fruits, "Keep only citrus fruits")
print(citrus)  # ["orange"]

numbers = [1, 15, 23, 8, 42, 7]
large = ai_filter(numbers, "Keep numbers greater than 10")
print(large)  # [15, 23, 42]
```

### Sort Lists

```python
from vibetools import ai_sort

fruits = ["apple", "banana", "pear", "orange"]
sorted_fruits = ai_sort(fruits, "Sort alphabetically")
print(sorted_fruits)  # ["apple", "banana", "orange", "pear"]

tasks = ["buy groceries", "urgent: call mom", "finish project", "book dentist"]
by_priority = ai_sort(tasks, "Sort by urgency")
print(by_priority)  # ["urgent: call mom", "finish project", "buy groceries", "book dentist"]
```

### Build Knowledge Base

```python
from vibetools import ai_knowledge_base

documents = ["intro.txt", "chapter1.md", "notes.md"]
kb = ai_knowledge_base(documents, "Create a comprehensive summary of all key concepts")
print(kb)  # Returns structured knowledge base content
```

### Edit Text

```python
from vibetools import ai_edit

text = "This is a draft document with some errors."
improved = ai_edit(text, "Fix grammar and make more professional")
print(improved)  # "This is a draft document that contains some errors."
```

---

## Configuration (Optional)

Works out of the box, configure only when needed:

```python
import vibetools

# Global defaults
vibetools.configure(model="gpt-4o", temperature=0.3)

# Environment variables
export OPENAI_API_KEY="your-key" 
export OPENAI_DEFAULT_MODEL="gpt-4o"

# Advanced usage
llm = llmtools.get_provider(llmtools.OpenAIProvider, model="gpt-4o")
answer = ai_ask("Question?", llm_provider=llm)
```

---

## Project Structure

```
vibe-tools/
│
├── vibetools/
│   ├── __init__.py       # Main function exports
│   ├── tools/            # Self-contained tool functions
│   │   ├── ask.py        # ai_ask function
│   │   ├── filter.py     # ai_filter function
│   │   ├── sorter.py     # ai_sort function
│   │   ├── summary.py  # ai_knowledge_base function
│   │   └── edit.py       # ai_edit function
│   │
│   ├── utils/            # Shared utilities
│   │   └── llm_client.py # OpenAI client utilities
│   │
├── examples/             # Simple usage examples
├── tests/                # Function tests
└── README.md
```

---

## Philosophy

* **Simplicity first** → single function calls solve specific problems
* **Minimal setup** → works out of the box with OpenAI, optional configuration
* **Self-contained functions** → no classes or complex setup required
* **Immediate productivity** → import and use in one line
* **Leverage structured output** → use LLM function calling and JSON schema internally
* **Predictable results** → consistent, machine-usable responses

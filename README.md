# llm-tools

A collection of **simple, self-contained LLM functions** for personal and project use.
Just import a function and use it - no classes, no configuration, no heavy dependencies.

Each function is designed to solve a specific problem with a single line of code, using OpenAI by default.

---

## Available Tools

* **`llm_filter(items, instruction)`** - Filter a list based on natural language criteria
* **`llm_sorter(items, instruction)`** - Sort a list using natural language instructions
* **`llm_knowledge_base(documents, instruction)`** - Build a knowledge base from documents
* **`llm_edit(text, instruction)`** - Edit or modify text content using LLM instructions

---

## Installation

```bash
git clone https://github.com/yourname/llm-tools.git
cd llm-tools
pip install -e .
```

Requires Python 3.9+ and automatically installs OpenAI for immediate use.
Set your `OPENAI_API_KEY` environment variable and you're ready to go.

---

## Usage Examples

### Filter Lists

```python
from llmtools import llm_filter

fruits = ["apple", "banana", "pear", "orange", "grape"]
citrus = llm_filter(fruits, "Keep only citrus fruits")
print(citrus)  # ["orange"]

numbers = [1, 15, 23, 8, 42, 7]
large = llm_filter(numbers, "Keep numbers greater than 10")
print(large)  # [15, 23, 42]
```

### Sort Lists

```python
from llmtools import llm_sorter

fruits = ["apple", "banana", "pear", "orange"]
sorted_fruits = llm_sorter(fruits, "Sort alphabetically")
print(sorted_fruits)  # ["apple", "banana", "orange", "pear"]

tasks = ["buy groceries", "urgent: call mom", "finish project", "book dentist"]
by_priority = llm_sorter(tasks, "Sort by urgency")
print(by_priority)  # ["urgent: call mom", "finish project", "buy groceries", "book dentist"]
```

### Build Knowledge Base

```python
from llmtools import llm_knowledge_base

documents = ["intro.txt", "chapter1.md", "notes.md"]
kb = llm_knowledge_base(documents, "Create a comprehensive summary of all key concepts")
print(kb)  # Returns structured knowledge base content
```

### Edit Text

```python
from llmtools import llm_edit

text = "This is a draft document with some errors."
improved = llm_edit(text, "Fix grammar and make more professional")
print(improved)  # "This is a draft document that contains some errors."
```

---

## Project Structure

```
llm-tools/
│
├── llmtools/
│   ├── __init__.py       # Main function exports
│   ├── tools/            # Self-contained tool functions
│   │   ├── filter.py     # llm_filter function
│   │   ├── sorter.py     # llm_sorter function
│   │   ├── knowledge.py  # llm_knowledge_base function
│   │   └── edit.py       # llm_edit function
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
* **No configuration** → works out of the box with OpenAI
* **Self-contained functions** → no classes or complex setup required
* **Immediate productivity** → import and use in one line
* **Leverage structured output** → use LLM function calling and JSON schema internally
* **Predictable results** → consistent, machine-usable responses

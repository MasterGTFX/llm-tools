# llm-tools

A modular set of **LLM utilities for personal and project use**, designed to be minimal, configurable, and composable.
Each tool can be imported independently (no heavy boilerplate, no large dependency trees).

The goal is to build a **Pythonic toolkit** that provides reusable building blocks for working with LLMs.

---

## Features (planned)

* **Knowledge Base (`llmtools.knowledge_base`)**

  * Build and iteratively update a knowledge base from a set of documents (any text form).
  * Supports an optional *initial knowledge base*.
  * Stores incremental versions under a `.history/` directory at the target output location.
  * Uses structured LLM output to manage diffs and track updates over time.

* **Sorter (`llmtools.reranker`)**

  * Sort or filter a Python list based on an instruction using an LLM.
  * **Strict mode** → output has the same length and content as the input, only re-ordered.
  * **Filter mode** → output is a subset of items that satisfy certain text conditions.
  * Useful for lightweight re-ranking, filtering, or ordering tasks.

* **Shared Components**

  * **Interfaces** → abstract definitions for LLMs and storage backends.
  * **Utils** → helpers like diff manager, chunking, embeddings, structured output parsing.
  * **Config** → Pydantic-based configs for flexibility.

---

## Installation

```bash
git clone https://github.com/yourname/llm-tools.git
cd llm-tools
pip install -e .
```

No extra dependencies required beyond Python 3.9+ and `pydantic`.
Optional integrations (like `openai`) can be installed if needed.

---

## Usage Example

### Knowledge Base

```python
from llmtools.knowledge_base import KnowledgeBase

# Initialize with a backend LLM client
kb = KnowledgeBase(config={"provider": "openai"}, instruction = "Create a comprehensive knowledge base containting all useful information from book", output_dir=None, init = None)

# Add a batch of documents
kb.add_documents(["intro.txt", "chapter1.md"])

# Process documents and return list of versions [-1] -> the latest
output_kb = kb.process() 

# Query knowledge base
response = kb.query("Summarize changes since the first version")
print(response)
```

### Sorter

```python
from llmtools.sorter import Sorter

items = ["apple", "banana", "pear", "orange"]

# Strict sorting (preserves all items, only reorders)
sorter = Sorter(mode="strict")
result = sorter.sort(items, instruction="Sort fruits alphabetically")
print(result)  # ["apple", "banana", "orange", "pear"]

# Filter mode (subset only)
filterer = Sorter(mode="filter")
result = filterer.sort(items, instruction="Keep only fruits with 'a'")
print(result)  # ["apple", "banana", "orange"]
```

---

## Project Structure

```
llm-tools/
│
├── llmtools/
│   ├── __init__.py
│   ├── knowledge_base/
│   │   ├── __init__.py
│   │   └── base.py
│   │
│   ├── sorter/
│   │   ├── __init__.py
│   │   └── base.py
│   │
│   ├── interfaces/
│   │   ├── __init__.py
│   │   └── llm.py        # abstract LLM interface
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── diff_manager.py
│   │
│   └── config.py         # Pydantic config models
│
├── tests/                # lightweight tests
│
└── README.md
```

---

## Philosophy

* **Minimal dependencies** → only install what you need.
* **Pythonic design** → clean interfaces, clear module boundaries.
* **Composable** → tools can be used independently or together.
* **Configurable** → Pydantic-based configs for flexibility.
* **Not a framework** → just utilities you can import and use.
* **Utilize LLM tool calling** → leverage model-native function/tool invocation for better modularity.
* **Structured output** → enforce predictable, machine-usable responses (JSON, diffs, versioning).

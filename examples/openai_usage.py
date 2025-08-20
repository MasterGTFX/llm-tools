"""
Simple OpenAI LLM provider usage examples.

Set your OPENAI_API_KEY environment variable before running.
Get your API key at: https://platform.openai.com/api-keys

For OpenRouter usage, set OPENAI_BASE_URL to https://openrouter.ai/api/v1
and use OpenRouter model names like "openai/gpt-4o" or "anthropic/claude-3-haiku".
"""

from typing import Any, Union

from pydantic import BaseModel

from llmtools import OpenAIProvider


class Person(BaseModel):
    """Example Pydantic model for structured output."""

    name: str
    age: int
    occupation: str


def main() -> None:
    """Run basic OpenAI examples."""
    # Initialize OpenAI provider (uses official OpenAI API by default)
    llm = OpenAIProvider(model="gpt-5-nano")

    # For OpenRouter usage, uncomment the following:
    # llm = OpenAIProvider(
    #     model="openai/gpt-5-nano",
    #     base_url="https://openrouter.ai/api/v1"
    # )

    # Example 1: Basic text generation
    print("1. Basic Text Generation")
    response = llm.generate("What is OpenAI? Answer briefly.")
    print(f"Response: {response}\n")

    # Example 2: Text generation with conversation history
    print("2. Text Generation with History")
    history = [
        {"role": "user", "content": "Hi there! My name is Bob."},
        {"role": "assistant", "content": "Hello! How can I help you?"},
    ]
    response = llm.generate("What's my name?", history=history)
    print(f"Response: {response}\n")

    # Example 3: Structured output with JSON schema
    print("3. Structured Output (JSON Schema)")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "occupation": {"type": "string"},
        },
        "required": ["name", "age", "occupation"],
        "additionalProperties": False,
    }
    structured_response: dict[str, Any] = llm.generate_structured(
        "Generate a person with occupation", schema
    )
    print(f"JSON Schema Response: {structured_response}\n")

    # Example 4: Structured output with Pydantic model
    print("4. Structured Output (Pydantic Model)")
    person = llm.generate_model(
        "Generate a fictional person with their details", Person
    )
    print(f"Person: {person}\n")

    # Example 5: Function/tool calling
    print("5. Function Calling")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    tool_response: Union[str, dict[str, Any]] = llm.generate_with_tools(
        "What's the weather in Paris?", tools
    )
    print(f"Tool Response: {tool_response}")


if __name__ == "__main__":
    main()

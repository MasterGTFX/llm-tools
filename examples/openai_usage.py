"""
Simple OpenAI LLM provider usage examples.

Set your OPENAI_API_KEY environment variable before running.
Get your API key at: https://platform.openai.com/api-keys

For OpenRouter usage, set OPENAI_BASE_URL to https://openrouter.ai/api/v1
and use OpenRouter model names like "openai/gpt-4o" or "anthropic/claude-3-haiku".
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel

from llmtools import OpenAIProvider

# Global LLM provider to avoid duplication
llm = OpenAIProvider(model="gpt-5-mini")


class Person(BaseModel):
    """Example Pydantic model for structured output."""

    name: str
    age: int
    occupation: str


def basic_text_generation() -> None:
    """Basic text generation example."""
    print("=== Basic Text Generation ===")
    response = llm.generate("What is OpenAI? Answer briefly.")
    print(f"Response: {response}")


def text_generation_with_history() -> None:
    """Text generation with conversation history example."""
    print("=== Text Generation with History ===")
    history = [
        {"role": "user", "content": "Hi there! My name is Bob."},
        {
            "role": "assistant",
            "content": "Hello Bob! Nice to meet you. How can I help you today?",
        },
    ]

    response = llm.generate("What's my name?", history=history)
    print(f"Response: {response}")


def structured_json_example() -> None:
    """Structured output with JSON schema example."""
    print("=== Structured JSON Output ===")
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

    response: dict[str, Any] = llm.generate_structured(
        "Generate a fictional person with their details", schema
    )
    print(f"JSON Response: {response}")


def structured_model_example() -> None:
    """Structured output with Pydantic model example."""
    print("=== Structured Model Output ===")
    person = llm.generate_model(
        "Generate a fictional person with their details", Person
    )
    print(f"Generated Person: {person}")


def function_calling_example() -> None:
    """Function calling example using functions parameter (auto-generation with enums)."""
    print("=== Function Calling (Auto-Generated Schema with Enums) ===")

    class TemperatureUnit(Enum):
        CELSIUS = "celsius"
        FAHRENHEIT = "fahrenheit"

    def get_weather(
        location: str, unit: TemperatureUnit = TemperatureUnit.CELSIUS
    ) -> str:
        """Get weather information for a location in specified temperature unit.

        Args:
            location: Name of the city to get weather for
            unit: Temperature unit (celsius or fahrenheit) (default celsius)

        Returns:
            Weather information as a formatted string
        """
        weather_data_celsius = {
            "Paris": 22,
            "London": 15,
            "Tokyo": 25,
            "New York": 18,
        }

        if location not in weather_data_celsius:
            return f"Weather data not available for {location}"

        temp_c = weather_data_celsius[location]

        if unit == TemperatureUnit.FAHRENHEIT:
            temp_f = (temp_c * 9 / 5) + 32
            return f"{location}: {temp_f}°F"
        else:
            return f"{location}: {temp_c}°C"

    # Just pass the functions - schemas are auto-generated with enum constraints!
    response = llm.generate_with_tools(
        prompt="What's the weather in Paris (in Celsius) and Tokyo (in Fahrenheit)?",
        functions=[get_weather],
    )
    print(f"Final Response: {response}")


def manual_function_calling_example() -> None:
    """Function calling example using manual function_map (original approach)."""
    print("=== Function Calling (Manual Schema) ===")

    def get_weather(location: str, unit: str = "celsius") -> str:
        """Get weather information for a location in specified temperature unit."""
        weather_data_celsius = {
            "Paris": 22,
            "London": 15,
            "Tokyo": 25,
            "New York": 18,
        }

        if location not in weather_data_celsius:
            return f"Weather data not available for {location}"

        temp_c = weather_data_celsius[location]

        if unit == "fahrenheit":
            temp_f = (temp_c * 9 / 5) + 32
            return f"{location}: {temp_f}°F"
        else:
            return f"{location}: {temp_c}°C"

    # Manual function map with function as key, schema as value (original format)
    function_map = {
        get_weather: {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a location in specified temperature unit",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name",
                        },
                        "unit": {
                            "type": "string",
                            "description": "Temperature unit",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    }

    # Automatic execution - LLM calls function and gets results automatically
    response = llm.generate_with_tools(
        "What's the weather in Paris in Celsius and New York in Fahrenheit?",
        function_map=function_map,
    )
    print(f"Final Response: {response}")


if __name__ == "__main__":
    # basic_text_generation()
    # print()
    # text_generation_with_history()
    # print()
    # structured_json_example()
    # print()
    # structured_model_example()
    # print()
    function_calling_example()
    print()
    manual_function_calling_example()

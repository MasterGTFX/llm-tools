"""
OpenRouter LLM provider usage examples.

This script demonstrates how to use the OpenRouter provider with llmtools.
Make sure to set your OPENROUTER_API_KEY environment variable before running.
"""

import os
from pathlib import Path
from llmtools import KnowledgeBase, Sorter, OpenRouterProvider


def main():
    """Run OpenRouter usage examples."""
    print("=== OpenRouter LLM Provider Examples ===\n")
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY environment variable not found!")
        print("Please set your OpenRouter API key before running this example.")
        print("You can get one at: https://openrouter.ai/keys")
        print("\nUsing mock responses for demonstration...\n")
        use_mock = True
    else:
        print("✅ Found OpenRouter API key, using real API calls\n")
        use_mock = False
    
    if use_mock:
        # Use mock provider for demonstration
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from tests.conftest import MockLLMProvider
        llm = MockLLMProvider()
    else:
        # Initialize OpenRouter provider
        llm = OpenRouterProvider(
            model="openai/gpt-4o-mini",  # Using a cheaper model for examples
            api_key=api_key
        )
    
    # Example 1: Basic text generation
    print("1. Basic Text Generation")
    try:
        response = llm.generate(
            prompt="Explain what OpenRouter is in one sentence.",
            system_prompt="You are a helpful AI assistant."
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Example 2: Text generation with conversation history
    print("2. Text Generation with History")
    try:
        history = [
            {"role": "user", "content": "Hi, I'm learning about AI APIs."},
            {"role": "assistant", "content": "Great! I'd be happy to help you learn about AI APIs."}
        ]
        response = llm.generate(
            prompt="What's the advantage of using OpenRouter vs calling OpenAI directly?",
            history=history
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Example 3: Structured output generation
    print("3. Structured Output Generation")
    try:
        schema = {
            "type": "object",
            "properties": {
                "pros": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of advantages"
                },
                "cons": {
                    "type": "array", 
                    "items": {"type": "string"},
                    "description": "List of disadvantages"
                }
            },
            "required": ["pros", "cons"],
            "additionalProperties": False
        }
        
        response = llm.generate_structured(
            prompt="List pros and cons of using OpenRouter for AI applications",
            schema=schema
        )
        print(f"Structured Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Example 4: Tool calling
    print("4. Tool/Function Calling")
    try:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        response = llm.generate_with_tools(
            prompt="What's the weather like in San Francisco?",
            tools=tools
        )
        print(f"Tool Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Example 5: Using with Sorter
    print("5. Integration with Sorter Component")
    try:
        sorter = Sorter(mode="strict", llm_provider=llm)
        cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
        
        sorted_cities = sorter.sort(cities, "Sort cities by population size (largest first)")
        print(f"Original cities: {cities}")
        print(f"Sorted by population: {sorted_cities}")
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Example 6: Using with KnowledgeBase
    print("6. Integration with KnowledgeBase Component")
    try:
        # Create temporary documents
        temp_dir = Path("temp_openrouter")
        temp_dir.mkdir(exist_ok=True)
        
        doc1 = temp_dir / "openrouter.txt"
        doc1.write_text("""
        OpenRouter is a unified API for hundreds of AI models. It provides:
        - Access to multiple AI providers through a single endpoint
        - Automatic fallbacks and load balancing
        - Cost optimization by choosing the most efficient model
        - Standardized interface across different model providers
        """)
        
        # Initialize knowledge base with OpenRouter
        kb = KnowledgeBase(
            config={"provider": "openrouter"},
            instruction="Create a knowledge base about AI API services and OpenRouter",
            output_dir=temp_dir / "kb_output",
            llm_provider=llm
        )
        
        # Add documents and process
        kb.add_documents([str(doc1)])
        versions = kb.process()
        
        print(f"Created {len(versions)} knowledge base version(s)")
        
        # Query the knowledge base
        response = kb.query("What are the main benefits of using OpenRouter?")
        print(f"KB Query Response: {response}")
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"Error: {e}")
    print()
    
    # Example 7: Provider configuration
    print("7. Provider Configuration")
    try:
        # Show current model info
        info = llm.get_model_info()
        print(f"Current provider info: {info}")
        
        # Configure provider (if not using mock)
        if not use_mock:
            llm.configure({
                "temperature": 0.3,
                "max_tokens": 150
            })
            print("✅ Updated provider configuration")
        else:
            print("(Mock provider - configuration demo only)")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== OpenRouter Examples Completed ===")
    if not api_key:
        print("\n💡 To use real OpenRouter API calls:")
        print("1. Get an API key at https://openrouter.ai/keys")
        print("2. Set environment variable: export OPENROUTER_API_KEY=your_key")
        print("3. Run this example again")


if __name__ == "__main__":
    main()
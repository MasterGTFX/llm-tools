"""Configuration system usage examples."""

import vibetools
from vibetools import OpenAIProvider, ai_ask


def basic_usage() -> None:
    """Basic provider usage - no configuration needed."""
    print("=== Basic Usage (Default Settings) ===")
    
    # Simple provider with defaults
    llm = OpenAIProvider()
    print(f"Model: {llm.model}")
    print(f"Temperature: {llm.config['temperature']}")
    print(f"Max iterations: {llm.config['max_tool_iterations']}")
    

def instance_configuration() -> None:
    """Configure individual provider instances."""
    print("=== Instance Configuration ===")
    
    # Override specific settings for this instance
    llm = OpenAIProvider(
        model="gpt-4o",
        temperature=0.9,
        max_tool_iterations=15
    )
    print(f"Model: {llm.model}")
    print(f"Temperature: {llm.config['temperature']}")
    print(f"Max iterations: {llm.config['max_tool_iterations']}")


def global_configuration() -> None:
    """Set global defaults for all future providers."""
    print("=== Global Configuration ===")
    
    # Configure global defaults including base_url
    vibetools.configure(
        model="gpt-4o", 
        temperature=0.8,
        max_tool_iterations=25,
        base_url="https://api.custom-llm.com/v1"
    )
    
    # New providers will use these defaults
    llm1 = vibetools.get_provider(OpenAIProvider)
    llm2 = vibetools.get_provider(OpenAIProvider, temperature=0.2)  # Override one setting
    
    print(f"LLM1 - Model: {llm1.model}, Temp: {llm1.config['temperature']}, Base URL: {llm1.base_url}")
    print(f"LLM2 - Model: {llm2.model}, Temp: {llm2.config['temperature']}, Base URL: {llm2.base_url}")


def temporary_configuration() -> None:
    """Temporarily change global settings."""
    print("=== Temporary Configuration ===")
    
    # Set some defaults
    vibetools.configure(temperature=0.7, model="gpt-4o-mini")
    
    print("Normal config:")
    llm1 = vibetools.get_provider(OpenAIProvider)
    print(f"  Model: {llm1.model}, Temp: {llm1.config['temperature']}")
    
    # Temporarily change settings
    with vibetools.temp_config(temperature=0.1, model="gpt-4o"):
        print("Inside temp_config:")
        llm2 = vibetools.get_provider(OpenAIProvider)
        print(f"  Model: {llm2.model}, Temp: {llm2.config['temperature']}")
    
    # Back to original settings
    print("After temp_config:")
    llm3 = vibetools.get_provider(OpenAIProvider)
    print(f"  Model: {llm3.model}, Temp: {llm3.config['temperature']}")


def environment_variables() -> None:
    """Environment variables are used as fallbacks when parameters not provided."""
    print("=== Environment Variables ===")
    
    import os
    env_model = os.getenv("OPENAI_DEFAULT_MODEL")
    env_base_url = os.getenv("OPENAI_BASE_URL")
    
    print("Current environment variables:")
    print(f"  OPENAI_DEFAULT_MODEL = {env_model}")  
    print(f"  OPENAI_BASE_URL = {env_base_url}")
    
    print("\nPrecedence demonstration:")
    print("1. User configurable (highest)")
    print("2. Environment variables (middle)")  
    print("3. Defaults (lowest)")
    
    # Test environment variable fallback
    llm_env = OpenAIProvider()  # Uses env vars as fallback
    print(f"\nNo params provided -> Model: {llm_env.model}, Base URL: {llm_env.base_url}")
    
    # Test user override
    llm_override = OpenAIProvider(model="gpt-4o", base_url="https://api.openai.com/v1")
    print(f"User params provided -> Model: {llm_override.model}, Base URL: {llm_override.base_url}")
    
    # Test partial override
    llm_partial = OpenAIProvider(model="claude-3")  # base_url from env
    print(f"Partial override -> Model: {llm_partial.model}, Base URL: {llm_partial.base_url}")


def realistic_example() -> None:
    """Realistic usage with tools."""
    print("=== Realistic Example ===")
    
    # Set up your preferred defaults
    import os
    vibetools.configure(
        temperature=0.3,  # Conservative for factual questions
        max_tool_iterations=10,
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model=os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini")
    )
    
    # Create provider with global defaults
    llm = vibetools.get_provider(OpenAIProvider)
    
    # Use tools normally - they inherit provider configuration
    try:
        answer = ai_ask("Is Python a programming language?", llm)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Error (likely missing API key): {e}")


if __name__ == "__main__":
    print("VibeTools Configuration Examples")
    print("=" * 40)
    
    basic_usage()
    print()
    
    instance_configuration()
    print()
    
    global_configuration()
    print()
    
    temporary_configuration()
    print()
    
    environment_variables()
    print()
    
    realistic_example()
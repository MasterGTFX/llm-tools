"""Tests for OpenRouter LLM provider."""

import os
from unittest.mock import Mock, patch

import pytest

from llmtools.interfaces.llm import LLMInterface


# Test imports with and without openai dependency
def test_openrouter_import_error():
    """Test that appropriate error is raised when OpenAI SDK is not available."""
    import sys

    original_openai = sys.modules.get("openai")

    # Remove openai from sys.modules temporarily
    if "openai" in sys.modules:
        del sys.modules["openai"]

    # Also remove any cached imports
    if "llmtools.interfaces.openrouter_llm" in sys.modules:
        del sys.modules["llmtools.interfaces.openrouter_llm"]

    # Mock the import to fail
    with patch.dict("sys.modules", {"openai": None}):
        with pytest.raises(ImportError, match="OpenAI SDK is required"):
            import importlib

            importlib.invalidate_caches()
            from llmtools.interfaces.openrouter_llm import (
                OpenRouterProvider,  # noqa: F401
            )

    # Restore original state
    if original_openai is not None:
        sys.modules["openai"] = original_openai


@pytest.mark.skipif(
    not pytest.importorskip("openai", minversion="1.0.0"),
    reason="OpenAI SDK not available",
)
class TestOpenRouterProvider:
    """Test OpenRouter provider functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        from llmtools.interfaces.openrouter_llm import OpenRouterProvider

        # Mock OpenAI client
        self.mock_client = Mock()

        with patch("llmtools.interfaces.openrouter_llm.OpenAI") as mock_openai:
            mock_openai.return_value = self.mock_client
            self.provider = OpenRouterProvider(
                api_key="test-key", model="openai/gpt-4o"
            )

    def test_initialization(self):
        """Test provider initialization."""
        assert self.provider.api_key == "test-key"
        assert self.provider.model == "openai/gpt-4o"
        assert self.provider.base_url == "https://openrouter.ai/api/v1"

    def test_initialization_with_env_var(self):
        """Test initialization with environment variable."""
        from llmtools.interfaces.openrouter_llm import OpenRouterProvider

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            with patch("llmtools.interfaces.openrouter_llm.OpenAI"):
                provider = OpenRouterProvider()
                assert provider.api_key == "env-key"

    def test_initialization_no_api_key(self):
        """Test that initialization fails without API key."""
        from llmtools.interfaces.openrouter_llm import OpenRouterProvider

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key is required"):
                OpenRouterProvider()

    def test_build_messages(self):
        """Test message building functionality."""
        messages = self.provider._build_messages(
            prompt="Hello",
            system_prompt="You are a helpful assistant",
            history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
        )

        expected = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Hello"},
        ]
        assert messages == expected

    def test_build_messages_no_system_or_history(self):
        """Test message building with just prompt."""
        messages = self.provider._build_messages("Hello")
        expected = [{"role": "user", "content": "Hello"}]
        assert messages == expected

    def test_generate(self):
        """Test basic text generation."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello, world!"
        self.mock_client.chat.completions.create.return_value = mock_response

        result = self.provider.generate("Say hello")
        assert result == "Hello, world!"

        # Verify API call
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "openai/gpt-4o"
        assert call_args["messages"] == [{"role": "user", "content": "Say hello"}]

    def test_generate_with_history(self):
        """Test text generation with conversation history."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "How can I help?"
        self.mock_client.chat.completions.create.return_value = mock_response

        history = [{"role": "user", "content": "Hi"}]
        result = self.provider.generate("Hello", history=history)
        assert result == "How can I help?"

        # Verify API call includes history
        call_args = self.mock_client.chat.completions.create.call_args[1]
        expected_messages = [
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": "Hello"},
        ]
        assert call_args["messages"] == expected_messages

    def test_generate_api_error(self):
        """Test error handling in generate method."""
        self.mock_client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(RuntimeError, match="OpenRouter API error"):
            self.provider.generate("Hello")

    def test_generate_structured(self):
        """Test structured output generation."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"name": "John", "age": 30}'
        self.mock_client.chat.completions.create.return_value = mock_response

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        }

        result = self.provider.generate_structured("Generate a person", schema)
        assert result == {"name": "John", "age": 30}

        # Verify API call includes response_format
        call_args = self.mock_client.chat.completions.create.call_args[1]
        assert "response_format" in call_args
        assert call_args["response_format"]["type"] == "json_schema"

    def test_generate_structured_json_error(self):
        """Test handling of invalid JSON in structured output."""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json"
        self.mock_client.chat.completions.create.return_value = mock_response

        schema = {"type": "object"}
        with pytest.raises(ValueError, match="Failed to parse structured response"):
            self.provider.generate_structured("Generate", schema)

    def test_generate_with_tools(self):
        """Test tool calling functionality."""
        # Mock response with tool call
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "test_function"
        mock_tool_call.function.arguments = '{"arg": "value"}'

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [mock_tool_call]
        self.mock_client.chat.completions.create.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "test_function"}}]
        result = self.provider.generate_with_tools("Use tool", tools)

        assert isinstance(result, dict)
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["id"] == "call_123"
        assert result["tool_calls"][0]["function"]["name"] == "test_function"

    def test_generate_with_tools_text_response(self):
        """Test tool calling with text response (no tools called)."""
        # Mock response without tool calls
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "I don't need to use any tools"
        mock_response.choices[0].message.tool_calls = None
        self.mock_client.chat.completions.create.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "test_function"}}]
        result = self.provider.generate_with_tools("Hello", tools)

        assert result == "I don't need to use any tools"

    def test_configure(self):
        """Test provider configuration."""
        config = {
            "model": "anthropic/claude-3-haiku",
            "temperature": 0.5,
            "max_tokens": 100,
        }

        self.provider.configure(config)

        assert self.provider.model == "anthropic/claude-3-haiku"
        assert self.provider.config["temperature"] == 0.5
        assert self.provider.config["max_tokens"] == 100

    def test_get_model_info(self):
        """Test model info retrieval."""
        info = self.provider.get_model_info()

        assert info["provider"] == "OpenRouter"
        assert info["model"] == "openai/gpt-4o"
        assert info["supports_structured"] is True
        assert info["supports_tools"] is True
        assert info["supports_history"] is True

    def test_implements_interface(self):
        """Test that OpenRouterProvider implements LLMInterface."""
        assert isinstance(self.provider, LLMInterface)

    def test_interface_methods_exist(self):
        """Test that all required interface methods are implemented."""
        assert hasattr(self.provider, "generate")
        assert hasattr(self.provider, "generate_structured")
        assert hasattr(self.provider, "generate_with_tools")
        assert hasattr(self.provider, "configure")
        assert hasattr(self.provider, "get_model_info")

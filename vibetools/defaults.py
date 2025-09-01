"""Global defaults for vibetools - simple configuration management.

This module provides a thread-safe global configuration system similar to 
matplotlib's rcParams, allowing users to set defaults that apply to all
LLM providers unless explicitly overridden.
"""

import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Type, TypeVar

# Avoid circular imports by using forward references  
T = TypeVar('T')

# Import interfaces for type hints
from vibetools.interfaces.llm import LLMInterface

# Global defaults registry - thread-safe configuration storage
_defaults_lock = threading.RLock()
_global_defaults: Dict[str, Any] = {
    # Core generation settings
    "model": "gpt-5-mini",
    "temperature": 1,
    "max_tokens": None,
    "timeout": 60,
    
    # Tool execution settings  
    "max_tool_iterations": 20,
    "tool_timeout": None,
    "handle_tool_errors": True,
    "tool_choice": "required",
    
    # Performance settings
    "max_retries": 3,
    "backoff_factor": 2.0,
    "request_timeout": 30,
    
    # Reasoning settings (gpt-5 and o-series models)
    "reasoning_effort": None,
    "reasoning_summary": None,
}


class DefaultsManager:
    """Thread-safe configuration manager for global defaults."""
    
    def __init__(self) -> None:
        self._lock = threading.RLock()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        with self._lock:
            return _global_defaults.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Args:
            key: Configuration key  
            value: Configuration value
        """
        with self._lock:
            _global_defaults[key] = value
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """Update multiple configuration values.
        
        Args:
            config_dict: Dictionary of configuration key-value pairs
        """
        with self._lock:
            _global_defaults.update(config_dict)
    
    def get_all(self) -> Dict[str, Any]:
        """Get a copy of all configuration values.
        
        Returns:
            Copy of all configuration settings
        """
        with self._lock:
            return _global_defaults.copy()
    
    def reset(self) -> None:
        """Reset all configuration to initial defaults."""
        with self._lock:
            _global_defaults.clear()
            _global_defaults.update({
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": None,
                "timeout": 30,
                "max_tool_iterations": 20,
                "tool_timeout": None, 
                "handle_tool_errors": True,
                "tool_choice": "required",
                "max_retries": 3,
                "backoff_factor": 2.0,
                "request_timeout": 30,
                "reasoning_effort": None,
                "reasoning_summary": None,
            })
    
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access for getting values."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Dictionary-style access for setting values."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists."""
        with self._lock:
            return key in _global_defaults
    
    def keys(self):
        """Get all configuration keys."""
        with self._lock:
            return _global_defaults.keys()
    
    def items(self):
        """Get all configuration key-value pairs."""
        with self._lock:
            return _global_defaults.items()


# Global configuration manager instance
defaults = DefaultsManager()

# Global cache for default provider
_default_provider_cache: Optional[LLMInterface] = None


def get_default_provider() -> LLMInterface:
    """Get cached default LLM provider using global configuration.
    
    Creates and caches a default provider instance with global configuration
    applied. The provider is cached for efficiency and reused across calls.
    
    Returns:
        Configured LLM provider instance using global defaults
        
    Examples:
        >>> provider = get_default_provider()
        >>> # Provider uses current global configuration
    """
    global _default_provider_cache
    if _default_provider_cache is None:
        # Import here to avoid circular dependencies at module level
        from vibetools.interfaces.openai_llm import OpenAIProvider
        _default_provider_cache = get_provider(OpenAIProvider)
    return _default_provider_cache


def reset_default_provider() -> None:
    """Reset cached default provider.
    
    Forces recreation of the default provider on next access. This is
    automatically called when global configuration changes via configure().
    
    Examples:
        >>> vibetools.configure(model="gpt-4o")  # Auto-resets cache
        >>> # Or manually:
        >>> vibetools.reset_default_provider()
    """
    global _default_provider_cache
    _default_provider_cache = None


def configure(**kwargs: Any) -> None:
    """Configure global defaults for all LLM providers.
    
    This function sets global default values that will be used by all
    new LLM provider instances unless explicitly overridden.
    Automatically resets the cached default provider.
    
    Args:
        **kwargs: Configuration options to set globally
        
    Examples:
        >>> import vibetools
        >>> vibetools.configure(temperature=0.8, max_tokens=2000)
        >>> vibetools.configure(model="gpt-4o", max_tool_iterations=15)
    """
    defaults.update(kwargs)
    reset_default_provider()  # Auto-invalidate cache when config changes


def reset_configuration() -> None:
    """Reset global configuration to initial defaults.
    
    Clears all global configuration settings and restores them to the
    original default values. Also resets the cached default provider.
    
    Examples:
        >>> import vibetools
        >>> vibetools.configure(temperature=0.1, model="gpt-4o")
        >>> # Later, reset to defaults
        >>> vibetools.reset_configuration()
        >>> # Now back to gpt-4o-mini with temperature=0.7
    """
    defaults.reset()
    reset_default_provider()  # Auto-invalidate cache when config resets


def get_provider(provider_class: Type[T], **kwargs: Any) -> T:
    """Create an LLM provider with global defaults applied.
    
    This factory function creates provider instances with global defaults
    automatically applied, while allowing per-instance overrides.
    
    Args:
        provider_class: The LLM provider class to instantiate
        **kwargs: Configuration overrides for this specific instance
        
    Returns:
        Configured LLM provider instance
        
    Examples:
        >>> from vibetools import OpenAIProvider, get_provider
        >>> # Uses global defaults
        >>> llm = get_provider(OpenAIProvider)
        >>> 
        >>> # Override specific settings
        >>> llm = get_provider(OpenAIProvider, temperature=0.9, model="gpt-4o")
    """
    import inspect
    
    # Get all global defaults
    config = defaults.get_all()
    
    # Override with instance-specific settings
    config.update(kwargs)
    
    # Filter config to only include parameters the provider accepts
    sig = inspect.signature(provider_class.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    
    # Only pass parameters the provider constructor accepts
    filtered_config = {k: v for k, v in config.items() if k in valid_params}
    
    # Create and return provider instance
    return provider_class(**filtered_config)


@contextmanager
def temp_config(**kwargs: Any) -> Iterator[None]:
    """Temporarily override global configuration settings.
    
    This context manager allows temporary changes to global defaults
    that are automatically restored when exiting the context.
    
    Args:
        **kwargs: Configuration overrides to apply temporarily
        
    Examples:
        >>> import vibetools
        >>> with vibetools.temp_config(temperature=0.1, max_tokens=500):
        ...     llm = vibetools.get_provider(vibetools.OpenAIProvider)
        ...     result = ai_ask("precise question", llm)
        >>> # Global config restored after context
    """
    # Save current state
    old_config = defaults.get_all()
    
    try:
        # Apply temporary overrides
        defaults.update(kwargs)
        yield
    finally:
        # Restore original state
        with defaults._lock:
            _global_defaults.clear()
            _global_defaults.update(old_config)
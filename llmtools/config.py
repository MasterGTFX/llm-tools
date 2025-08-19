"""Pydantic configuration models for llmtools components."""

from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _default_logging_config() -> "LoggingConfig":
    """Factory function for default LoggingConfig."""
    return LoggingConfig()  # type: ignore[call-arg]


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        "INFO", description="Logging level"
    )
    format: Optional[str] = Field(None, description="Custom log format string")
    handler_type: Literal["console", "file", "both"] = Field(
        "console", description="Where to send log messages"
    )
    log_file: Optional[str] = Field(
        None, description="Log file path (when using file handler)"
    )

    model_config = ConfigDict(use_enum_values=True)


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""

    provider: str
    model: Optional[str] = Field(None, description="Specific model name")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Custom base URL for API")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(
        None, gt=0, description="Maximum tokens to generate"
    )
    timeout: int = Field(30, gt=0, description="Request timeout in seconds")
    extra_params: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific parameters"
    )

    model_config = ConfigDict(extra="allow")


class KnowledgeBaseConfig(BaseModel):
    """Configuration for KnowledgeBase operations."""

    llm: LLMConfig
    instruction: str = Field(
        "Create a comprehensive knowledge base containing all useful information",
        description="Instruction for knowledge base creation",
    )
    output_dir: Optional[Union[str, Path]] = Field(
        None, description="Output directory for knowledge base"
    )
    history_dir: str = Field(
        ".history", description="Directory name for version history"
    )
    chunk_size: int = Field(
        4000, gt=0, description="Maximum chunk size for document processing"
    )
    chunk_overlap: int = Field(200, ge=0, description="Overlap between document chunks")
    max_versions: int = Field(
        10, gt=0, description="Maximum number of versions to keep"
    )
    logging: LoggingConfig = Field(
        default_factory=_default_logging_config, description="Logging configuration"
    )

    @field_validator("output_dir")
    def validate_output_dir(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v)


class SorterConfig(BaseModel):
    """Configuration for Sorter operations."""

    llm: LLMConfig
    mode: Literal["strict", "filter"] = Field("strict", description="Sorting mode")
    max_retries: int = Field(
        3, ge=0, description="Maximum retries for failed operations"
    )
    validate_output: bool = Field(True, description="Whether to validate output format")
    logging: LoggingConfig = Field(
        default_factory=_default_logging_config, description="Logging configuration"
    )

    model_config = ConfigDict(use_enum_values=True)


class DiffManagerConfig(BaseModel):
    """Configuration for diff management operations."""

    diff_format: Literal["unified", "context", "ndiff"] = Field(
        "unified", description="Format for diff output"
    )
    context_lines: int = Field(3, ge=0, description="Number of context lines in diffs")
    ignore_whitespace: bool = Field(
        False, description="Whether to ignore whitespace changes"
    )
    logging: LoggingConfig = Field(
        default_factory=_default_logging_config, description="Logging configuration"
    )

    model_config = ConfigDict(use_enum_values=True)


class ComponentConfig(BaseModel):
    """Base configuration that can be extended by components."""

    debug: bool = Field(False, description="Enable debug logging")
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_ttl: int = Field(3600, gt=0, description="Cache TTL in seconds")
    logging: LoggingConfig = Field(
        default_factory=_default_logging_config, description="Logging configuration"
    )

    model_config = ConfigDict(extra="allow")

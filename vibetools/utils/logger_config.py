"""Centralized logging utilities for vibetools components."""

import logging
import os
import sys
from typing import Optional


def setup_logger(
    name: str,
    level: Optional[str] = None,
    format_string: Optional[str] = None,
    handler_type: str = "console",
) -> logging.Logger:
    """Set up a logger for vibetools components.

    Args:
        name: Logger name (usually __name__ of the module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        handler_type: Type of handler ('console', 'file', or 'both')

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Set logging level (always update this regardless of existing handlers)
    log_level = _get_log_level(level)
    logger.setLevel(log_level)

    # Don't add handlers if they already exist
    if logger.handlers:
        return logger

    # Set format
    if format_string is None:
        format_string = _get_default_format()

    formatter = logging.Formatter(format_string, datefmt="%H:%M:%S")

    # Add console handler
    if handler_type in ("console", "both"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if requested
    if handler_type in ("file", "both"):
        log_file = os.getenv("VIBETOOLS_LOG_FILE", "vibetools.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger


def _get_log_level(level: Optional[str]) -> int:
    """Get logging level from string or environment.

    Args:
        level: Logging level string

    Returns:
        Logging level constant
    """
    # Check environment variable first
    if level is None:
        level = os.getenv("VIBETOOLS_LOG_LEVEL", "CRITICAL")

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_map.get(level.upper(), logging.INFO)


def _get_default_format() -> str:
    """Get default log format string.

    Returns:
        Format string for log messages
    """
    # Check for custom format from environment
    custom_format = os.getenv("VIBETOOLS_LOG_FORMAT")
    if custom_format:
        return custom_format

    # Compact format: [time] LEVEL component: message
    return "%(levelname)s [%(name)s] %(message)s"


def get_component_logger(component_name: str) -> logging.Logger:
    """Get a logger for a specific vibetools component.

    Args:
        component_name: Name of the component (e.g., 'knowledge_base', 'sorter')

    Returns:
        Configured logger for the component
    """
    logger_name = f"vibetools.{component_name}"
    return setup_logger(logger_name)

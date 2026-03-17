"""Logging configuration for cvat-ultralytics-bot.

This module provides a centralized logging setup for the entire application.
It uses Python's standard logging module with sensible defaults that can be
customized via environment variables or direct configuration.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

# Default log format
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log level environment variable
ENV_LOG_LEVEL = "CVAT_BOT_LOG_LEVEL"


def get_log_level(level: str | None = None) -> int:
    """Convert string log level to logging constant.

    Args:
        level: Log level string (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
               If None, reads from CVAT_BOT_LOG_LEVEL environment variable.

    Returns:
        Logging level constant (e.g., logging.DEBUG, logging.INFO).
    """
    if level is None:
        level = os.environ.get(ENV_LOG_LEVEL, "INFO")

    level_upper = level.upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_map.get(level_upper, logging.INFO)


def setup_logging(
    level: int | str | None = None,
    log_format: str | None = None,
    date_format: str | None = None,
    handlers: list[logging.Handler] | None = None,
) -> logging.Logger:
    """Configure logging for the application.

    This function sets up a root logger with a StreamHandler that outputs
    to stderr. The configuration can be customized via parameters or
    environment variables.

    Args:
        level: Log level as int or string. If None, uses CVAT_BOT_LOG_LEVEL
               env var or defaults to INFO.
        log_format: Format string for log messages. If None, uses DEFAULT_FORMAT.
        date_format: Date/time format string. If None, uses DEFAULT_DATE_FORMAT.
        handlers: Custom list of handlers. If None, creates a StreamHandler.

    Returns:
        The configured root logger.

    Example:
        >>> logger = setup_logging(level="DEBUG")
        >>> logger.debug("Debug message")
        2024-01-15 10:30:00 | DEBUG    | root | Debug message
    """
    # Determine log level
    if isinstance(level, str):
        level = get_log_level(level)
    elif level is None:
        level = get_log_level()

    # Use defaults if not provided
    if log_format is None:
        log_format = DEFAULT_FORMAT
    if date_format is None:
        date_format = DEFAULT_DATE_FORMAT

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to avoid duplicate logs
    if not handlers:
        handlers = [logging.StreamHandler(sys.stderr)]
        # Set formatter for each handler
        formatter = logging.Formatter(log_format, datefmt=date_format)
        for handler in handlers:
            handler.setFormatter(formatter)
            # Avoid duplicate handlers
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)
    else:
        # Use provided handlers
        for handler in handlers:
            handler.setLevel(level)
            if not handler.formatter:
                handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
            if handler not in root_logger.handlers:
                root_logger.addHandler(handler)

    # Suppress noisy third-party loggers by default
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("cvat_sdk").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name. If None, returns the root logger.
              Typically should be __name__ of the calling module.

    Returns:
        A logger instance with the specified name.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name)


# Convenience function to configure logging with a dictionary
def configure_logging_from_dict(config: dict[str, Any]) -> logging.Logger:
    """Configure logging from a dictionary.

    This is useful for programmatic configuration, e.g., from a config file.

    Args:
        config: Dictionary with keys 'level', 'format', 'date_format'.

    Returns:
        The configured root logger.
    """
    return setup_logging(
        level=config.get("level"),
        log_format=config.get("format"),
        date_format=config.get("date_format"),
    )

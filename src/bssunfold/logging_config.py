"""Logging configuration for bssunfold package.

This module sets up logging for the package with configurable verbosity.
"""

import logging
from typing import Optional

__all__ = [
    "get_logger",
    "setup_logging",
    "PACKAGE_LOGGER_NAME",
]

PACKAGE_LOGGER_NAME = "bssunfold"

# Module-level logger instance
_logger: Optional[logging.Logger] = None


def setup_logging(
    level: int = logging.WARNING,
    format_string: Optional[str] = None,
    use_handler: bool = False,
) -> logging.Logger:
    """Set up logging for the bssunfold package.
    
    Parameters
    ----------
    level : int, optional
        Logging level (default: logging.WARNING).
    format_string : str, optional
        Custom format string. If None, uses default format.
    use_handler : bool, optional
        If True, adds a StreamHandler to the logger.
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    global _logger
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    _logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    _logger.setLevel(level)
    
    # Avoid adding multiple handlers
    if use_handler and not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
    
    return _logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance.
    
    Parameters
    ----------
    name : str, optional
        Logger name. If None, returns package logger.
    
    Returns
    -------
    logging.Logger
        Logger instance.
    """
    global _logger
    
    if _logger is None:
        _logger = setup_logging()
    
    if name is None:
        return _logger
    
    return logging.getLogger(f"{PACKAGE_LOGGER_NAME}.{name}")


# Initialize default logger on module load
_logger = setup_logging()

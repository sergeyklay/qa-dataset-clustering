"""
Logging configuration for QA Dataset Clustering.
"""

import logging
from typing import Optional


def setup_logging(level: Optional[int] = None) -> None:
    """
    Set up logging configuration for the application.

    Args:
        level: The logging level (defaults to INFO if None)
    """
    if level is None:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: The name for the logger (typically __name__)

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)

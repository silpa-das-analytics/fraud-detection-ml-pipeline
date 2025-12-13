"""
Custom logging configuration for the fraud detection pipeline.
Provides structured logging with different levels for dev vs prod.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_level="INFO", log_to_file=True):
    """
    Configure logger with console and file handlers.

    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to write logs to file

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    # Format string - includes timestamp, level, module, and message
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler - always enabled
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - optional
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # Separate log file per day
        log_file = log_dir / f"fraud_detection_{datetime.now().strftime('%Y%m%d')}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Example usage
if __name__ == "__main__":
    test_logger = setup_logger(__name__, "DEBUG")
    test_logger.debug("This is a debug message")
    test_logger.info("Pipeline started")
    test_logger.warning("Missing configuration, using defaults")
    test_logger.error("Failed to load data file")

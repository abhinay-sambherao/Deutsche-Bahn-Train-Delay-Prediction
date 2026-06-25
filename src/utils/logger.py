"""
Logging configuration module.

Provides a centralized logger setup with console and file handlers,
formatted for both human readability and machine parsing.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional

from src.utils.config import CONFIG


def setup_logger(
    name: str = "db_delay_pipeline",
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger instance with console and file handlers.

    Args:
        name: Logger name.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file. If None, uses config default.

    Returns:
        Configured logger instance.
    """
    if level is None:
        level = CONFIG.log_level
    log_level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_file is None:
        log_dir = CONFIG.paths.logs_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = setup_logger()

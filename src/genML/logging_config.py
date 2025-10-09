"""
Centralized Logging Configuration for GenML Pipeline

This module sets up logging for the entire ML pipeline, directing all logs
to both console and timestamped log files in outputs/logs/.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class TeeStream:
    """Stream that writes to both original stream and a file"""
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, data):
        self.original_stream.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()


def setup_logging(dataset_name: Optional[str] = None, log_level: int = logging.INFO) -> str:
    """
    Configure logging for the ML pipeline with both console and file output.
    Captures ALL output including print statements and logging statements.

    Args:
        dataset_name: Optional name of the dataset being processed
        log_level: Logging level (default: INFO)

    Returns:
        Path to the log file created
    """
    # Create logs directory
    logs_dir = Path("outputs/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate descriptive timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dataset_name:
        log_filename = f"ml_pipeline_{dataset_name}_{timestamp}.log"
    else:
        log_filename = f"ml_pipeline_{timestamp}.log"

    log_filepath = logs_dir / log_filename

    # Open log file for all output
    log_file = open(log_filepath, 'w', encoding='utf-8', buffering=1)

    # Redirect stdout and stderr to capture print statements
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels in root logger

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        fmt='%(levelname)s:%(name)s:%(message)s'
    )

    # File handler - capture ALL logging messages (DEBUG and above)
    file_handler = logging.FileHandler(log_filepath, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Console handler - show INFO and above
    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Log initialization
    root_logger.info(f"Logging initialized: {log_filepath}")
    if dataset_name:
        root_logger.info(f"Dataset: {dataset_name}")

    return str(log_filepath)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Name of the module (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

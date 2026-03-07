"""
Logging utilities for the SGI project.
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from .paths import get_logs_dir


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_file: Optional log file name (will be placed in logs directory)
        level: Logging level
        console: Whether to also log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if specified
    if log_file:
        log_path = get_logs_dir() / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_experiment_logger(
    experiment_name: str,
    timestamp: Optional[str] = None
) -> logging.Logger:
    """
    Get a logger for an experiment with timestamped log file.
    
    Args:
        experiment_name: Name of the experiment
        timestamp: Optional timestamp (defaults to current time)
        
    Returns:
        Configured logger
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_file = f"{experiment_name}_{timestamp}.log"
    return setup_logger(experiment_name, log_file=log_file)


def log_config(logger: logging.Logger, config: dict, prefix: str = "") -> None:
    """
    Log configuration dictionary.
    
    Args:
        logger: Logger instance
        config: Configuration dictionary
        prefix: Prefix for log messages
    """
    logger.info(f"{prefix}Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for k, v in value.items():
                logger.info(f"    {k}: {v}")
        else:
            logger.info(f"  {key}: {value}")


def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame") -> None:
    """
    Log basic information about a DataFrame.
    
    Args:
        logger: Logger instance
        df: pandas DataFrame
        name: Name to use in log messages
    """
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"{name} columns: {list(df.columns)}")
    
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"{name} has {missing} missing values")


class ExperimentLogger:
    """Context manager for experiment logging."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = None
        self.start_time = None
    
    def __enter__(self) -> logging.Logger:
        self.logger = get_experiment_logger(self.experiment_name, self.timestamp)
        self.start_time = datetime.now()
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        self.logger.info(f"Timestamp: {self.timestamp}")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        if exc_type is not None:
            self.logger.error(f"Experiment failed with error: {exc_val}")
        else:
            self.logger.info(f"Experiment completed successfully")
        
        self.logger.info(f"Duration: {duration}")
        return False

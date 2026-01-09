"""
Logging utilities.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "gnn_depmap",
    log_file: str = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Parameters
    ----------
    name : str
        Logger name
    log_file : str, optional
        Path to log file
    level : int
        Logging level
    console : bool
        Whether to log to console
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

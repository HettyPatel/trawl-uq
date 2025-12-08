import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Setup logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level

    Returns:
        logger: Configured logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # clear any existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    #File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_timestamp():
    """
    Get current timestamp in YYYYMMDD_HHMMSS format.

    Returns:
        str: Current timestamp
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')
"""Module: logger."""

"""
Project-wide logging configuration.

Every module should import the logger using:

from src.utils.logger import get_logger

logger = get_logger(__name__)
"""

import logging
from pathlib import Path

from configs.settings import PROJECT_ROOT

# =============================================================================
# Log Directory
# =============================================================================

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "pipeline.log"

# =============================================================================
# Logger Configuration
# =============================================================================

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ],
)

# =============================================================================
# Logger Factory
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Usually pass __name__.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
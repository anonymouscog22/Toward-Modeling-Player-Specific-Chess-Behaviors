"""Utility helpers for logging and instrumentation.

This module provides a small convenience wrapper to obtain a module-scoped
logger configured with a standard timestamped format suitable for experiments
and reproducible runs.
"""

import logging


def getLogger():
    """Return a configured logger for the calling module.

    The returned logger uses a timestamped format and a default INFO level so
    that informational instrumentation is visible during normal runs while more
    verbose output may be enabled by setting the logging level externally.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    return logger

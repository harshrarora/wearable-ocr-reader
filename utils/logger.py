"""
Application-wide logging configuration.
"""

import logging
import sys
from config import LOG_LEVEL, LOG_FILE


def setup_logger():
    """Configure root logger with console + file output."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s [%(name)-22s] %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # — console —
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # — file —
    try:
        fh = logging.FileHandler(LOG_FILE, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass  # headless / read-only FS

    return root
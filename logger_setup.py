"""
logger_setup.py
---------------
Structured logging configuration for the AI Companion.

Provides a coloured console handler and an optional rotating file handler
so every module can do ``from logger_setup import get_logger`` and get a
nicely formatted, module-scoped logger.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "aicompanion.log")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_BYTES = 2 * 1024 * 1024   # 2 MB per file
BACKUP_COUNT = 3

_initialised = False


def _init_root_logger() -> None:
    """Set up handlers on the root logger (called once)."""
    global _initialised
    if _initialised:
        return
    _initialised = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── Console handler (INFO+) ──────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(console)

    # ── File handler (DEBUG+) ────────────────────────────────────────
    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root.addHandler(file_handler)

    # ── Silence noisy third-party loggers on console ─────────────────
    for name in ("httpx", "httpcore", "groq", "urllib3", "tensorflow", "absl"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, initialising the root logger if needed."""
    _init_root_logger()
    return logging.getLogger(name)

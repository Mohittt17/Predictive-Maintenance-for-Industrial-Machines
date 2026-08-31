"""
Structured logging for the Predictive Maintenance system.
Uses Python's standard logging with Rich formatting for human-readable output
and a plain file handler for persistent logs.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_CONFIGURED: set[str] = set()


def get_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger.  On first call the logger is configured;
    subsequent calls return the same instance unchanged.

    Args:
        name:     Module or component name (use ``__name__``).
        level:    Logging level (default INFO).
        log_file: Optional log-file name inside logs/.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # ── Console handler ───────────────────────────────────────────────────────
    if _RICH_AVAILABLE:
        console_handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s — %(message)s",
                              datefmt="%H:%M:%S")
        )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # ── File handler (always plain text) ─────────────────────────────────────
    fname = log_file or f"{name.replace('.', '_')}.log"
    file_handler = logging.FileHandler(_LOG_DIR / fname, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    _CONFIGURED.add(name)
    return logger

"""Application-wide logging setup."""

from __future__ import annotations

import logging
import sys

__all__ = ["setup_logging"]

_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3")

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger for the application.

    Idempotent — safe to call multiple times.
    """
    root = logging.getLogger()

    # Avoid adding duplicate handlers on repeated calls.
    if any(isinstance(h, logging.StreamHandler) and h.stream is sys.stdout for h in root.handlers):
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root.setLevel(numeric_level)
    root.addHandler(handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

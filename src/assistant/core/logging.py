"""Application-wide logging setup using rich for colored output."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

from assistant.core.ui import get_console

__all__ = ["setup_logging"]

_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "openai", "asyncio")

_LOG_FORMAT = "%(name)s | %(message)s"
_DATE_FORMAT = "[%X]"


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger with a RichHandler. Idempotent."""
    root = logging.getLogger()

    if any(isinstance(h, RichHandler) for h in root.handlers):
        return

    handler = RichHandler(
        console=get_console(),
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
        log_time_format=_DATE_FORMAT,
    )
    handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT))

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root.setLevel(numeric_level)
    root.addHandler(handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

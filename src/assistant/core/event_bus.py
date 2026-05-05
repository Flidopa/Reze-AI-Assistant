"""Async event bus for inter-module communication."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from assistant.core.events import Event

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Event)

Handler = Callable[[Any], Awaitable[None]]


class EventBus:
    """Simple async pub/sub bus. Handlers are called concurrently via asyncio.gather."""

    def __init__(self) -> None:
        self._handlers: dict[type, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: type[T], handler: Callable[[T], Awaitable[None]]) -> None:
        """Register *handler* to be called whenever *event_type* is emitted."""
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: type[T], handler: Callable[[T], Awaitable[None]]) -> None:
        """Remove *handler* from *event_type* subscribers. No-op if not subscribed."""
        handlers = self._handlers.get(event_type, [])
        with contextlib.suppress(ValueError):
            handlers.remove(handler)

    async def emit(self, event: Event) -> None:
        """Emit *event* to all subscribed handlers concurrently.

        A failing handler is logged at ERROR level and does not prevent other
        handlers from running.
        """
        event_type = type(event)
        handlers = list(self._handlers.get(event_type, []))
        logger.debug("emit %s → %d handler(s)", event_type.__name__, len(handlers))

        if not handlers:
            return

        results = await asyncio.gather(*[h(event) for h in handlers], return_exceptions=True)

        for handler, result in zip(handlers, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    "Handler %s raised an exception for event %s",
                    getattr(handler, "__qualname__", repr(handler)),
                    event_type.__name__,
                    exc_info=result,
                )

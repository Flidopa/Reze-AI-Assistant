"""ToolExecutor: subscribes to ToolCallRequested and dispatches to the registry."""

from __future__ import annotations

import logging

from assistant.core.event_bus import EventBus
from assistant.core.events import ToolCallCompleted, ToolCallRequested
from assistant.tools.registry import ToolRegistry

__all__ = ["ToolExecutor"]

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Listens for ToolCallRequested events, runs the matching tool, emits ToolCallCompleted."""

    def __init__(self, event_bus: EventBus, registry: ToolRegistry) -> None:
        self._event_bus = event_bus
        self._registry = registry

    def start(self) -> None:
        """Subscribe to ToolCallRequested. Call once at startup."""
        self._event_bus.subscribe(ToolCallRequested, self._handle_tool_call)

    async def _handle_tool_call(self, event: ToolCallRequested) -> None:
        tool = self._registry.get(event.tool_name)
        if tool is None:
            logger.warning("Unknown tool requested: %s", event.tool_name)
            await self._event_bus.emit(
                ToolCallCompleted(
                    tool_name=event.tool_name,
                    result=f"Unknown tool: {event.tool_name}",
                    success=False,
                )
            )
            return

        try:
            result = await tool.execute(**event.arguments)
            await self._event_bus.emit(
                ToolCallCompleted(
                    tool_name=event.tool_name,
                    result=result.content,
                    success=result.success,
                )
            )
        except Exception:
            logger.error("Tool '%s' raised an exception", event.tool_name, exc_info=True)
            await self._event_bus.emit(
                ToolCallCompleted(
                    tool_name=event.tool_name,
                    result=f"Tool {event.tool_name} failed",
                    success=False,
                )
            )

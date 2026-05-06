"""Tests for tools/executor.py."""

from __future__ import annotations

from typing import Any

import pytest

from assistant.core.event_bus import EventBus
from assistant.core.events import ToolCallCompleted, ToolCallRequested
from assistant.tools.base import Tool, ToolResult
from assistant.tools.executor import ToolExecutor
from assistant.tools.registry import ToolRegistry


class _EchoTool(Tool):
    """Returns whatever is passed as 'text', or 'echo' by default."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes text back"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(content=kwargs.get("text", "echo"))


class _FailingTool(Tool):
    @property
    def name(self) -> str:
        return "failing"

    @property
    def description(self) -> str:
        return "Always raises"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        raise RuntimeError("tool error")


@pytest.fixture()
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_EchoTool())
    reg.register(_FailingTool())
    return reg


@pytest.fixture()
def executor(event_bus: EventBus, registry: ToolRegistry) -> ToolExecutor:
    ex = ToolExecutor(event_bus, registry)
    ex.start()
    return ex


async def _collect_completed(event_bus: EventBus) -> list[ToolCallCompleted]:
    collected: list[ToolCallCompleted] = []

    async def _handler(e: ToolCallCompleted) -> None:
        collected.append(e)

    event_bus.subscribe(ToolCallCompleted, _handler)
    return collected


class TestSuccessfulExecution:
    async def test_emits_tool_call_completed(
        self, event_bus: EventBus, executor: ToolExecutor
    ) -> None:
        completed = await _collect_completed(event_bus)
        await event_bus.emit(ToolCallRequested(tool_name="echo", arguments={"text": "hi"}))
        assert len(completed) == 1

    async def test_result_content_correct(
        self, event_bus: EventBus, executor: ToolExecutor
    ) -> None:
        completed = await _collect_completed(event_bus)
        await event_bus.emit(ToolCallRequested(tool_name="echo", arguments={"text": "hello"}))
        assert completed[0].result == "hello"

    async def test_success_flag_true(self, event_bus: EventBus, executor: ToolExecutor) -> None:
        completed = await _collect_completed(event_bus)
        await event_bus.emit(ToolCallRequested(tool_name="echo", arguments={}))
        assert completed[0].success is True

    async def test_tool_name_preserved(self, event_bus: EventBus, executor: ToolExecutor) -> None:
        completed = await _collect_completed(event_bus)
        await event_bus.emit(ToolCallRequested(tool_name="echo", arguments={}))
        assert completed[0].tool_name == "echo"


class TestUnknownTool:
    async def test_emits_completed_with_failure(
        self, event_bus: EventBus, executor: ToolExecutor
    ) -> None:
        completed = await _collect_completed(event_bus)
        await event_bus.emit(ToolCallRequested(tool_name="ghost", arguments={}))
        assert len(completed) == 1
        assert completed[0].success is False

    async def test_result_mentions_tool_name(
        self, event_bus: EventBus, executor: ToolExecutor
    ) -> None:
        completed = await _collect_completed(event_bus)
        await event_bus.emit(ToolCallRequested(tool_name="ghost", arguments={}))
        assert "ghost" in completed[0].result


class TestToolException:
    async def test_emits_completed_with_failure(
        self, event_bus: EventBus, executor: ToolExecutor
    ) -> None:
        completed = await _collect_completed(event_bus)
        await event_bus.emit(ToolCallRequested(tool_name="failing", arguments={}))
        assert len(completed) == 1
        assert completed[0].success is False

    async def test_does_not_propagate_exception(
        self, event_bus: EventBus, executor: ToolExecutor
    ) -> None:
        # Should not raise
        await event_bus.emit(ToolCallRequested(tool_name="failing", arguments={}))

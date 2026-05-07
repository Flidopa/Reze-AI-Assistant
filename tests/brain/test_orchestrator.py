"""Tests for brain/orchestrator.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from assistant.brain.llm_client import LLMClient, LLMResponse, ToolCall
from assistant.brain.orchestrator import MAX_TOOL_ROUNDS, Orchestrator
from assistant.core.config import AssistantConfig
from assistant.core.event_bus import EventBus
from assistant.core.events import (
    LLMResponseReady,
    ToolCallCompleted,
    ToolCallRequested,
    UserSpeechDetected,
)
from assistant.tools.base import Tool, ToolResult
from assistant.tools.registry import ToolRegistry


def _make_config(**overrides: object) -> AssistantConfig:
    return AssistantConfig(openrouter_api_key="key", **overrides)  # type: ignore[arg-type]


def _text_llm(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], model="x-ai/grok-4-fast", usage_tokens=10)


def _tool_llm(name: str, args: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="tc1", name=name, arguments=args)],
        model="x-ai/grok-4-fast",
        usage_tokens=15,
    )


@pytest.fixture()
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def llm_mock() -> AsyncMock:
    return AsyncMock(spec=LLMClient)


@pytest.fixture()
def orchestrator(event_bus: EventBus, llm_mock: AsyncMock) -> Orchestrator:
    orc = Orchestrator(event_bus, llm_mock, _make_config())
    orc.start()
    return orc


class TestTextResponse:
    async def test_emits_llm_response_ready(
        self, event_bus: EventBus, orchestrator: Orchestrator, llm_mock: AsyncMock
    ) -> None:
        llm_mock.complete.return_value = _text_llm("Привет!")
        received: list[LLMResponseReady] = []
        event_bus.subscribe(LLMResponseReady, lambda e: received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        await event_bus.emit(UserSpeechDetected(text="Привет"))

        assert len(received) == 1
        assert received[0].text == "Привет!"

    async def test_no_tool_call_event_on_text(
        self, event_bus: EventBus, orchestrator: Orchestrator, llm_mock: AsyncMock
    ) -> None:
        llm_mock.complete.return_value = _text_llm("Ок")
        tc_received: list[ToolCallRequested] = []
        event_bus.subscribe(ToolCallRequested, lambda e: tc_received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        await event_bus.emit(UserSpeechDetected(text="hi"))

        assert tc_received == []


class TestHistoryAccumulation:
    async def test_history_grows_correctly(
        self, orchestrator: Orchestrator, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        llm_mock.complete.return_value = _text_llm("Ответ 1")
        await event_bus.emit(UserSpeechDetected(text="Вопрос 1"))

        llm_mock.complete.return_value = _text_llm("Ответ 2")
        await event_bus.emit(UserSpeechDetected(text="Вопрос 2"))

        # system + user1 + assistant1 + user2 + assistant2
        assert len(orchestrator._history) == 5
        assert orchestrator._history[1].content == "Вопрос 1"
        assert orchestrator._history[2].content == "Ответ 1"
        assert orchestrator._history[3].content == "Вопрос 2"
        assert orchestrator._history[4].content == "Ответ 2"

    async def test_system_message_always_first(
        self, orchestrator: Orchestrator, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        llm_mock.complete.return_value = _text_llm("ок")
        await event_bus.emit(UserSpeechDetected(text="test"))
        assert orchestrator._history[0].role == "system"


class TestHistoryTrimming:
    async def test_history_capped_at_max(self, event_bus: EventBus, llm_mock: AsyncMock) -> None:
        orc = Orchestrator(event_bus, llm_mock, _make_config(llm_max_history=3))
        orc.start()

        llm_mock.complete.return_value = _text_llm("r1")
        await event_bus.emit(UserSpeechDetected(text="q1"))  # system+user1+assistant1 = 3

        llm_mock.complete.return_value = _text_llm("r2")
        await event_bus.emit(UserSpeechDetected(text="q2"))  # would be 5 → trimmed to 3

        assert len(orc._history) == 3

    async def test_system_first_after_trim(self, event_bus: EventBus, llm_mock: AsyncMock) -> None:
        orc = Orchestrator(event_bus, llm_mock, _make_config(llm_max_history=3))
        orc.start()

        for i in range(4):
            llm_mock.complete.return_value = _text_llm(f"ans{i}")
            await event_bus.emit(UserSpeechDetected(text=f"q{i}"))

        assert orc._history[0].role == "system"
        assert len(orc._history) <= 3

    async def test_trim_never_starts_with_tool_role(
        self, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        """Trimming must not leave an orphaned tool message at index 1."""
        registry = _make_registry("dummy", "result")
        cfg = _make_config(llm_max_history=5)
        orc = Orchestrator(event_bus, llm_mock, cfg, tool_registry=registry)
        orc.start()

        # Turn 1: tool_call → tool_result → text (3 LLM calls worth)
        llm_mock.complete.side_effect = [
            _tool_llm("dummy", {}),
            _text_llm("done1"),
        ]
        await event_bus.emit(UserSpeechDetected(text="turn1"))
        # history: [system, user1, assistant_tc, tool, assistant_text] = 5

        # Turn 2: plain text response; this would push to 7 → trimmed to 5
        llm_mock.complete.return_value = _text_llm("done2")
        await event_bus.emit(UserSpeechDetected(text="turn2"))

        assert orc._history[0].role == "system"
        assert orc._history[1].role == "user"  # never a tool message at [1]


class TestToolCall:
    async def test_emits_tool_call_requested(
        self, orchestrator: Orchestrator, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        llm_mock.complete.return_value = _tool_llm("get_weather", {"city": "Moscow"})
        tc_received: list[ToolCallRequested] = []
        event_bus.subscribe(ToolCallRequested, lambda e: tc_received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        await event_bus.emit(UserSpeechDetected(text="погода?"))

        assert len(tc_received) == 1
        assert tc_received[0].tool_name == "get_weather"
        assert tc_received[0].arguments == {"city": "Moscow"}

    async def test_no_llm_response_ready_on_tool_call(
        self, orchestrator: Orchestrator, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        llm_mock.complete.return_value = _tool_llm("get_weather", {})
        lr_received: list[LLMResponseReady] = []
        event_bus.subscribe(LLMResponseReady, lambda e: lr_received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        await event_bus.emit(UserSpeechDetected(text="погода?"))

        assert lr_received == []


class TestErrorHandling:
    async def test_llm_error_does_not_propagate(
        self, orchestrator: Orchestrator, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        llm_mock.complete.side_effect = RuntimeError("boom")

        # Should not raise
        await event_bus.emit(UserSpeechDetected(text="hi"))

    async def test_history_unchanged_on_error(
        self, orchestrator: Orchestrator, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        llm_mock.complete.side_effect = RuntimeError("boom")
        history_len_before = len(orchestrator._history)

        await event_bus.emit(UserSpeechDetected(text="hi"))

        assert len(orchestrator._history) == history_len_before

    async def test_error_is_logged(
        self, orchestrator: Orchestrator, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        llm_mock.complete.side_effect = RuntimeError("boom")

        with patch("assistant.brain.orchestrator.logger") as mock_logger:
            await orchestrator._handle_user_speech(UserSpeechDetected(text="hi"))
            mock_logger.error.assert_called_once()


# ─── helpers ──────────────────────────────────────────────────────────────────


async def _noop() -> None:
    pass


def _make_registry(tool_name: str = "dummy", result: str = "ok") -> ToolRegistry:
    class _SimpleTool(Tool):
        @property
        def name(self) -> str:
            return tool_name

        @property
        def description(self) -> str:
            return "A simple test tool"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs: Any) -> ToolResult:
            return ToolResult(content=result)

    reg = ToolRegistry()
    reg.register(_SimpleTool())
    return reg


class TestMultiTurnToolLoop:
    async def test_tool_then_text_emits_response(
        self, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        registry = _make_registry("get_weather", "Солнечно, +20°C")
        orc = Orchestrator(event_bus, llm_mock, _make_config(), tool_registry=registry)
        orc.start()

        llm_mock.complete.side_effect = [
            _tool_llm("get_weather", {"city": "Moscow"}),
            _text_llm("В Москве солнечно, +20°C"),
        ]
        received: list[LLMResponseReady] = []
        event_bus.subscribe(LLMResponseReady, lambda e: received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        await event_bus.emit(UserSpeechDetected(text="Погода?"))

        assert len(received) == 1
        assert received[0].text == "В Москве солнечно, +20°C"

    async def test_tool_result_in_history(self, event_bus: EventBus, llm_mock: AsyncMock) -> None:
        registry = _make_registry("dummy", "result_text")
        orc = Orchestrator(event_bus, llm_mock, _make_config(), tool_registry=registry)
        orc.start()

        llm_mock.complete.side_effect = [
            _tool_llm("dummy", {}),
            _text_llm("Готово"),
        ]
        await event_bus.emit(UserSpeechDetected(text="go"))

        tool_msgs = [m for m in orc._history if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "result_text"

    async def test_assistant_tool_calls_message_in_history(
        self, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        registry = _make_registry("dummy", "r")
        orc = Orchestrator(event_bus, llm_mock, _make_config(), tool_registry=registry)
        orc.start()

        llm_mock.complete.side_effect = [_tool_llm("dummy", {}), _text_llm("OK")]
        await event_bus.emit(UserSpeechDetected(text="test"))

        assistant_tc_msgs = [m for m in orc._history if m.role == "assistant" and m.tool_calls]
        assert len(assistant_tc_msgs) == 1

    async def test_max_rounds_stops_loop(self, event_bus: EventBus, llm_mock: AsyncMock) -> None:
        registry = _make_registry("loop_tool", "keep going")
        orc = Orchestrator(event_bus, llm_mock, _make_config(), tool_registry=registry)
        orc.start()

        llm_mock.complete.return_value = _tool_llm("loop_tool", {})

        received: list[LLMResponseReady] = []
        event_bus.subscribe(LLMResponseReady, lambda e: received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        await event_bus.emit(UserSpeechDetected(text="spin"))

        assert received == []
        assert llm_mock.complete.call_count == MAX_TOOL_ROUNDS

    async def test_max_rounds_reverts_tool_history(
        self, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        """After hitting MAX_TOOL_ROUNDS, tool messages must be rolled back."""
        registry = _make_registry("loop_tool", "keep going")
        orc = Orchestrator(event_bus, llm_mock, _make_config(), tool_registry=registry)
        orc.start()

        llm_mock.complete.return_value = _tool_llm("loop_tool", {})
        history_len_before = len(orc._history)

        await event_bus.emit(UserSpeechDetected(text="spin"))

        tool_msgs = [m for m in orc._history if m.role in ("tool", "assistant") and m.tool_calls]
        assert tool_msgs == []
        # User message stays, tool exchanges reverted → +1 from initial state
        assert len(orc._history) == history_len_before + 1

    async def test_tool_completed_event_emitted(
        self, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        registry = _make_registry("my_tool", "the result")
        orc = Orchestrator(event_bus, llm_mock, _make_config(), tool_registry=registry)
        orc.start()

        llm_mock.complete.side_effect = [_tool_llm("my_tool", {}), _text_llm("done")]

        completed: list[ToolCallCompleted] = []
        event_bus.subscribe(ToolCallCompleted, lambda e: completed.append(e) or _noop())  # type: ignore[arg-type, return-value]

        await event_bus.emit(UserSpeechDetected(text="go"))

        assert len(completed) == 1
        assert completed[0].tool_name == "my_tool"
        assert completed[0].result == "the result"
        assert completed[0].success is True

    async def test_unknown_tool_returns_error_to_llm(
        self, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        registry = ToolRegistry()  # empty — no tools registered
        orc = Orchestrator(event_bus, llm_mock, _make_config(), tool_registry=registry)
        orc.start()

        llm_mock.complete.side_effect = [
            _tool_llm("ghost", {}),
            _text_llm("Инструмент не найден"),
        ]

        received: list[LLMResponseReady] = []
        event_bus.subscribe(LLMResponseReady, lambda e: received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        await event_bus.emit(UserSpeechDetected(text="test"))

        assert len(received) == 1

        tool_msgs = [m for m in orc._history if m.role == "tool"]
        assert "ghost" in tool_msgs[0].content  # type: ignore[operator]

    async def test_tool_call_requested_emitted_for_observability(
        self, event_bus: EventBus, llm_mock: AsyncMock
    ) -> None:
        """Registry path emits ToolCallRequested before executing each tool (UI hook)."""
        registry = _make_registry("dummy", "r")
        orc = Orchestrator(event_bus, llm_mock, _make_config(), tool_registry=registry)
        orc.start()

        llm_mock.complete.side_effect = [_tool_llm("dummy", {"x": 1}), _text_llm("OK")]

        tc_requested: list[ToolCallRequested] = []
        event_bus.subscribe(ToolCallRequested, lambda e: tc_requested.append(e) or _noop())  # type: ignore[arg-type, return-value]

        await event_bus.emit(UserSpeechDetected(text="test"))

        assert len(tc_requested) == 1
        assert tc_requested[0].tool_name == "dummy"
        assert tc_requested[0].arguments == {"x": 1}

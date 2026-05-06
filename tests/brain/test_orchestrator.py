"""Tests for brain/orchestrator.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from assistant.brain.llm_client import LLMClient, LLMResponse, ToolCall
from assistant.brain.orchestrator import Orchestrator
from assistant.core.config import AssistantConfig
from assistant.core.event_bus import EventBus
from assistant.core.events import LLMResponseReady, ToolCallRequested, UserSpeechDetected


def _make_config(**overrides: object) -> AssistantConfig:
    return AssistantConfig(openrouter_api_key="key", **overrides)  # type: ignore[arg-type]


def _text_llm(content: str) -> LLMResponse:
    return LLMResponse(content=content, tool_calls=[], model="gpt-4o-mini", usage_tokens=10)


def _tool_llm(name: str, args: dict[str, object]) -> LLMResponse:
    return LLMResponse(
        content=None,
        tool_calls=[ToolCall(id="tc1", name=name, arguments=args)],
        model="gpt-4o-mini",
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

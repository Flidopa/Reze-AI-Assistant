"""Tests for EventBus."""

import pytest

from assistant.core.event_bus import EventBus
from assistant.core.events import LLMResponseReady, UserSpeechDetected, WakeWordDetected


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


async def test_basic_emit(bus: EventBus) -> None:
    received: list[UserSpeechDetected] = []

    async def handler(event: UserSpeechDetected) -> None:
        received.append(event)

    bus.subscribe(UserSpeechDetected, handler)
    await bus.emit(UserSpeechDetected(text="hello"))

    assert len(received) == 1
    assert received[0].text == "hello"


async def test_multiple_handlers(bus: EventBus) -> None:
    calls: list[str] = []

    async def handler_a(event: WakeWordDetected) -> None:
        calls.append("a")

    async def handler_b(event: WakeWordDetected) -> None:
        calls.append("b")

    bus.subscribe(WakeWordDetected, handler_a)
    bus.subscribe(WakeWordDetected, handler_b)
    await bus.emit(WakeWordDetected())

    assert sorted(calls) == ["a", "b"]


async def test_handler_not_called_for_other_event_type(bus: EventBus) -> None:
    called = False

    async def handler(event: UserSpeechDetected) -> None:
        nonlocal called
        called = True

    bus.subscribe(UserSpeechDetected, handler)
    await bus.emit(WakeWordDetected())

    assert not called


async def test_unsubscribe(bus: EventBus) -> None:
    calls: list[int] = []

    async def handler(event: WakeWordDetected) -> None:
        calls.append(1)

    bus.subscribe(WakeWordDetected, handler)
    await bus.emit(WakeWordDetected())
    assert calls == [1]

    bus.unsubscribe(WakeWordDetected, handler)
    await bus.emit(WakeWordDetected())
    assert calls == [1]  # second emit did not reach handler


async def test_failing_handler_does_not_prevent_others(bus: EventBus) -> None:
    completed: list[str] = []

    async def bad_handler(event: LLMResponseReady) -> None:
        raise RuntimeError("boom")

    async def good_handler(event: LLMResponseReady) -> None:
        completed.append("good")

    bus.subscribe(LLMResponseReady, bad_handler)
    bus.subscribe(LLMResponseReady, good_handler)

    # Must not raise
    await bus.emit(LLMResponseReady(text="hi"))

    assert completed == ["good"]


async def test_emit_with_no_subscribers_does_not_raise(bus: EventBus) -> None:
    await bus.emit(WakeWordDetected())

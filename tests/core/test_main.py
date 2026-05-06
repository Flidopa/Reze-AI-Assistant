"""Tests for assistant/main.py."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant.core.config import get_config
from assistant.core.events import AssistantShutdown, AssistantStarted
from assistant.main import main


@pytest.fixture(autouse=True)
def _clear_config_cache() -> None:  # type: ignore[return]
    yield
    get_config.cache_clear()


@pytest.fixture(autouse=True)
def _patch_heavy_deps() -> None:  # type: ignore[return]
    """Prevent WhisperModel download and pvporcupine import during tests."""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], MagicMock())

    with (
        patch("assistant.audio.stt.ctranslate2") as mock_ct,
        patch("assistant.audio.stt.WhisperModel", return_value=mock_model),
        patch("assistant.audio.wake_word.WakeWordDetector.start"),
        patch("assistant.audio.wake_word.WakeWordDetector.stop"),
    ):
        mock_ct.get_cuda_device_count.return_value = 0
        yield


async def _run_then_cancel(coro: object) -> None:
    """Start a coroutine and cancel it after one event-loop tick."""
    task = asyncio.create_task(coro)  # type: ignore[arg-type]
    await asyncio.sleep(0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


class TestMainLifecycle:
    async def test_runs_and_shuts_down_cleanly(self) -> None:
        await _run_then_cancel(main())

    async def test_emits_assistant_started(self) -> None:
        emitted: list[object] = []

        from assistant.core.event_bus import EventBus

        with patch.object(EventBus, "emit", new_callable=AsyncMock) as mock_emit:
            task = asyncio.create_task(main())
            await asyncio.sleep(0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            for call in mock_emit.call_args_list:
                emitted.append(call.args[0])

        assert any(isinstance(e, AssistantStarted) for e in emitted)

    async def test_emits_assistant_shutdown_on_cancel(self) -> None:
        emitted: list[object] = []

        from assistant.core.event_bus import EventBus

        with patch.object(EventBus, "emit", new_callable=AsyncMock) as mock_emit:
            task = asyncio.create_task(main())
            await asyncio.sleep(0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            for call in mock_emit.call_args_list:
                emitted.append(call.args[0])

        assert any(isinstance(e, AssistantShutdown) for e in emitted)

    async def test_wake_detector_stop_called_on_shutdown(self) -> None:
        from assistant.audio.wake_word import WakeWordDetector

        with patch.object(WakeWordDetector, "stop") as mock_stop:
            task = asyncio.create_task(main())
            await asyncio.sleep(0)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        mock_stop.assert_called_once()

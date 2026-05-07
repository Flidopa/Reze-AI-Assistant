"""Tests for audio/tts.py."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest

from assistant.audio.tts import TTSEngine
from assistant.core.config import AssistantConfig
from assistant.core.event_bus import EventBus
from assistant.core.events import AudioPlaybackRequested, SpeechSynthesisRequested


def _make_config(**overrides: object) -> AssistantConfig:
    return AssistantConfig(openrouter_api_key="key", **overrides)  # type: ignore[arg-type]


async def _fake_stream(chunks: list[bytes]) -> AsyncGenerator[dict[str, object], None]:
    for data in chunks:
        yield {"type": "audio", "data": data}


def _patch_communicate(chunks: list[bytes]) -> MagicMock:
    """Return a patch target that makes Communicate().stream() yield *chunks*."""
    mock_comm = MagicMock()
    mock_comm.stream.return_value = _fake_stream(chunks)
    return mock_comm


@pytest.fixture()
def config() -> AssistantConfig:
    return _make_config()


@pytest.fixture()
def engine(config: AssistantConfig) -> TTSEngine:
    return TTSEngine(config)


@pytest.fixture()
def event_bus() -> EventBus:
    return EventBus()


class TestSynthesize:
    async def test_returns_bytes(self, engine: TTSEngine) -> None:
        fake_mp3 = b"\xff\xfb\x90\x00" + b"\x00" * 100
        _target = "assistant.audio.tts.edge_tts.Communicate"
        with patch(_target, return_value=_patch_communicate([fake_mp3])):
            result = await engine.synthesize("Привет")
        assert isinstance(result, bytes)
        assert result == fake_mp3

    async def test_concatenates_chunks(self, engine: TTSEngine) -> None:
        chunk_a = b"AAA"
        chunk_b = b"BBB"
        _target = "assistant.audio.tts.edge_tts.Communicate"
        with patch(_target, return_value=_patch_communicate([chunk_a, chunk_b])):
            result = await engine.synthesize("тест")
        assert result == chunk_a + chunk_b

    async def test_empty_text_does_not_raise(self, engine: TTSEngine) -> None:
        with patch("assistant.audio.tts.edge_tts.Communicate", return_value=_patch_communicate([])):
            result = await engine.synthesize("")
        assert isinstance(result, bytes)

    async def test_ignores_non_audio_chunks(self, engine: TTSEngine) -> None:
        async def _mixed_stream() -> AsyncGenerator[dict[str, object], None]:
            yield {"type": "WordBoundary", "data": b"meta"}
            yield {"type": "audio", "data": b"PCM"}

        mock_comm = MagicMock()
        mock_comm.stream.return_value = _mixed_stream()
        with patch("assistant.audio.tts.edge_tts.Communicate", return_value=mock_comm):
            result = await engine.synthesize("hi")
        assert result == b"PCM"


class TestSpeechSynthesisEvent:
    async def test_emits_audio_playback_requested(
        self, engine: TTSEngine, event_bus: EventBus
    ) -> None:
        engine.start(event_bus)
        received: list[AudioPlaybackRequested] = []
        event_bus.subscribe(AudioPlaybackRequested, lambda e: received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        fake_mp3 = b"\xff\xfb" + b"\x00" * 50
        with patch(
            "assistant.audio.tts.edge_tts.Communicate",
            return_value=_patch_communicate([fake_mp3]),
        ):
            await event_bus.emit(SpeechSynthesisRequested(text="Привет"))

        assert len(received) == 1
        assert isinstance(received[0].audio_data, bytes)
        assert len(received[0].audio_data) > 0

    async def test_no_playback_event_on_error(self, engine: TTSEngine, event_bus: EventBus) -> None:
        engine.start(event_bus)
        received: list[AudioPlaybackRequested] = []
        event_bus.subscribe(AudioPlaybackRequested, lambda e: received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        mock_comm = MagicMock()
        mock_comm.stream.side_effect = RuntimeError("network error")
        with patch("assistant.audio.tts.edge_tts.Communicate", return_value=mock_comm):
            await event_bus.emit(SpeechSynthesisRequested(text="тест"))

        assert received == []

    async def test_error_is_logged(self, engine: TTSEngine, event_bus: EventBus) -> None:
        engine.start(event_bus)

        mock_comm = MagicMock()
        mock_comm.stream.side_effect = RuntimeError("boom")
        with (
            patch("assistant.audio.tts.edge_tts.Communicate", return_value=mock_comm),
            patch("assistant.audio.tts.logger") as mock_logger,
        ):
            await engine._handle_synthesis_request(SpeechSynthesisRequested(text="x"))
            mock_logger.error.assert_called_once()


# ─── helpers ──────────────────────────────────────────────────────────────────


async def _noop() -> None:
    pass

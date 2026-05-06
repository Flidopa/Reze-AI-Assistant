"""Tests for audio/stt.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from assistant.audio.stt import STTEngine
from assistant.core.config import AssistantConfig
from assistant.core.event_bus import EventBus
from assistant.core.events import UserSpeechDetected, WakeWordDetected


def _make_config(**overrides: object) -> AssistantConfig:
    return AssistantConfig(openrouter_api_key="key", stt_model="base", **overrides)  # type: ignore[arg-type]


def _segment(text: str) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    return seg


@pytest.fixture()
def mock_model() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def engine(mock_model: MagicMock) -> STTEngine:
    with (
        patch("assistant.audio.stt.ctranslate2") as mock_ct,
        patch("assistant.audio.stt.WhisperModel", return_value=mock_model),
    ):
        mock_ct.get_cuda_device_count.return_value = 0
        return STTEngine(_make_config())


@pytest.fixture()
def event_bus() -> EventBus:
    return EventBus()


class TestTranscribe:
    def test_returns_text_from_segments(self, engine: STTEngine, mock_model: MagicMock) -> None:
        mock_model.transcribe.return_value = ([_segment("Привет, мир!")], MagicMock())
        audio = np.zeros(16000, dtype=np.float32)
        result = engine._transcribe(audio)
        assert result == "Привет, мир!"

    def test_silence_returns_none(self, engine: STTEngine, mock_model: MagicMock) -> None:
        mock_model.transcribe.return_value = ([], MagicMock())
        audio = np.zeros(16000, dtype=np.float32)
        result = engine._transcribe(audio)
        assert result is None

    def test_multiple_segments_joined(self, engine: STTEngine, mock_model: MagicMock) -> None:
        segments = [_segment("Привет,"), _segment("как дела?")]
        mock_model.transcribe.return_value = (segments, MagicMock())
        audio = np.zeros(16000, dtype=np.float32)
        result = engine._transcribe(audio)
        assert result == "Привет, как дела?"

    def test_whitespace_only_segment_returns_none(
        self, engine: STTEngine, mock_model: MagicMock
    ) -> None:
        mock_model.transcribe.return_value = ([_segment("   ")], MagicMock())
        audio = np.zeros(16000, dtype=np.float32)
        result = engine._transcribe(audio)
        assert result is None


class TestWakeWordHandling:
    async def test_emits_user_speech_detected(
        self, engine: STTEngine, event_bus: EventBus
    ) -> None:
        engine.start(event_bus)
        received: list[UserSpeechDetected] = []
        event_bus.subscribe(UserSpeechDetected, lambda e: received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        with patch.object(engine, "listen_and_transcribe", new_callable=AsyncMock) as mock_l:
            mock_l.return_value = "Включи музыку"
            await event_bus.emit(WakeWordDetected())

        assert len(received) == 1
        assert received[0].text == "Включи музыку"

    async def test_silence_does_not_emit(
        self, engine: STTEngine, event_bus: EventBus
    ) -> None:
        engine.start(event_bus)
        received: list[UserSpeechDetected] = []
        event_bus.subscribe(UserSpeechDetected, lambda e: received.append(e) or _noop())  # type: ignore[arg-type, return-value]

        with patch.object(engine, "listen_and_transcribe", new_callable=AsyncMock) as mock_l:
            mock_l.return_value = None
            await event_bus.emit(WakeWordDetected())

        assert received == []


class TestListenAndTranscribe:
    async def test_returns_transcribed_text(
        self, engine: STTEngine, mock_model: MagicMock
    ) -> None:
        mock_model.transcribe.return_value = ([_segment("Тест работает")], MagicMock())
        silence = np.zeros(int(engine._config.audio_listen_duration * 16000), dtype=np.float32)

        with patch.object(engine, "_record_audio", return_value=silence):
            result = await engine.listen_and_transcribe()

        assert result == "Тест работает"

    async def test_returns_none_on_silence(
        self, engine: STTEngine, mock_model: MagicMock
    ) -> None:
        mock_model.transcribe.return_value = ([], MagicMock())
        silence = np.zeros(16000, dtype=np.float32)

        with patch.object(engine, "_record_audio", return_value=silence):
            result = await engine.listen_and_transcribe()

        assert result is None


# ─── helpers ──────────────────────────────────────────────────────────────────

async def _noop() -> None:
    pass

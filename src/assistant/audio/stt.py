"""Speech-to-text transcription using faster-whisper. Push-to-talk recording."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import ctranslate2
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from assistant.core.config import AssistantConfig
from assistant.core.event_bus import EventBus
from assistant.core.events import (
    RecordingStarted,
    RecordingStopped,
    UserSpeechDetected,
    WakeWordDetected,
)

__all__ = ["STTEngine"]

logger = logging.getLogger(__name__)


class STTEngine:
    def __init__(self, config: AssistantConfig) -> None:
        self._config = config
        self._event_bus: EventBus | None = None

        self._model = self._load_model(config)

    @staticmethod
    def _load_model(config: AssistantConfig) -> WhisperModel:
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info("Loading Whisper '%s' on %s (%s)", config.stt_model, device, compute_type)
        model = WhisperModel(config.stt_model, device=device, compute_type=compute_type)

        if device == "cuda":
            # Smoke test: force one encoder pass to detect missing cuBLAS DLLs early.
            try:
                dummy = np.zeros(1600, dtype=np.float32)
                list(model.transcribe(dummy, language=config.stt_language)[0])
            except RuntimeError as exc:
                logger.warning("CUDA inference failed (%s) — falling back to CPU", exc)
                model = WhisperModel(config.stt_model, device="cpu", compute_type="int8")
                logger.info("Whisper reloaded on cpu (int8)")

        return model

    def start(self, event_bus: EventBus) -> None:
        """Subscribe to WakeWordDetected. Call once at startup."""
        self._event_bus = event_bus
        event_bus.subscribe(WakeWordDetected, self._handle_wake_word)

    async def _handle_wake_word(self, event: WakeWordDetected) -> None:
        assert self._event_bus is not None
        text = await self.listen_and_transcribe()
        if text:
            await self._event_bus.emit(UserSpeechDetected(text=text))

    async def listen_and_transcribe(self) -> str | None:
        """Push-to-talk: record until Enter is pressed, then transcribe."""
        assert self._event_bus is not None
        loop = asyncio.get_event_loop()

        await self._event_bus.emit(RecordingStarted())
        try:
            audio = await loop.run_in_executor(None, self._record_until_enter)
        finally:
            await self._event_bus.emit(RecordingStopped())

        if audio.size == 0:
            logger.info("No audio captured")
            return None

        t0 = time.monotonic()
        text = await loop.run_in_executor(None, self._transcribe, audio)
        logger.debug("Transcription in %.0f ms", (time.monotonic() - t0) * 1000)

        if text:
            logger.info("Transcribed: %s", text)
        else:
            logger.info("Silence detected")
        return text

    def _record_until_enter(self) -> np.ndarray:
        """Capture mic audio until the user presses Enter. Blocks the calling thread."""
        chunks: list[np.ndarray] = []
        stop_event = threading.Event()

        def on_audio(indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
            if status:
                logger.warning("Audio stream status: %s", status)
            chunks.append(indata.copy())

        def wait_for_enter() -> None:
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
            stop_event.set()

        enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
        enter_thread.start()

        with sd.InputStream(
            samplerate=self._config.audio_sample_rate,
            channels=1,
            dtype=np.float32,
            callback=on_audio,
        ):
            stop_event.wait()

        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).flatten()

    def _transcribe(self, audio: np.ndarray) -> str | None:
        segments, _ = self._model.transcribe(
            audio,
            language=self._config.stt_language,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text if text else None

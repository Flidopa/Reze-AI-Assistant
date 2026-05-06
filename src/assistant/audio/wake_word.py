"""Wake-word detection using Picovoice Porcupine."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

import sounddevice as sd

from assistant.core.config import AssistantConfig
from assistant.core.event_bus import EventBus
from assistant.core.events import WakeWordDetected

__all__ = ["WakeWordDetector"]

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """Background thread that listens for the wake word and emits WakeWordDetected."""

    def __init__(self, config: AssistantConfig) -> None:
        self._config = config
        self._event_bus: EventBus | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._porcupine: Any = None

    def start(self, event_bus: EventBus) -> None:
        """Start background detection. No-op if access key or pvporcupine is missing."""
        if not self._config.porcupine_access_key:
            logger.warning("PORCUPINE_ACCESS_KEY not set — wake word detection disabled")
            return

        try:
            import pvporcupine  # noqa: PLC0415
        except ImportError:
            logger.warning("pvporcupine not installed — wake word detection disabled")
            return

        self._event_bus = event_bus
        self._loop = asyncio.get_event_loop()

        if self._config.porcupine_keyword_path:
            self._porcupine = pvporcupine.create(
                access_key=self._config.porcupine_access_key,
                keyword_paths=[self._config.porcupine_keyword_path],
            )
            logger.info("Using custom wake word from: %s", self._config.porcupine_keyword_path)
        else:
            self._porcupine = pvporcupine.create(
                access_key=self._config.porcupine_access_key,
                keywords=[self._config.porcupine_keyword],
                sensitivities=[self._config.wake_word_sensitivity],
            )
            logger.info("Using built-in keyword: '%s'", self._config.porcupine_keyword)

        self._running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        logger.info("Wake word detector started")

    def stop(self) -> None:
        """Stop detection thread and release Porcupine resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._porcupine is not None:
            self._porcupine.delete()
            self._porcupine = None
        logger.info("Wake word detector stopped")

    def _detection_loop(self) -> None:
        assert self._porcupine is not None
        frame_length: int = self._porcupine.frame_length
        sample_rate: int = self._porcupine.sample_rate

        logger.debug("Wake word loop running (frame=%d sr=%d)", frame_length, sample_rate)

        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocksize=frame_length,
        ) as stream:
            while self._running:
                data, _ = stream.read(frame_length)
                pcm: list[int] = data[:, 0].tolist()
                result: int = self._porcupine.process(pcm)
                if result >= 0:
                    logger.info("Wake word detected!")
                    if self._loop is not None and self._event_bus is not None:
                        asyncio.run_coroutine_threadsafe(
                            self._event_bus.emit(WakeWordDetected()),
                            self._loop,
                        )

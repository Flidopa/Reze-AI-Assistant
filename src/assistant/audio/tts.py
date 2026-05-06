"""Text-to-speech synthesis using edge-tts (Microsoft Azure Neural voices)."""

from __future__ import annotations

import logging
import time

import edge_tts

from assistant.core.config import AssistantConfig
from assistant.core.event_bus import EventBus
from assistant.core.events import AudioPlaybackRequested, SpeechSynthesisRequested

__all__ = ["TTSEngine"]

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, config: AssistantConfig) -> None:
        self._config = config
        self._event_bus: EventBus | None = None

    def start(self, event_bus: EventBus) -> None:
        """Subscribe to synthesis events. Call once at startup."""
        self._event_bus = event_bus
        event_bus.subscribe(SpeechSynthesisRequested, self._handle_synthesis_request)

    async def _handle_synthesis_request(self, event: SpeechSynthesisRequested) -> None:
        assert self._event_bus is not None
        try:
            audio_data = await self.synthesize(event.text, event.emotion)
        except Exception:
            logger.error("TTS synthesis failed", exc_info=True)
            return
        await self._event_bus.emit(AudioPlaybackRequested(audio_data=audio_data))

    async def synthesize(self, text: str, emotion: str = "neutral") -> bytes:
        """Synthesise *text* → raw MP3 bytes via edge-tts."""
        logger.info("Synthesising %d chars", len(text))
        t0 = time.monotonic()

        communicate = edge_tts.Communicate(text=text, voice=self._config.tts_voice)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("Synthesis done in %.0f ms", elapsed_ms)
        return b"".join(chunks)

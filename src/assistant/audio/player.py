"""Audio playback via sounddevice + miniaudio MP3 decoding."""

from __future__ import annotations

import asyncio
import logging

import miniaudio
import numpy as np
import sounddevice as sd

from assistant.core.event_bus import EventBus
from assistant.core.events import AudioPlaybackRequested

__all__ = ["AudioPlayer"]

logger = logging.getLogger(__name__)


class AudioPlayer:
    def __init__(self) -> None:
        pass

    def start(self, event_bus: EventBus) -> None:
        """Subscribe to playback events. Call once at startup."""
        event_bus.subscribe(AudioPlaybackRequested, self._handle_playback_request)

    async def _handle_playback_request(self, event: AudioPlaybackRequested) -> None:
        try:
            await self.play(event.audio_data)
        except Exception:
            logger.error("Audio playback failed", exc_info=True)

    async def play(self, audio_data: bytes) -> None:
        """Decode MP3 *audio_data* and play it, awaiting completion."""
        decoded = miniaudio.decode(
            audio_data,
            output_format=miniaudio.SampleFormat.SIGNED16,
        )
        samples = np.frombuffer(decoded.samples, dtype=np.int16)
        if decoded.nchannels > 1:
            samples = samples.reshape(-1, decoded.nchannels)

        duration_s = len(decoded.samples) / (decoded.nchannels * decoded.sample_rate * 2)
        logger.info("Playing %.1f s of audio", duration_s)

        sd.play(samples, samplerate=decoded.sample_rate)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sd.wait)

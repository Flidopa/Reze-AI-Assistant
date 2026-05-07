"""Entry point: wires all modules to the event bus and runs the async loop."""

from __future__ import annotations

import asyncio
import logging

from assistant.audio.player import AudioPlayer
from assistant.audio.stt import STTEngine
from assistant.audio.tts import TTSEngine
from assistant.audio.wake_word import WakeWordDetector
from assistant.brain.llm_client import LLMClient
from assistant.brain.orchestrator import Orchestrator
from assistant.core.config import get_config
from assistant.core.event_bus import EventBus
from assistant.core.events import (
    AssistantShutdown,
    AssistantStarted,
    LLMResponseReady,
    SpeechSynthesisRequested,
)
from assistant.core.logging import setup_logging
from assistant.core.ui import ConsoleUI, get_console, print_banner
from assistant.tools.currency import CurrencyRateTool
from assistant.tools.datetime_tool import DateTimeTool
from assistant.tools.registry import ToolRegistry
from assistant.tools.weather import WeatherTool
from assistant.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


async def main() -> None:
    """Initialise all modules, wire them together, and run until cancelled."""
    config = get_config()
    print_banner(config.assistant_name)
    setup_logging(config.log_level)
    logger.info("Assistant starting...")

    # --- infrastructure ---
    event_bus = EventBus()
    console_ui = ConsoleUI(get_console(), config.assistant_name)
    console_ui.start(event_bus)  # subscribe first so spinner stops before TTS bridge runs

    # --- tools ---
    tool_registry = ToolRegistry()
    tool_registry.register(WebSearchTool())
    tool_registry.register(CurrencyRateTool())
    tool_registry.register(WeatherTool(config.openweather_api_key))
    tool_registry.register(DateTimeTool())

    # --- brain ---
    llm_client = LLMClient(config)
    orchestrator = Orchestrator(event_bus, llm_client, config, tool_registry=tool_registry)

    # --- audio ---
    stt_engine = STTEngine(config)
    tts_engine = TTSEngine(config)
    audio_player = AudioPlayer()
    wake_detector = WakeWordDetector(config)

    # --- subscriptions (order matters) ---
    orchestrator.start()  # UserSpeechDetected → LLMResponseReady
    stt_engine.start(event_bus)  # WakeWordDetected  → UserSpeechDetected
    tts_engine.start(event_bus)  # SpeechSynthesisRequested → AudioPlaybackRequested
    audio_player.start(event_bus)  # AudioPlaybackRequested → playback
    wake_detector.start(event_bus)  # background thread → WakeWordDetected

    # bridge: LLM text response → TTS
    async def _on_llm_response(event: LLMResponseReady) -> None:
        await event_bus.emit(SpeechSynthesisRequested(text=event.text))

    event_bus.subscribe(LLMResponseReady, _on_llm_response)

    await event_bus.emit(AssistantStarted())
    logger.info(
        "Assistant ready. Say '%s' to activate.",
        config.porcupine_keyword_path or config.porcupine_keyword,
    )

    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    finally:
        logger.info("Shutting down...")
        wake_detector.stop()
        await event_bus.emit(AssistantShutdown())
        logger.info("Goodbye.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")

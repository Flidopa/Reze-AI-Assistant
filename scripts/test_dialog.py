"""Ручной тест диалога через CLI. Запуск: uv run python scripts/test_dialog.py"""

from __future__ import annotations

import asyncio

from assistant.audio.player import AudioPlayer
from assistant.audio.stt import STTEngine
from assistant.audio.tts import TTSEngine
from assistant.brain.llm_client import LLMClient
from assistant.brain.orchestrator import Orchestrator
from assistant.core.config import get_config
from assistant.core.event_bus import EventBus
from assistant.core.events import (
    LLMResponseReady,
    SpeechSynthesisRequested,
    UserSpeechDetected,
    WakeWordDetected,
)
from assistant.core.logging import setup_logging
from assistant.tools.datetime_tool import DateTimeTool
from assistant.tools.registry import ToolRegistry
from assistant.tools.weather import WeatherTool
from assistant.tools.web_search import WebSearchTool


async def main() -> None:
    config = get_config()
    setup_logging(config.log_level)

    event_bus = EventBus()

    registry = ToolRegistry()
    registry.register(WebSearchTool())
    registry.register(WeatherTool(config.openweather_api_key))
    registry.register(DateTimeTool())

    llm_client = LLMClient(config)
    orchestrator = Orchestrator(event_bus, llm_client, config, tool_registry=registry)
    orchestrator.start()

    tts_engine = TTSEngine(config)
    tts_engine.start(event_bus)

    audio_player = AudioPlayer()
    audio_player.start(event_bus)

    stt_engine = STTEngine(config)
    stt_engine.start(event_bus)

    async def on_llm_response(event: LLMResponseReady) -> None:
        print(f"\nАссистент: {event.text}\n")
        await event_bus.emit(SpeechSynthesisRequested(text=event.text))

    event_bus.subscribe(LLMResponseReady, on_llm_response)

    print("Диалог с ассистентом")
    print("  [Enter]           — говорить голосом")
    print("  [текст + Enter]   — текстовый ввод")
    print("  [Ctrl+C]          — выход\n")

    while True:
        try:
            user_input = input("Ты: ").strip()  # noqa: ASYNC250
            if user_input == "":
                print("  Слушаю...")
                await event_bus.emit(WakeWordDetected())
            else:
                await event_bus.emit(UserSpeechDetected(text=user_input))
        except (KeyboardInterrupt, EOFError):
            break


asyncio.run(main())

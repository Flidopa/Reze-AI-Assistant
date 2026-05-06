"""Central orchestrator: routes UserSpeechDetected → LLM → events."""

from __future__ import annotations

import logging

from assistant.brain.llm_client import LLMClient, Message
from assistant.brain.prompts import build_system_prompt
from assistant.core.config import AssistantConfig
from assistant.core.event_bus import EventBus
from assistant.core.events import LLMResponseReady, ToolCallRequested, UserSpeechDetected

__all__ = ["Orchestrator"]

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        event_bus: EventBus,
        llm_client: LLMClient,
        config: AssistantConfig,
    ) -> None:
        self._event_bus = event_bus
        self._llm_client = llm_client
        self._config = config
        self._history: list[Message] = [
            LLMClient.system_message(build_system_prompt())
        ]

    def start(self) -> None:
        """Subscribe to events. Call once at startup."""
        self._event_bus.subscribe(UserSpeechDetected, self._handle_user_speech)

    async def _handle_user_speech(self, event: UserSpeechDetected) -> None:
        logger.info("User: %s", event.text)
        self._history.append(LLMClient.user_message(event.text))

        try:
            response = await self._llm_client.complete(messages=self._history)
        except Exception:
            logger.error("LLM request failed", exc_info=True)
            self._history.pop()
            return

        if response.tool_calls:
            for tc in response.tool_calls:
                logger.info("Tool call: %s args=%s", tc.name, tc.arguments)
                await self._event_bus.emit(
                    ToolCallRequested(tool_name=tc.name, arguments=tc.arguments)
                )
        elif response.content is not None:
            logger.info("Assistant: %s", response.content)
            self._history.append(LLMClient.assistant_message(response.content))
            await self._event_bus.emit(LLMResponseReady(text=response.content))

        self._trim_history()

    def _trim_history(self) -> None:
        max_h = self._config.llm_max_history
        if len(self._history) > max_h:
            self._history = [self._history[0]] + self._history[-(max_h - 1) :]

"""Central orchestrator: routes UserSpeechDetected → LLM → events."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from assistant.brain.llm_client import LLMClient, LLMResponse, ToolCall
from assistant.brain.prompts import build_system_prompt
from assistant.core.config import AssistantConfig
from assistant.core.event_bus import EventBus
from assistant.core.events import (
    LLMResponseReady,
    ToolCallCompleted,
    ToolCallRequested,
    UserSpeechDetected,
)

if TYPE_CHECKING:
    from assistant.tools.registry import ToolRegistry

__all__ = ["MAX_TOOL_ROUNDS", "Orchestrator"]

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 5


class Orchestrator:
    def __init__(
        self,
        event_bus: EventBus,
        llm_client: LLMClient,
        config: AssistantConfig,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._llm_client = llm_client
        self._config = config
        self._tool_registry = tool_registry
        self._history = [LLMClient.system_message(build_system_prompt())]

    def start(self) -> None:
        """Subscribe to events. Call once at startup."""
        self._event_bus.subscribe(UserSpeechDetected, self._handle_user_speech)

    async def _handle_user_speech(self, event: UserSpeechDetected) -> None:
        logger.info("User: %s", event.text)
        history_len_before = len(self._history)
        self._history.append(LLMClient.user_message(event.text))

        tools = self._tool_registry.all_schemas() if self._tool_registry else None
        response: LLMResponse | None = None

        try:
            if self._tool_registry is not None:
                response = await self._multi_turn_loop(tools)
            else:
                response = await self._single_turn()
        except Exception:
            logger.error("LLM request failed", exc_info=True)
            self._history = self._history[:history_len_before]
            return

        if response is not None and response.content is not None:
            logger.info("Assistant: %s", response.content)
            self._history.append(LLMClient.assistant_message(response.content))
            await self._event_bus.emit(LLMResponseReady(text=response.content))

        self._trim_history()

    async def _multi_turn_loop(self, tools: list[dict[str, Any]] | None) -> LLMResponse | None:
        """Run up to MAX_TOOL_ROUNDS of LLM → tool → LLM exchanges."""
        response: LLMResponse | None = None
        for _round in range(MAX_TOOL_ROUNDS):
            response = await self._llm_client.complete(
                messages=self._history,
                tools=tools,
            )
            if not response.tool_calls:
                break
            self._history.append(LLMClient.assistant_tool_calls_message(response.tool_calls))
            for tc in response.tool_calls:
                result = await self._execute_tool(tc)
                self._history.append(LLMClient.tool_message(tc.id, result))
        else:
            logger.warning("Max tool rounds (%d) reached without a text response", MAX_TOOL_ROUNDS)
        return response

    async def _single_turn(self) -> LLMResponse:
        """Legacy single-turn path (no tool registry): emit ToolCallRequested for each tool call."""
        response = await self._llm_client.complete(messages=self._history)
        if response.tool_calls:
            for tc in response.tool_calls:
                logger.info("Tool call: %s args=%s", tc.name, tc.arguments)
                await self._event_bus.emit(
                    ToolCallRequested(tool_name=tc.name, arguments=tc.arguments)
                )
        return response

    async def _execute_tool(self, tc: ToolCall) -> str:
        """Execute a single tool call, emit observability events, return result string."""
        assert self._tool_registry is not None
        tool = self._tool_registry.get(tc.name)
        if tool is None:
            logger.warning("Unknown tool: %s", tc.name)
            msg = f"Unknown tool: {tc.name}"
            await self._event_bus.emit(
                ToolCallCompleted(tool_name=tc.name, result=msg, success=False)
            )
            return msg

        try:
            result = await tool.execute(**tc.arguments)
            await self._event_bus.emit(
                ToolCallCompleted(tool_name=tc.name, result=result.content, success=result.success)
            )
            return result.content
        except Exception:
            logger.error("Tool '%s' raised an exception", tc.name, exc_info=True)
            msg = f"Tool {tc.name} failed"
            await self._event_bus.emit(
                ToolCallCompleted(tool_name=tc.name, result=msg, success=False)
            )
            return msg

    def _trim_history(self) -> None:
        max_h = self._config.llm_max_history
        if len(self._history) > max_h:
            self._history = [self._history[0]] + self._history[-(max_h - 1) :]

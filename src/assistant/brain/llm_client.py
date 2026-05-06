"""Async LLM client using OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import openai

from assistant.core.config import AssistantConfig

__all__ = ["LLMClient", "LLMResponse", "Message", "ToolCall"]

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    content: str
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    usage_tokens: int = 0


class LLMClient:
    def __init__(self, config: AssistantConfig) -> None:
        self._config = config
        self._client = openai.AsyncOpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.llm_base_url,
            timeout=float(config.llm_timeout),
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send messages to the LLM and return a structured response."""
        logger.debug(
            "LLM request: model=%s messages=%d tools=%s",
            self._config.llm_model,
            len(messages),
            bool(tools),
        )

        api_messages = [self._message_to_dict(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": self._config.llm_model,
            "messages": api_messages,
            "temperature": temperature,
            "max_tokens": self._config.llm_max_tokens,
        }
        if tools is not None:
            kwargs["tools"] = tools

        for attempt in range(2):
            try:
                response = await self._client.chat.completions.create(**kwargs)
                break
            except openai.RateLimitError:
                if attempt == 0:
                    logger.warning("Rate limit hit — retrying in 5s")
                    await asyncio.sleep(5)
                    continue
                raise
            except openai.APIError:
                logger.error("LLM API error", exc_info=True)
                raise

        choice = response.choices[0]
        usage_tokens = response.usage.total_tokens if response.usage else 0
        logger.debug("LLM response: tokens=%d", usage_tokens)

        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                import json

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
            return LLMResponse(
                content=None,
                tool_calls=tool_calls,
                model=response.model,
                usage_tokens=usage_tokens,
            )

        return LLMResponse(
            content=choice.message.content,
            tool_calls=[],
            model=response.model,
            usage_tokens=usage_tokens,
        )

    # --- helpers ---

    @staticmethod
    def system_message(content: str) -> Message:
        return Message(role="system", content=content)

    @staticmethod
    def user_message(content: str) -> Message:
        return Message(role="user", content=content)

    @staticmethod
    def assistant_message(content: str) -> Message:
        return Message(role="assistant", content=content)

    @staticmethod
    def _message_to_dict(msg: Message) -> dict[str, Any]:
        d: dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.tool_call_id is not None:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name is not None:
            d["name"] = msg.name
        return d

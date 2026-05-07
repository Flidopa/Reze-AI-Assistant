"""Async LLM client using OpenRouter (OpenAI-compatible API)."""

from __future__ import annotations

import asyncio
import json
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
    content: str | None
    tool_call_id: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


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
            default_headers={
                "HTTP-Referer": "https://github.com/Flidopa/Reze-AI-Assistant",
                "X-Title": "Reze AI-Assistant",
            },
        )

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.9,
        top_p: float = 0.95,
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
            "top_p": top_p,
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

    # --- static message constructors ---

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
    def tool_message(tool_call_id: str, content: str) -> Message:
        """Result message returned to the LLM after a tool execution."""
        return Message(role="tool", content=content, tool_call_id=tool_call_id)

    @staticmethod
    def assistant_tool_calls_message(tool_calls: list[ToolCall]) -> Message:
        """Assistant message that contains tool call requests (no text content)."""
        raw_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in tool_calls
        ]
        return Message(role="assistant", content=None, tool_calls=raw_calls)

    @staticmethod
    def _message_to_dict(msg: Message) -> dict[str, Any]:
        d: dict[str, Any] = {"role": msg.role}
        # Omit content entirely when it's None and we have tool_calls — some
        # providers (notably non-OpenAI ones via OpenRouter) treat explicit
        # null as a malformed message and ignore the following tool result.
        if msg.content is not None:
            d["content"] = msg.content
        elif msg.tool_calls is None:
            d["content"] = ""
        if msg.tool_call_id is not None:
            d["tool_call_id"] = msg.tool_call_id
        if msg.name is not None:
            d["name"] = msg.name
        if msg.tool_calls is not None:
            d["tool_calls"] = msg.tool_calls
        return d

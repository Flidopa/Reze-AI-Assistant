"""Tests for brain/llm_client.py."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import openai
import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from assistant.brain.llm_client import LLMClient, LLMResponse, Message, ToolCall
from assistant.core.config import AssistantConfig


def _make_config(**overrides: Any) -> AssistantConfig:
    return AssistantConfig(
        openrouter_api_key="test-key",
        llm_model="openai/gpt-4o-mini",
        llm_timeout=10,
        llm_max_tokens=256,
        **overrides,
    )


def _text_response(content: str, total_tokens: int = 15) -> ChatCompletion:
    return ChatCompletion(
        id="resp-1",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        created=1700000000,
        model="openai/gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=5, prompt_tokens=total_tokens - 5, total_tokens=total_tokens
        ),
    )


def _tool_response(tool_name: str, args: dict[str, Any], total_tokens: int = 30) -> ChatCompletion:
    return ChatCompletion(
        id="resp-2",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_abc",
                            type="function",
                            function=Function(
                                name=tool_name, arguments=json.dumps(args)
                            ),
                        )
                    ],
                ),
            )
        ],
        created=1700000000,
        model="openai/gpt-4o-mini",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=10, prompt_tokens=total_tokens - 10, total_tokens=total_tokens
        ),
    )


@pytest.fixture()
def client() -> LLMClient:
    return LLMClient(_make_config())


@pytest.fixture()
def mock_create(client: LLMClient) -> AsyncMock:
    mock = AsyncMock()
    client._client.chat.completions.create = mock  # type: ignore[method-assign]
    return mock


class TestTextResponse:
    async def test_content_populated(self, client: LLMClient, mock_create: AsyncMock) -> None:
        mock_create.return_value = _text_response("Hello, world!")
        result = await client.complete([LLMClient.user_message("hi")])
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello, world!"
        assert result.tool_calls == []

    async def test_model_and_tokens(self, client: LLMClient, mock_create: AsyncMock) -> None:
        mock_create.return_value = _text_response("ok", total_tokens=42)
        result = await client.complete([LLMClient.user_message("hi")])
        assert result.model == "openai/gpt-4o-mini"
        assert result.usage_tokens == 42


class TestToolCallResponse:
    async def test_content_is_none(self, client: LLMClient, mock_create: AsyncMock) -> None:
        mock_create.return_value = _tool_response("get_weather", {"city": "Moscow"})
        result = await client.complete([LLMClient.user_message("weather?")])
        assert result.content is None

    async def test_tool_calls_populated(self, client: LLMClient, mock_create: AsyncMock) -> None:
        mock_create.return_value = _tool_response("get_weather", {"city": "Moscow"})
        result = await client.complete([LLMClient.user_message("weather?")])
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.id == "call_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"city": "Moscow"}


class TestRetry:
    async def test_retries_on_rate_limit(self, client: LLMClient, mock_create: AsyncMock) -> None:
        mock_create.side_effect = [
            openai.RateLimitError(
                "rate limit",
                response=AsyncMock(status_code=429, headers={}),  # type: ignore[arg-type]
                body=None,
            ),
            _text_response("retried ok"),
        ]

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.complete([LLMClient.user_message("hi")])

        assert result.content == "retried ok"
        assert mock_create.call_count == 2
        mock_sleep.assert_awaited_once_with(5)

    async def test_raises_after_two_rate_limits(
        self, client: LLMClient, mock_create: AsyncMock
    ) -> None:
        err = openai.RateLimitError(
            "rate limit",
            response=AsyncMock(status_code=429, headers={}),  # type: ignore[arg-type]
            body=None,
        )
        mock_create.side_effect = [err, err]

        with patch("asyncio.sleep", new_callable=AsyncMock), pytest.raises(openai.RateLimitError):
            await client.complete([LLMClient.user_message("hi")])


class TestAPIError:
    async def test_propagates_api_error(self, client: LLMClient, mock_create: AsyncMock) -> None:
        mock_create.side_effect = openai.APIStatusError(
            "server error",
            response=AsyncMock(status_code=500, headers={}),  # type: ignore[arg-type]
            body=None,
        )
        with pytest.raises(openai.APIError):
            await client.complete([LLMClient.user_message("hi")])


class TestHelpers:
    def test_system_message(self) -> None:
        msg = LLMClient.system_message("Be helpful.")
        assert msg.role == "system"
        assert msg.content == "Be helpful."

    def test_user_message(self) -> None:
        msg = LLMClient.user_message("hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_assistant_message(self) -> None:
        msg = LLMClient.assistant_message("hi there")
        assert msg.role == "assistant"
        assert msg.content == "hi there"

    def test_message_is_dataclass(self) -> None:
        msg = Message(role="user", content="x")
        assert msg.tool_call_id is None
        assert msg.name is None


class TestTokenCount:
    async def test_usage_tokens_matches_mock(
        self, client: LLMClient, mock_create: AsyncMock
    ) -> None:
        mock_create.return_value = _text_response("answer", total_tokens=99)
        result = await client.complete([LLMClient.user_message("q")])
        assert result.usage_tokens == 99

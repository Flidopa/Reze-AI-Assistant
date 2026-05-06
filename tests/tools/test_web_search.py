"""Tests for tools/web_search.py."""

from __future__ import annotations

from typing import Any

import pytest

from assistant.tools.web_search import WebSearchTool

_FAKE_RESULTS: list[dict[str, Any]] = [
    {"title": "Заголовок 1", "body": "Описание 1", "href": "https://example.com/1"},
    {"title": "Заголовок 2", "body": "Описание 2", "href": "https://example.com/2"},
]


@pytest.fixture()
def tool() -> WebSearchTool:
    return WebSearchTool()


class TestSuccessfulSearch:
    async def test_success_flag_true(self, tool: WebSearchTool) -> None:
        tool._search = lambda q, n: _FAKE_RESULTS  # type: ignore[method-assign]
        result = await tool.execute(query="test")
        assert result.success is True

    async def test_result_contains_titles(self, tool: WebSearchTool) -> None:
        tool._search = lambda q, n: _FAKE_RESULTS  # type: ignore[method-assign]
        result = await tool.execute(query="test")
        assert "Заголовок 1" in result.content
        assert "Заголовок 2" in result.content

    async def test_result_contains_urls(self, tool: WebSearchTool) -> None:
        tool._search = lambda q, n: _FAKE_RESULTS  # type: ignore[method-assign]
        result = await tool.execute(query="test")
        assert "https://example.com/1" in result.content
        assert "https://example.com/2" in result.content

    async def test_result_contains_bodies(self, tool: WebSearchTool) -> None:
        tool._search = lambda q, n: _FAKE_RESULTS  # type: ignore[method-assign]
        result = await tool.execute(query="test")
        assert "Описание 1" in result.content


class TestEmptyResult:
    async def test_success_true_when_nothing_found(self, tool: WebSearchTool) -> None:
        tool._search = lambda q, n: []  # type: ignore[method-assign]
        result = await tool.execute(query="ничего")
        assert result.success is True

    async def test_content_mentions_not_found(self, tool: WebSearchTool) -> None:
        tool._search = lambda q, n: []  # type: ignore[method-assign]
        result = await tool.execute(query="ничего")
        assert "Ничего не найдено" in result.content


class TestMaxResults:
    async def test_max_results_capped_at_5(self, tool: WebSearchTool) -> None:
        captured: list[int] = []

        def fake(q: str, n: int) -> list[dict[str, Any]]:
            captured.append(n)
            return []

        tool._search = fake  # type: ignore[method-assign]
        await tool.execute(query="test", max_results=10)
        assert captured[0] == 5

    async def test_max_results_below_5_unchanged(self, tool: WebSearchTool) -> None:
        captured: list[int] = []

        def fake(q: str, n: int) -> list[dict[str, Any]]:
            captured.append(n)
            return []

        tool._search = fake  # type: ignore[method-assign]
        await tool.execute(query="test", max_results=2)
        assert captured[0] == 2

    async def test_default_max_results_is_3(self, tool: WebSearchTool) -> None:
        captured: list[int] = []

        def fake(q: str, n: int) -> list[dict[str, Any]]:
            captured.append(n)
            return []

        tool._search = fake  # type: ignore[method-assign]
        await tool.execute(query="test")
        assert captured[0] == 3


class TestNetworkError:
    async def test_returns_failure_on_exception(self, tool: WebSearchTool) -> None:
        def fail(q: str, n: int) -> list[dict[str, Any]]:
            raise RuntimeError("network error")

        tool._search = fail  # type: ignore[method-assign]
        result = await tool.execute(query="test")
        assert result.success is False

    async def test_does_not_propagate_exception(self, tool: WebSearchTool) -> None:
        def fail(q: str, n: int) -> list[dict[str, Any]]:
            raise RuntimeError("network error")

        tool._search = fail  # type: ignore[method-assign]
        result = await tool.execute(query="test")
        assert result is not None

    async def test_error_content_non_empty(self, tool: WebSearchTool) -> None:
        def fail(q: str, n: int) -> list[dict[str, Any]]:
            raise RuntimeError("timeout")

        tool._search = fail  # type: ignore[method-assign]
        result = await tool.execute(query="test")
        assert len(result.content) > 0


class TestSchema:
    def test_schema_name(self, tool: WebSearchTool) -> None:
        assert tool.to_openai_schema()["function"]["name"] == "web_search"

    def test_schema_type_function(self, tool: WebSearchTool) -> None:
        assert tool.to_openai_schema()["type"] == "function"

    def test_schema_has_parameters(self, tool: WebSearchTool) -> None:
        assert "parameters" in tool.to_openai_schema()["function"]

    def test_query_is_required(self, tool: WebSearchTool) -> None:
        required = tool.to_openai_schema()["function"]["parameters"]["required"]
        assert "query" in required

    def test_query_property_exists(self, tool: WebSearchTool) -> None:
        props = tool.to_openai_schema()["function"]["parameters"]["properties"]
        assert "query" in props
        assert props["query"]["type"] == "string"

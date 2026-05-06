"""Web search tool using DuckDuckGo (no API key required)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from assistant.tools.base import Tool, ToolResult

__all__ = ["WebSearchTool"]

logger = logging.getLogger(__name__)


class WebSearchTool(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Поиск актуальной информации в интернете. "
            "Используй когда нужны свежие новости, текущие цены, погода, "
            "факты которых может не быть в базе знаний, "
            "или любая информация требующая проверки."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Поисковый запрос на русском или английском языке",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Максимальное количество результатов (1-5)",
                    "default": 3,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        query: str = kwargs["query"]
        max_results: int = int(kwargs.get("max_results", 3))
        try:
            results = await asyncio.to_thread(self._search, query, min(max_results, 5))
        except Exception as exc:
            logger.error("Web search failed: %s", exc)
            return ToolResult(content=f"Ошибка поиска: {exc}", success=False)

        if not results:
            return ToolResult(content="Ничего не найдено по запросу.", success=True)

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. {r['title']}\n{r['body']}\nИсточник: {r['href']}")
        return ToolResult(content="\n\n".join(formatted), success=True)

    def _search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Synchronous DuckDuckGo search — runs in a thread via asyncio.to_thread."""
        from duckduckgo_search import DDGS  # noqa: PLC0415

        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

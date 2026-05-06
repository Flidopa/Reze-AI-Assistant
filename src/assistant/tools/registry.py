"""Tool registry: holds all registered tools and exposes their schemas to the LLM."""

from __future__ import annotations

from typing import Any

from assistant.tools.base import Tool

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """Container for Tool instances, keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool, overwriting any existing tool with the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the tool with the given name, or None if not found."""
        return self._tools.get(name)

    def all_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling schemas for all registered tools."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        names = list(self._tools)
        return f"ToolRegistry(tools={names!r})"

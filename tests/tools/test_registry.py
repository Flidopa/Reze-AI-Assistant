"""Tests for tools/registry.py."""

from __future__ import annotations

from typing import Any

import pytest

from assistant.tools.base import Tool, ToolResult
from assistant.tools.registry import ToolRegistry


class _NamedTool(Tool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Tool {self._name}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(content="ok")


@pytest.fixture()
def registry() -> ToolRegistry:
    return ToolRegistry()


@pytest.fixture()
def tool_a() -> _NamedTool:
    return _NamedTool("tool_a")


@pytest.fixture()
def tool_b() -> _NamedTool:
    return _NamedTool("tool_b")


class TestRegister:
    def test_register_and_get(self, registry: ToolRegistry, tool_a: _NamedTool) -> None:
        registry.register(tool_a)
        assert registry.get("tool_a") is tool_a

    def test_get_unknown_returns_none(self, registry: ToolRegistry) -> None:
        assert registry.get("nonexistent") is None

    def test_register_overwrites_same_name(self, registry: ToolRegistry) -> None:
        first = _NamedTool("dup")
        second = _NamedTool("dup")
        registry.register(first)
        registry.register(second)
        assert registry.get("dup") is second

    def test_multiple_tools(
        self, registry: ToolRegistry, tool_a: _NamedTool, tool_b: _NamedTool
    ) -> None:
        registry.register(tool_a)
        registry.register(tool_b)
        assert registry.get("tool_a") is tool_a
        assert registry.get("tool_b") is tool_b


class TestLen:
    def test_empty(self, registry: ToolRegistry) -> None:
        assert len(registry) == 0

    def test_after_register(
        self, registry: ToolRegistry, tool_a: _NamedTool, tool_b: _NamedTool
    ) -> None:
        registry.register(tool_a)
        assert len(registry) == 1
        registry.register(tool_b)
        assert len(registry) == 2

    def test_overwrite_does_not_increase_len(self, registry: ToolRegistry) -> None:
        registry.register(_NamedTool("same"))
        registry.register(_NamedTool("same"))
        assert len(registry) == 1


class TestAllSchemas:
    def test_empty_returns_empty_list(self, registry: ToolRegistry) -> None:
        assert registry.all_schemas() == []

    def test_schema_shape(self, registry: ToolRegistry, tool_a: _NamedTool) -> None:
        registry.register(tool_a)
        schemas = registry.all_schemas()
        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "tool_a"
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]

    def test_all_tools_represented(
        self, registry: ToolRegistry, tool_a: _NamedTool, tool_b: _NamedTool
    ) -> None:
        registry.register(tool_a)
        registry.register(tool_b)
        names = {s["function"]["name"] for s in registry.all_schemas()}
        assert names == {"tool_a", "tool_b"}

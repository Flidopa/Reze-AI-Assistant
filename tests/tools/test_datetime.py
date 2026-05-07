"""Tests for tools/datetime_tool.py."""

from __future__ import annotations

from datetime import datetime

import pytest

from assistant.tools.datetime_tool import DateTimeTool


@pytest.fixture()
def tool() -> DateTimeTool:
    return DateTimeTool()


def _patched(t: DateTimeTool, dt: datetime) -> None:
    t._now = lambda: dt  # type: ignore[method-assign]


class TestFormat:
    async def test_success_true(self, tool: DateTimeTool) -> None:
        result = await tool.execute()
        assert result.success is True

    async def test_format_example_wednesday_may(self, tool: DateTimeTool) -> None:
        # 2026-05-07 is a Thursday actually; use a known Wednesday: 2026-05-06
        _patched(tool, datetime(2026, 5, 6, 13, 4))
        result = await tool.execute()
        assert result.content == "Сейчас 6 мая 2026, среда, 13:04"

    async def test_day_no_leading_zero(self, tool: DateTimeTool) -> None:
        _patched(tool, datetime(2026, 1, 3, 9, 5))
        result = await tool.execute()
        assert " 3 января " in result.content

    async def test_time_has_leading_zero(self, tool: DateTimeTool) -> None:
        _patched(tool, datetime(2026, 6, 1, 7, 3))
        result = await tool.execute()
        assert "07:03" in result.content

    async def test_all_months(self, tool: DateTimeTool) -> None:
        expected = [
            "января", "февраля", "марта", "апреля", "мая", "июня",
            "июля", "августа", "сентября", "октября", "ноября", "декабря",
        ]
        for m, name in enumerate(expected, start=1):
            _patched(tool, datetime(2026, m, 15, 12, 0))
            result = await tool.execute()
            assert name in result.content

    async def test_weekday_monday(self, tool: DateTimeTool) -> None:
        _patched(tool, datetime(2026, 5, 4, 10, 0))  # Monday
        result = await tool.execute()
        assert "понедельник" in result.content

    async def test_weekday_sunday(self, tool: DateTimeTool) -> None:
        _patched(tool, datetime(2026, 5, 10, 10, 0))  # Sunday
        result = await tool.execute()
        assert "воскресенье" in result.content


class TestSchema:
    def test_name(self, tool: DateTimeTool) -> None:
        assert tool.name == "get_datetime"

    def test_no_required_params(self, tool: DateTimeTool) -> None:
        params = tool.to_openai_schema()["function"]["parameters"]
        assert params.get("required", []) == []

    def test_schema_type_function(self, tool: DateTimeTool) -> None:
        assert tool.to_openai_schema()["type"] == "function"

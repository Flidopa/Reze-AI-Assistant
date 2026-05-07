"""Datetime tool: returns current date and time formatted in Russian."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from assistant.tools.base import Tool, ToolResult

__all__ = ["DateTimeTool"]


_MONTHS_RU: tuple[str, ...] = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)

_WEEKDAYS_RU: tuple[str, ...] = (
    "понедельник", "вторник", "среда", "четверг",
    "пятница", "суббота", "воскресенье",
)


class DateTimeTool(Tool):
    @property
    def name(self) -> str:
        return "get_datetime"

    @property
    def description(self) -> str:
        return (
            "Получить текущую дату, день недели и время. "
            "Использовать для вопросов «какое сегодня число», «какой день недели», "
            "«сколько сейчас времени». Данные берутся локально — НЕ вызывай web_search."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> ToolResult:
        now = self._now()
        month = _MONTHS_RU[now.month - 1]
        weekday = _WEEKDAYS_RU[now.weekday()]
        content = (
            f"Сейчас {now.day} {month} {now.year}, {weekday}, "
            f"{now.hour:02d}:{now.minute:02d}"
        )
        return ToolResult(content=content, success=True)

    def _now(self) -> datetime:
        return datetime.now()

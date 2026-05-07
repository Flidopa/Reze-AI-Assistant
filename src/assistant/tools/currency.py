"""Currency rate tool using open.er-api.com (no API key required)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from assistant.tools.base import Tool, ToolResult

__all__ = ["CurrencyRateTool"]

logger = logging.getLogger(__name__)

_BASE_URL = "https://open.er-api.com/v6/latest"

_CURRENCY_NAMES: dict[str, str] = {
    "USD": "доллар США",
    "EUR": "евро",
    "GBP": "британский фунт",
    "CNY": "китайский юань",
    "JPY": "японская иена",
    "RUB": "российский рубль",
    "TRY": "турецкая лира",
    "BTC": "биткоин",
}


class CurrencyRateTool(Tool):
    @property
    def name(self) -> str:
        return "get_currency_rate"

    @property
    def description(self) -> str:
        return (
            "Получить актуальный курс валюты. "
            "Используй для вопросов о курсе доллара, евро, юаня "
            "и других валют к рублю или между собой."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "base": {
                    "type": "string",
                    "description": "Базовая валюта (ISO 4217, например USD, EUR, GBP)",
                    "default": "USD",
                },
                "target": {
                    "type": "string",
                    "description": "Целевая валюта (ISO 4217, например RUB, EUR, USD)",
                    "default": "RUB",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        base: str = kwargs.get("base", "USD").upper()
        target: str = kwargs.get("target", "RUB").upper()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{_BASE_URL}/{base}")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            logger.error("Currency API request failed: %s", exc)
            return ToolResult(content=f"Ошибка запроса к API курсов валют: {exc}", success=False)

        if data.get("result") != "success":
            msg = data.get("error-type", "unknown error")
            logger.error("Currency API returned error: %s", msg)
            return ToolResult(content=f"API вернул ошибку: {msg}", success=False)

        rates = data.get("rates", {})
        if target not in rates:
            return ToolResult(content=f"Валюта {target} не найдена в ответе API.", success=False)

        rate = rates[target]
        base_name = _CURRENCY_NAMES.get(base, base)
        target_name = _CURRENCY_NAMES.get(target, target)
        updated = data.get("time_last_update_utc", "неизвестно")

        content = (
            f"Курс {base_name} ({base}) к {target_name} ({target}): {rate:.4f}\n"
            f"Обновлено: {updated}"
        )
        return ToolResult(content=content, success=True)

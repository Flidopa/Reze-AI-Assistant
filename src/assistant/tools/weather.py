"""Weather tool using OpenWeather API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from assistant.tools.base import Tool, ToolResult

__all__ = ["WeatherTool"]

logger = logging.getLogger(__name__)

_OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


class WeatherTool(Tool):
    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return (
            "ТОЛЬКО для получения погоды и температуры в конкретном городе. "
            "Вызывать только если в запросе пользователя ЯВНО упомянуты: "
            "погода, температура, дождь, снег, осадки, прогноз, ветер, влажность, "
            "и при этом указан город. "
            "СТРОГО ЗАПРЕЩЕНО использовать для: курсов валют, доллара, евро, юаня, "
            "рубля, биткоина, любых финансовых данных, новостей, цен, фактов, "
            "времени, дат, поиска информации и любых других тем. "
            "Если пользователь не спросил про погоду — НЕ ВЫЗЫВАЙ этот инструмент."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Название города на английском или русском языке",
                },
            },
            "required": ["city"],
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        city: str | None = kwargs.get("city")
        if not city:
            return ToolResult(content="Ошибка: не указан город для запроса погоды.", success=False)

        if not self._api_key:
            return ToolResult(content="OpenWeather API key not configured", success=False)

        try:
            data = await self._fetch(city)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return ToolResult(content=f"Город '{city}' не найден", success=False)
            logger.error("Weather API HTTP error: %s", exc)
            return ToolResult(content=f"Ошибка API: {exc}", success=False)
        except Exception as exc:
            logger.error("Weather tool error: %s", exc)
            return ToolResult(content=f"Ошибка при запросе погоды: {exc}", success=False)

        content = (
            f"Погода в {data['name']}: "
            f"{data['weather'][0]['description']}, "
            f"температура {data['main']['temp']:.0f}°C "
            f"(ощущается как {data['main']['feels_like']:.0f}°C), "
            f"влажность {data['main']['humidity']}%, "
            f"ветер {data['wind']['speed']} м/с."
        )
        return ToolResult(content=content, success=True)

    async def _fetch(self, city: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                _OPENWEATHER_URL,
                params={
                    "q": city,
                    "appid": self._api_key,
                    "units": "metric",
                    "lang": "ru",
                },
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]

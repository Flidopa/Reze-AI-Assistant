"""Tests for tools/weather.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from assistant.tools.weather import WeatherTool

_FAKE_DATA: dict[str, Any] = {
    "name": "Moscow",
    "weather": [{"description": "ясно"}],
    "main": {"temp": 18.3, "feels_like": 16.5, "humidity": 60},
    "wind": {"speed": 3.5},
}


@pytest.fixture()
def tool() -> WeatherTool:
    return WeatherTool(api_key="test_key")


def _mock_fetch(data: dict[str, Any]):  # type: ignore[no-untyped-def]
    async def _fetch(city: str) -> dict[str, Any]:
        return data

    return _fetch


def _mock_fetch_http_error(status_code: int):  # type: ignore[no-untyped-def]
    async def _fetch(city: str) -> dict[str, Any]:
        mock_response = MagicMock()
        mock_response.status_code = status_code
        raise httpx.HTTPStatusError(
            f"{status_code}",
            request=MagicMock(),
            response=mock_response,
        )

    return _fetch


def _mock_fetch_exception():  # type: ignore[no-untyped-def]
    async def _fetch(city: str) -> dict[str, Any]:
        raise OSError("connection refused")

    return _fetch


class TestSuccess:
    async def test_success_flag(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch(_FAKE_DATA)  # type: ignore[method-assign]
        result = await tool.execute(city="Москва")
        assert result.success is True

    async def test_contains_city_name(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch(_FAKE_DATA)  # type: ignore[method-assign]
        result = await tool.execute(city="Москва")
        assert "Moscow" in result.content

    async def test_contains_temperature(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch(_FAKE_DATA)  # type: ignore[method-assign]
        result = await tool.execute(city="Москва")
        assert "°C" in result.content

    async def test_contains_humidity(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch(_FAKE_DATA)  # type: ignore[method-assign]
        result = await tool.execute(city="Москва")
        assert "влажность" in result.content
        assert "60" in result.content

    async def test_contains_wind(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch(_FAKE_DATA)  # type: ignore[method-assign]
        result = await tool.execute(city="Москва")
        assert "ветер" in result.content
        assert "3.5" in result.content

    async def test_contains_description(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch(_FAKE_DATA)  # type: ignore[method-assign]
        result = await tool.execute(city="Москва")
        assert "ясно" in result.content


class TestCityNotFound:
    async def test_success_false(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch_http_error(404)  # type: ignore[method-assign]
        result = await tool.execute(city="НесуществующийГород")
        assert result.success is False

    async def test_error_mentions_city(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch_http_error(404)  # type: ignore[method-assign]
        result = await tool.execute(city="НесуществующийГород")
        assert "НесуществующийГород" in result.content


class TestNoApiKey:
    async def test_success_false_without_key(self) -> None:
        tool = WeatherTool(api_key="")
        result = await tool.execute(city="Москва")
        assert result.success is False

    async def test_error_mentions_api_key(self) -> None:
        tool = WeatherTool(api_key="")
        result = await tool.execute(city="Москва")
        assert "API key" in result.content


class TestNetworkError:
    async def test_success_false_on_exception(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch_exception()  # type: ignore[method-assign]
        result = await tool.execute(city="Москва")
        assert result.success is False

    async def test_does_not_raise(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch_exception()  # type: ignore[method-assign]
        result = await tool.execute(city="Москва")
        assert result is not None

    async def test_content_non_empty(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch_exception()  # type: ignore[method-assign]
        result = await tool.execute(city="Москва")
        assert len(result.content) > 0

    async def test_http_error_non_404(self, tool: WeatherTool) -> None:
        tool._fetch = _mock_fetch_http_error(500)  # type: ignore[method-assign]
        result = await tool.execute(city="Москва")
        assert result.success is False


class TestSchema:
    def test_name(self, tool: WeatherTool) -> None:
        assert tool.name == "get_weather"

    def test_schema_type(self, tool: WeatherTool) -> None:
        assert tool.to_openai_schema()["type"] == "function"

    def test_schema_name(self, tool: WeatherTool) -> None:
        assert tool.to_openai_schema()["function"]["name"] == "get_weather"

    def test_city_is_required(self, tool: WeatherTool) -> None:
        required = tool.to_openai_schema()["function"]["parameters"]["required"]
        assert "city" in required

    def test_city_property_type(self, tool: WeatherTool) -> None:
        props = tool.to_openai_schema()["function"]["parameters"]["properties"]
        assert props["city"]["type"] == "string"

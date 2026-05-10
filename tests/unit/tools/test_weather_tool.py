"""Functional tests for :class:`tether.tools.weather_tool.GetWeatherTool`.

Style B Annotated migration (synthesis §4 Phase 4 step 43; A2 step 7).
Mocks ``requests.get`` to avoid real HTTP traffic; covers:

* Schema reports ``location`` (required) and ``unit`` (Literal enum).
* Successful path: geocoding + weather call composes correctly.
* Geocoding-not-found → tool returns ``{"error": ...}`` (the geocoding
  helper short-circuits before the weather call).
* Geocoding network failure → ``{"error": ...}``.
* Weather network failure → ``{"error": ...}``.
* Unit defaults to celsius and switches °F when fahrenheit is passed.
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch, MagicMock

import pytest
import requests

from tether.tools.weather_tool import GetWeatherTool


@pytest.fixture
def tool() -> GetWeatherTool:
    return GetWeatherTool()


def _geocoding_ok(lat: float = 51.5, lon: float = -0.12, name: str = "London", country: str = "UK") -> MagicMock:
    """Build a mock requests.Response for the geocoding API."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "results": [
            {
                "latitude": lat,
                "longitude": lon,
                "country": country,
                "name": name,
            }
        ]
    }
    return resp


def _weather_ok(temp: float = 14.5, wind: float = 12.3, code: int = 3) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "current_weather": {
            "temperature": temp,
            "windspeed": wind,
            "weathercode": code,
            "time": "2026-05-09T16:00",
        }
    }
    return resp


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestGetWeatherToolSchema:
    def test_schema_function_name(self, tool: GetWeatherTool):
        assert tool.auto_schema["function"]["name"] == "weather"

    def test_schema_location_required(self, tool: GetWeatherTool):
        params = tool.auto_schema["function"]["parameters"]
        assert "location" in params["required"]

    def test_schema_unit_enum(self, tool: GetWeatherTool):
        params = tool.auto_schema["function"]["parameters"]["properties"]
        assert set(params["unit"]["enum"]) == {"celsius", "fahrenheit"}
        assert params["unit"]["default"] == "celsius"


# ---------------------------------------------------------------------------
# Functional behavior — happy path
# ---------------------------------------------------------------------------


class TestGetWeatherToolHappyPath:
    @pytest.mark.asyncio
    async def test_invoke_with_location_returns_celsius_weather(self, tool: GetWeatherTool):
        with patch(
            "tether.tools.weather_tool.requests.get",
            side_effect=[_geocoding_ok(), _weather_ok(temp=14.5)],
        ):
            result = await tool.invoke({"location": "London"})

        assert result["location"] == "London"
        assert result["country"] == "UK"
        assert result["temperature"] == "14.5°C"
        assert "wind_speed" in result
        assert "weather_code" in result
        assert "last_updated" in result

    @pytest.mark.asyncio
    async def test_invoke_with_fahrenheit_unit(self, tool: GetWeatherTool):
        with patch(
            "tether.tools.weather_tool.requests.get",
            side_effect=[_geocoding_ok(), _weather_ok(temp=68.0)],
        ):
            result = await tool.invoke({"location": "London", "unit": "fahrenheit"})

        assert result["temperature"] == "68.0°F"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestGetWeatherToolErrors:
    @pytest.mark.asyncio
    async def test_unknown_location_returns_error_dict(self, tool: GetWeatherTool):
        empty_geocoding = MagicMock()
        empty_geocoding.raise_for_status = MagicMock()
        empty_geocoding.json.return_value = {"results": []}
        with patch(
            "tether.tools.weather_tool.requests.get",
            return_value=empty_geocoding,
        ):
            result = await tool.invoke({"location": "ZZZ-Nowhere-ZZZ"})

        assert "error" in result
        assert "ZZZ-Nowhere-ZZZ" in result["error"]

    @pytest.mark.asyncio
    async def test_geocoding_request_failure(self, tool: GetWeatherTool):
        with patch(
            "tether.tools.weather_tool.requests.get",
            side_effect=requests.RequestException("dns timeout"),
        ):
            result = await tool.invoke({"location": "London"})

        assert "error" in result
        assert "geocoding" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_weather_request_failure(self, tool: GetWeatherTool):
        with patch(
            "tether.tools.weather_tool.requests.get",
            side_effect=[_geocoding_ok(), requests.RequestException("upstream 502")],
        ):
            result = await tool.invoke({"location": "London"})

        assert "error" in result
        assert "weather" in result["error"].lower()

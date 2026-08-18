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

from unittest.mock import MagicMock, patch

import pytest
import requests

from tether.tools.weather_tool import GetWeatherTool, _normalize_unit


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

class TestUnitNormalization:
    """Models supply units the schema enum does not list.

    The ``Literal["celsius","fahrenheit"]`` annotation documents the contract
    but nothing enforces it at runtime, so an unrecognised value used to be
    forwarded verbatim to open-meteo and returned a 400. Observed in the wild:
    an 8B model answering "what's the weather in Dublin?" sent
    ``unit="metric"``, turning a working lookup into a failed turn.
    """

    @pytest.mark.parametrize(
        "supplied,expected",
        [
            ("metric", "celsius"),
            ("Metric", "celsius"),
            ("C", "celsius"),
            ("centigrade", "celsius"),
            ("imperial", "fahrenheit"),
            ("F", "fahrenheit"),
            ("FAHRENHEIT", "fahrenheit"),
            ("celsius", "celsius"),
            ("nonsense", "celsius"),
            ("", "celsius"),
            (None, "celsius"),
            (7, "celsius"),
        ],
    )
    def test_normalize_unit(self, supplied, expected):
        assert _normalize_unit(supplied) == expected

    @pytest.mark.asyncio
    async def test_unknown_unit_still_produces_a_reading(self, tool: GetWeatherTool):
        """A bad unit must degrade to celsius, not fail the turn."""
        with patch(
            "tether.tools.weather_tool.requests.get",
            side_effect=[_geocoding_ok(), _weather_ok()],
        ) as mock_get:
            result = await tool.invoke({"location": "London", "unit": "metric"})

        assert "error" not in result
        assert result["temperature"].endswith("\u00b0C")
        # The upstream call must carry a value open-meteo accepts.
        weather_params = mock_get.call_args_list[1].kwargs["params"]
        assert weather_params["temperature_unit"] == "celsius"

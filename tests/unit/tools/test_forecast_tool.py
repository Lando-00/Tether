"""Functional tests for :class:`tether.tools.weather_tool.GetForecastTool`.

Style B Annotated migration (synthesis §4 Phase 4 step 43; A2 step 7).
Mocks ``requests.get`` to avoid real HTTP traffic; covers:

* Schema reports ``location`` (required), ``days`` with ``ge=1`` /
  ``le=16`` Field bounds lifted into ``minimum`` / ``maximum``, and
  ``unit`` (Literal enum).
* Successful path: 3-day forecast composes correctly and returns a
  list of per-day dicts.
* Defensive ``days`` clamping: 0 → 1, 999 → 16.
* Geocoding-not-found → ``{"error": ...}``.
* Network failures surface as error dicts.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from tether.tools.weather_tool import GetForecastTool


@pytest.fixture
def tool() -> GetForecastTool:
    return GetForecastTool()


def _geocoding_ok(name: str = "London") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "results": [
            {
                "latitude": 51.5,
                "longitude": -0.12,
                "country": "UK",
                "name": name,
            }
        ]
    }
    return resp


def _forecast_ok(days: int) -> MagicMock:
    """Build a mock forecast response for ``days`` days."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "daily": {
            "time": [f"2026-05-{(9 + i):02d}" for i in range(days)],
            "temperature_2m_max": [18.0 + i for i in range(days)],
            "temperature_2m_min": [10.0 + i for i in range(days)],
            "weathercode": [3 for _ in range(days)],
            "precipitation_probability_max": [40 + i for i in range(days)],
        }
    }
    return resp


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestGetForecastToolSchema:
    def test_schema_function_name(self, tool: GetForecastTool):
        assert tool.auto_schema["function"]["name"] == "forecast"

    def test_schema_location_required(self, tool: GetForecastTool):
        params = tool.auto_schema["function"]["parameters"]
        assert "location" in params["required"]

    def test_schema_days_lifts_field_bounds(self, tool: GetForecastTool):
        """``Field(ge=1, le=16)`` should surface as
        ``minimum: 1, maximum: 16`` in the JSON schema."""
        params = tool.auto_schema["function"]["parameters"]["properties"]
        days = params["days"]
        assert days["type"] == "integer"
        assert days["minimum"] == 1
        assert days["maximum"] == 16
        assert days["default"] == 3

    def test_schema_unit_enum(self, tool: GetForecastTool):
        params = tool.auto_schema["function"]["parameters"]["properties"]
        assert set(params["unit"]["enum"]) == {"celsius", "fahrenheit"}


# ---------------------------------------------------------------------------
# Functional behavior — happy path
# ---------------------------------------------------------------------------


class TestGetForecastToolHappyPath:
    @pytest.mark.asyncio
    async def test_three_day_forecast(self, tool: GetForecastTool):
        with patch(
            "tether.tools.weather_tool.requests.get",
            side_effect=[_geocoding_ok(), _forecast_ok(3)],
        ):
            result = await tool.invoke({"location": "London", "days": 3})

        assert result["location"] == "London"
        assert result["country"] == "UK"
        assert result["forecast_days"] == 3
        assert result["unit"] == "celsius"
        assert len(result["forecasts"]) == 3
        first = result["forecasts"][0]
        assert "date" in first and "day_of_week" in first
        assert "min_temp" in first and "max_temp" in first
        assert "weather_code" in first
        assert "chance_of_precipitation" in first

    @pytest.mark.asyncio
    async def test_default_days_is_three(self, tool: GetForecastTool):
        with patch(
            "tether.tools.weather_tool.requests.get",
            side_effect=[_geocoding_ok(), _forecast_ok(3)],
        ):
            result = await tool.invoke({"location": "London"})

        assert result["forecast_days"] == 3


# ---------------------------------------------------------------------------
# Defensive clamping (Style B does NOT Pydantic-validate kwargs)
# ---------------------------------------------------------------------------


class TestGetForecastToolDaysClamping:
    @pytest.mark.asyncio
    async def test_days_zero_clamped_to_one(self, tool: GetForecastTool):
        """Style B path: invoke({"days": 0}) bypasses Pydantic; the
        ``run`` body's ``max(1, min(16, days))`` clamp must rescue it."""
        with patch(
            "tether.tools.weather_tool.requests.get",
            side_effect=[_geocoding_ok(), _forecast_ok(1)],
        ) as mock_get:
            result = await tool.invoke({"location": "London", "days": 0})

        # The forecast call should have used days=1 after the clamp.
        assert result["forecast_days"] == 1
        # Verify the request was issued with forecast_days=1.
        forecast_call = mock_get.call_args_list[1]
        assert forecast_call.kwargs["params"]["forecast_days"] == 1

    @pytest.mark.asyncio
    async def test_days_above_max_clamped_to_sixteen(self, tool: GetForecastTool):
        with patch(
            "tether.tools.weather_tool.requests.get",
            side_effect=[_geocoding_ok(), _forecast_ok(16)],
        ) as mock_get:
            result = await tool.invoke({"location": "London", "days": 999})

        assert result["forecast_days"] == 16
        forecast_call = mock_get.call_args_list[1]
        assert forecast_call.kwargs["params"]["forecast_days"] == 16


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestGetForecastToolErrors:
    @pytest.mark.asyncio
    async def test_unknown_location_returns_error(self, tool: GetForecastTool):
        empty_geocoding = MagicMock()
        empty_geocoding.raise_for_status = MagicMock()
        empty_geocoding.json.return_value = {"results": []}
        with patch(
            "tether.tools.weather_tool.requests.get",
            return_value=empty_geocoding,
        ):
            result = await tool.invoke({"location": "ZZZ-Nowhere-ZZZ", "days": 3})

        assert "error" in result
        assert "ZZZ-Nowhere-ZZZ" in result["error"]

    @pytest.mark.asyncio
    async def test_forecast_request_failure(self, tool: GetForecastTool):
        with patch(
            "tether.tools.weather_tool.requests.get",
            side_effect=[_geocoding_ok(), requests.RequestException("upstream 502")],
        ):
            result = await tool.invoke({"location": "London", "days": 3})

        assert "error" in result
        assert "forecast" in result["error"].lower()

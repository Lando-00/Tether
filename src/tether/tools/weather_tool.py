"""
weather_tool.py - Weather-related tools using real-time weather API.

This module provides weather-related tools that fetch live data from the
Open-Meteo API.

Style B (synthesis §4 Phase 4 step 43; A2 step 7): both ``GetWeatherTool``
and ``GetForecastTool`` use ``Annotated[T, Field(...)]`` parameters on
``run()`` so :meth:`BaseTool.auto_schema` lifts descriptions and bounds
into the JSON schema.
"""

from datetime import datetime
from typing import Annotated, Any, Dict, Literal

import requests
from pydantic import Field

from tether.tools.base import BaseTool
from tether.tools.registration import tool

# --- API Configuration ---
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"


def _get_location_lat_lon(location: str) -> Dict[str, Any]:
    """Helper to get latitude and longitude for a location."""
    params: Dict[str, Any] = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    try:
        response = requests.get(GEOCODING_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if not data.get("results"):
            return {"error": f"Location '{location}' not found."}

        result = data["results"][0]
        return {
            "latitude": result["latitude"],
            "longitude": result["longitude"],
            "country": result.get("country", "N/A"),
            "name": result["name"]
        }
    except requests.RequestException as e:
        return {"error": f"Failed to connect to geocoding service: {e}"}


# Synonyms models reach for instead of the schema's exact enum values. The
# ``Literal[...]`` annotation documents the contract but nothing enforces it at
# runtime, so an unrecognised value used to be forwarded verbatim to open-meteo
# and came back as a 400. Observed in the wild: an 8B model answering "what's
# the weather in Dublin?" sent ``unit="metric"``.
_UNIT_ALIASES: Dict[str, Literal["celsius", "fahrenheit"]] = {
    "celsius": "celsius", "c": "celsius", "°c": "celsius",
    "metric": "celsius", "centigrade": "celsius", "si": "celsius",
    "fahrenheit": "fahrenheit", "f": "fahrenheit", "°f": "fahrenheit",
    "imperial": "fahrenheit", "us": "fahrenheit",
}


def _normalize_unit(unit: Any) -> Literal["celsius", "fahrenheit"]:
    """Map a model-supplied unit onto open-meteo's accepted values.

    Falls back to ``celsius`` rather than raising: a wrong-but-plausible unit
    should not turn a working weather lookup into a failed turn.
    """
    if not isinstance(unit, str):
        return "celsius"
    return _UNIT_ALIASES.get(unit.strip().lower(), "celsius")


@tool(name="weather")
class GetWeatherTool(BaseTool):
    """Get the current weather conditions for a location."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        location: Annotated[
            str,
            Field(description="The city name."),
        ],
        unit: Annotated[
            Literal["celsius", "fahrenheit"],
            Field(description="The temperature unit (celsius or fahrenheit)."),
        ] = "celsius",
    ) -> Dict[str, Any]:
        """Get the current weather conditions for a location."""
        unit = _normalize_unit(unit)
        location_info = _get_location_lat_lon(location)
        if "error" in location_info:
            return location_info

        params: Dict[str, Any] = {
            "latitude": location_info["latitude"],
            "longitude": location_info["longitude"],
            "current_weather": "true",
            "temperature_unit": unit,
            "windspeed_unit": "kmh",
        }

        try:
            response = requests.get(WEATHER_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            current = data["current_weather"]
            temp_unit = "°C" if unit == "celsius" else "°F"

            return {
                "location": location_info["name"],
                "country": location_info["country"],
                "temperature": f"{current['temperature']}{temp_unit}",
                "wind_speed": f"{current['windspeed']} km/h",
                "weather_code": current['weathercode'],
                "last_updated": datetime.fromisoformat(current['time']).strftime("%Y-%m-%d %H:%M")
            }
        except requests.RequestException as e:
            return {"error": f"Failed to fetch weather data: {e}"}
        except (KeyError, IndexError):
            return {"error": "Could not parse weather data from API response."}


@tool(name="forecast")
class GetForecastTool(BaseTool):
    """Get a weather forecast for a location."""

    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        location: Annotated[
            str,
            Field(description="The city name."),
        ],
        days: Annotated[
            int,
            Field(
                ge=1,
                le=16,
                description="Number of days to forecast (1-16).",
            ),
        ] = 3,
        unit: Annotated[
            Literal["celsius", "fahrenheit"],
            Field(description="The temperature unit (celsius or fahrenheit)."),
        ] = "celsius",
    ) -> Dict[str, Any]:
        """Get a weather forecast for a location.

        ``days`` is clamped to the API's [1, 16] range defensively in
        case a Style B caller bypasses the schema bound.
        """
        unit = _normalize_unit(unit)
        location_info = _get_location_lat_lon(location)
        if "error" in location_info:
            return location_info

        # Defensive clamp — Style B does not Pydantic-validate at invoke
        # time, so a hand-rolled caller could pass days=0 or days=999.
        days = max(1, min(16, days))

        params: Dict[str, Any] = {
            "latitude": location_info["latitude"],
            "longitude": location_info["longitude"],
            "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": days,
            "temperature_unit": unit,
        }

        try:
            response = requests.get(WEATHER_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

            daily_data = data["daily"]
            temp_unit = "°C" if unit == "celsius" else "°F"

            forecasts = []
            for i in range(len(daily_data["time"])):
                forecasts.append({
                    "date": daily_data["time"][i],
                    "day_of_week": datetime.fromisoformat(daily_data["time"][i]).strftime("%A"),
                    "min_temp": f"{daily_data['temperature_2m_min'][i]}{temp_unit}",
                    "max_temp": f"{daily_data['temperature_2m_max'][i]}{temp_unit}",
                    "weather_code": daily_data['weathercode'][i],
                    "chance_of_precipitation": f"{daily_data['precipitation_probability_max'][i]}%"
                })

            return {
                "location": location_info["name"],
                "country": location_info["country"],
                "forecast_days": days,
                "unit": unit,
                "forecasts": forecasts,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        except requests.RequestException as e:
            return {"error": f"Failed to fetch forecast data: {e}"}
        except (KeyError, IndexError):
            return {"error": "Could not parse forecast data from API response."}

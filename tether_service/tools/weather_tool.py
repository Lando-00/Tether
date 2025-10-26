"""
weather_tool.py - Weather-related tools using real-time weather API.

This module provides weather-related tools that fetch live data from the
Open-Meteo API.
"""

from typing import Dict, Any, Literal
from datetime import datetime
import requests
from tether_service.tools.base import BaseTool


# --- API Configuration ---
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"


def _get_location_lat_lon(location: str) -> Dict[str, Any]:
    """Helper to get latitude and longitude for a location."""
    params = {"name": location, "count": 1, "language": "en", "format": "json"}
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


class GetWeatherTool(BaseTool):
    """Get the current weather conditions for a location."""
    
    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        location: str,
        unit: Literal["celsius", "fahrenheit"] = "celsius"
    ) -> Dict[str, Any]:
        """
        Get the current weather conditions for a location.
        
        Args:
            location: The city name.
            unit: The temperature unit (celsius or fahrenheit).
            
        Returns:
            Dictionary containing weather information.
        """
        location_info = _get_location_lat_lon(location)
        if "error" in location_info:
            return location_info

        params = {
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


class GetForecastTool(BaseTool):
    """Get a weather forecast for a location."""
    
    def __init__(self):
        super().__init__()

    @property
    def schema(self) -> Dict[str, Any]:
        return self.auto_schema

    async def run(
        self,
        location: str,
        days: int = 3,
        unit: Literal["celsius", "fahrenheit"] = "celsius"
    ) -> Dict[str, Any]:
        """
        Get a weather forecast for a location.
        
        Args:
            location: The city name.
            days: Number of days to forecast (1-16).
            unit: The temperature unit (celsius or fahrenheit).
            
        Returns:
            Dictionary containing forecast information.
        """
        location_info = _get_location_lat_lon(location)
        if "error" in location_info:
            return location_info

        # Limit days to API's max
        days = max(1, min(16, days))

        params = {
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


# Alias for backward compatibility with config
WeatherTool = GetWeatherTool
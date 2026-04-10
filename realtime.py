"""
realtime.py — CrashSense Live Weather & Simulation
Fetches live weather from OpenWeatherMap and applies user simulations.
"""

import os
import requests
import streamlit as st
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
#  OpenWeatherMap API Key
# ─────────────────────────────────────────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["OPENWEATHER_API_KEY"]
    except:
        return os.environ.get("OPENWEATHER_API_KEY", "YOUR_API_KEY_HERE")

_DEFAULT_API_KEY = get_api_key()

WEATHER_CONDITIONS = {
    "Clear":       {"Rain": False, "visibility_mi": 10.0},
    "Rain":        {"Rain": True,  "visibility_mi":  3.0},
    "Heavy Rain":  {"Rain": True,  "visibility_mi":  1.5},
    "Fog":         {"Rain": False, "visibility_mi":  0.5},
    "Snow":        {"Rain": False, "visibility_mi":  2.0},
    "Thunderstorm":{"Rain": True,  "visibility_mi":  1.0},
    "Cloudy":      {"Rain": False, "visibility_mi":  8.0},
}


# ─────────────────────────────────────────────
#  Live Weather
# ─────────────────────────────────────────────

def get_live_weather(city_name: str, api_key: str = None) -> dict:
    """
    Fetch current weather for a city using OpenWeatherMap API.

    Parameters
    ----------
    city_name : str
        City to look up (e.g. "Los Angeles").
    api_key : str, optional
        OWM API key. Falls back to env var OPENWEATHER_API_KEY.

    Returns
    -------
    dict with keys:
        city, description, temperature_c, humidity,
        visibility_mi, wind_speed_mph,
        Is_Rain (bool), Avg_Visibility (float),
        raw (full API response dict), error (str|None)
    """
    key = api_key or _DEFAULT_API_KEY

    if key == "YOUR_API_KEY_HERE":
        return _mock_weather(city_name)

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q":     city_name,
        "appid": key,
        "units": "metric",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Visibility: API returns metres, convert to miles
        vis_m  = data.get("visibility", 16093)   # default 10 mi
        vis_mi = round(vis_m / 1609.34, 2)

        condition = data["weather"][0]["main"]     # e.g. "Rain"
        is_rain   = "rain" in condition.lower() or "storm" in condition.lower()

        return {
            "city":            data.get("name", city_name),
            "description":     data["weather"][0]["description"].title(),
            "temperature_c":   data["main"]["temp"],
            "humidity":        data["main"]["humidity"],
            "visibility_mi":   vis_mi,
            "wind_speed_mph":  round(data["wind"].get("speed", 0) * 2.237, 1),
            "Is_Rain":         is_rain,
            "Avg_Visibility":  vis_mi,
            "raw":             data,
            "error":           None,
        }

    except requests.exceptions.ConnectionError:
        return _mock_weather(city_name, error="No internet connection – using mock data")
    except requests.exceptions.HTTPError as e:
        return _mock_weather(city_name, error=f"API error: {e} – using mock data")
    except Exception as e:
        return _mock_weather(city_name, error=f"Unexpected error: {e} – using mock data")


def _mock_weather(city_name: str, error: Optional[str] = None) -> dict:
    """Return plausible mock weather when API is unavailable."""
    return {
        "city":           city_name or "Unknown",
        "description":    "Clear Sky (Mock)",
        "temperature_c":  22.0,
        "humidity":       55,
        "visibility_mi":  10.0,
        "wind_speed_mph": 8.0,
        "Is_Rain":        False,
        "Avg_Visibility": 10.0,
        "raw":            {},
        "error":          error or "API key not set (using mock data)",
    }


# ─────────────────────────────────────────────
#  Simulation Helper
# ─────────────────────────────────────────────

def apply_simulation(
    hour: int,
    weather_condition: str,
    grid_mean_accident_count: float = 10.0,
) -> dict:
    """
    Convert UI slider / dropdown values into model-compatible feature overrides.

    Parameters
    ----------
    hour : int
        Hour of day selected by user (0–23).
    weather_condition : str
        One of the keys in WEATHER_CONDITIONS.
    grid_mean_accident_count : float
        Average accident count per grid cell — used to scale Rain_Accidents.

    Returns
    -------
    dict  suitable for passing as `user_inputs` to predict_real_time_risk().
    """
    cond = WEATHER_CONDITIONS.get(weather_condition, WEATHER_CONDITIONS["Clear"])

    # Rain_Accidents: if raining, simulate ~30% of accidents being rain-related
    rain_factor   = 0.30 if cond["Rain"] else 0.02
    rain_accidents = round(grid_mean_accident_count * rain_factor, 1)

    return {
        "Avg_Hour":       float(hour),
        "Rain_Accidents": rain_accidents,
        "Avg_Visibility": cond["visibility_mi"],
    }


def weather_condition_from_live(weather_data: dict) -> str:
    """
    Map a live-weather dict back to one of our display condition strings.
    """
    desc = weather_data.get("description", "").lower()
    if "thunderstorm" in desc:
        return "Thunderstorm"
    if "heavy rain" in desc or "shower" in desc:
        return "Heavy Rain"
    if "rain" in desc or "drizzle" in desc:
        return "Rain"
    if "fog" in desc or "mist" in desc or "haze" in desc:
        return "Fog"
    if "snow" in desc or "sleet" in desc:
        return "Snow"
    if "cloud" in desc or "overcast" in desc:
        return "Cloudy"
    return "Clear"

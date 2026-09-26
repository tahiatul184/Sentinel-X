"""Keyless weather context for AEROSENTINEL v2.8.0.

The service uses Open-Meteo's public forecast endpoint by default. Weather
values are context for resilience screening only; they are not certified
aviation weather, METAR/TAF, or a substitute for official flight-weather
products.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Callable, Mapping

OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
CURRENT_FIELDS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "rain",
    "weather_code", "cloud_cover", "wind_speed_10m", "wind_gusts_10m",
    "visibility",
]
HOURLY_FIELDS = [
    "temperature_2m", "precipitation", "rain", "weather_code", "cloud_cover",
    "visibility", "wind_speed_10m", "wind_gusts_10m", "cape",
]


class WeatherServiceError(RuntimeError):
    pass


def _finite(value, default=None):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _iso_utc(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _hourly_rows(payload: Mapping):
    hourly = payload.get("hourly") or {}
    times = list(hourly.get("time") or [])
    rows = []
    for i, raw in enumerate(times):
        row = {"time": _iso_utc(raw)}
        for field in HOURLY_FIELDS:
            values = hourly.get(field) or []
            row[field] = _finite(values[i]) if i < len(values) else None
        rows.append(row)
    return rows


def _max(rows, field, default=None):
    vals = [r.get(field) for r in rows if r.get(field) is not None]
    return max(vals) if vals else default


def _min(rows, field, default=None):
    vals = [r.get(field) for r in rows if r.get(field) is not None]
    return min(vals) if vals else default


def _sum(rows, field):
    return float(sum(r.get(field) or 0.0 for r in rows))


def summarize_weather(payload: Mapping, fetched_at: str | None = None):
    current = dict(payload.get("current") or {})
    current_time = _iso_utc(current.get("time"))
    rows = _hourly_rows(payload)
    now = datetime.now(timezone.utc)
    if current_time:
        try:
            now = datetime.fromisoformat(current_time)
        except ValueError:
            pass
    future = []
    for row in rows:
        if not row.get("time"):
            continue
        try:
            dt = datetime.fromisoformat(row["time"])
        except ValueError:
            continue
        if dt >= now.replace(minute=0, second=0, microsecond=0):
            future.append(row)
    next6 = future[:6]
    next24 = future[:24]
    thunder_codes = {95, 96, 99}
    thunder_hours = sum(1 for r in next24 if int(r.get("weather_code") or -1) in thunder_codes)
    summary = {
        "current_time": current_time,
        "temperature_c": _finite(current.get("temperature_2m")),
        "relative_humidity_percent": _finite(current.get("relative_humidity_2m")),
        "current_precipitation_mm": _finite(current.get("precipitation"), 0.0),
        "current_rain_mm": _finite(current.get("rain"), 0.0),
        "current_weather_code": _finite(current.get("weather_code")),
        "current_cloud_cover_percent": _finite(current.get("cloud_cover")),
        "current_wind_kmh": _finite(current.get("wind_speed_10m")),
        "current_gust_kmh": _finite(current.get("wind_gusts_10m")),
        "current_visibility_m": _finite(current.get("visibility")),
        "rain_next_6h_mm": _sum(next6, "precipitation"),
        "rain_next_24h_mm": _sum(next24, "precipitation"),
        "max_wind_next_6h_kmh": _max(next6, "wind_speed_10m"),
        "max_gust_next_6h_kmh": _max(next6, "wind_gusts_10m"),
        "min_visibility_next_6h_m": _min(next6, "visibility"),
        "max_cloud_next_6h_percent": _max(next6, "cloud_cover"),
        "max_temperature_next_24h_c": _max(next24, "temperature_2m"),
        "max_cape_next_24h_jkg": _max(next24, "cape"),
        "thunderstorm_hours_next_24h": int(thunder_hours),
        "forecast_hours_available": len(future),
    }
    return {
        "provider": "Open-Meteo",
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
        "latitude": _finite(payload.get("latitude")),
        "longitude": _finite(payload.get("longitude")),
        "timezone": payload.get("timezone") or "UTC",
        "summary": summary,
        "hourly": future[:72],
        "disclaimer": (
            "Weather is public forecast context for resilience research. It is not certified aviation weather, "
            "METAR/TAF, or an operational flight-release product."
        ),
    }


def fetch_weather(latitude: float, longitude: float, forecast_days: int = 3,
                  timeout: float = 25.0, get: Callable | None = None):
    """Fetch current + hourly public weather without an API key.

    ``get`` can be injected in tests. It must accept ``url, params, timeout``
    and return a requests-like response.
    """
    lat = float(latitude); lon = float(longitude)
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Invalid weather coordinates.")
    days = max(1, min(7, int(forecast_days)))
    if get is None:
        import requests
        get = requests.get
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ",".join(CURRENT_FIELDS),
        "hourly": ",".join(HOURLY_FIELDS),
        "forecast_days": days,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }
    try:
        response = get(OPEN_METEO_FORECAST, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise WeatherServiceError(f"Weather download failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, Mapping) or not payload.get("hourly"):
        raise WeatherServiceError("Weather provider returned no hourly forecast.")
    return summarize_weather(payload)

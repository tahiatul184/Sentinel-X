"""Multi-source EO discovery for the first AEROSENTINEL architecture block.

This module discovers source availability and retrieves weather/rainfall context.
It deliberately does not pretend that STAC discovery equals analysis-ready data:
Sentinel-1 calibration, optical QA/cloud masking, reprojection and co-registration
must be completed before learned multimodal inference.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import math
import requests

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def _day(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def bbox_around(latitude, longitude, radius_km):
    lat = float(latitude); lon = float(longitude); radius = float(radius_km)
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("Latitude/longitude are outside WGS84 bounds.")
    if not 0.1 <= radius <= 100:
        raise ValueError("Discovery radius must be 0.1-100 km.")
    dlat = radius / 111.32
    dlon = radius / max(111.32 * math.cos(math.radians(lat)), 5.0)
    return [lon-dlon, lat-dlat, lon+dlon, lat+dlat]


def _stac_search(session, collections, bbox, start, end, limit=20):
    payload = {
        "collections": list(collections), "bbox": bbox,
        "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
        "limit": int(limit),
    }
    response = session.post(EARTH_SEARCH, json=payload, timeout=30)
    response.raise_for_status()
    body = response.json()
    return body.get("features", [])


def _safe_search(session, alternatives, bbox, start, end, limit=20):
    last = None
    for collection in alternatives:
        try:
            items = _stac_search(session, [collection], bbox, start, end, limit)
            return collection, items
        except requests.RequestException as exc:
            last = exc
    if last:
        raise last
    return alternatives[0], []


def _item_summary(item):
    props = item.get("properties", {})
    return {
        "id": item.get("id"),
        "collection": item.get("collection"),
        "datetime": props.get("datetime") or props.get("start_datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "polarizations": props.get("sar:polarizations"),
        "assets": sorted(item.get("assets", {}).keys())[:30],
    }


def _weather(session, latitude, longitude, start, end):
    params = {
        "latitude": float(latitude), "longitude": float(longitude),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": "precipitation,rain", "timezone": "UTC",
    }
    response = session.get(OPEN_METEO_ARCHIVE, params=params, timeout=30)
    response.raise_for_status()
    body = response.json()
    hourly = body.get("hourly") or {}
    times = hourly.get("time") or []
    precipitation = hourly.get("precipitation") or []
    rain = hourly.get("rain") or []
    valid_precip = [float(v) for v in precipitation if v is not None]
    return {
        "provider": "Open-Meteo Historical Weather API",
        "rows": min(len(times), len(precipitation)),
        "precipitation_sum_mm": float(sum(valid_precip)) if valid_precip else None,
        "max_hourly_precipitation_mm": float(max(valid_precip)) if valid_precip else None,
        "sample": [
            {"timestamp": t, "precipitation_mm": p, "rain_mm": r}
            for t, p, r in list(zip(times, precipitation, rain))[:48]
        ],
    }


def discover_multisource(latitude, longitude, start_date, end_date,
                         radius_km=5.0, max_items=20, session=None):
    """Discover all five architecture source groups for an AOI/time range."""
    start = _day(start_date); end = _day(end_date)
    if end < start:
        raise ValueError("End date cannot be earlier than start date.")
    if (end-start).days > 3660:
        raise ValueError("Discovery range cannot exceed 10 years in one request.")
    bbox = bbox_around(latitude, longitude, radius_km)
    owned = session is None
    session = session or requests.Session()
    session.headers.update({"User-Agent": "AEROSENTINEL/2.3 research prototype"})
    try:
        s2_collection, s2 = _safe_search(session, ["sentinel-2-l2a"], bbox, start, end, max_items)
        s1_collection, s1 = _safe_search(session, ["sentinel-1", "sentinel-1-grd"], bbox, start, end, max_items)
        dem_collection, dem = _safe_search(session, ["cop-dem-glo-30"], bbox, start, end, max_items)
        history_end = start - timedelta(days=1)
        history_start = history_end - timedelta(days=365)
        _, historical = _safe_search(session, ["sentinel-2-l2a"], bbox, history_start, history_end, max_items)
        weather = _weather(session, latitude, longitude, start, end)
        groups = {
            "Sentinel-1 SAR": {"collection": s1_collection, "count": len(s1), "items": [_item_summary(v) for v in s1]},
            "Sentinel-2 Optical": {"collection": s2_collection, "count": len(s2), "items": [_item_summary(v) for v in s2]},
            "DEM": {"collection": dem_collection, "count": len(dem), "items": [_item_summary(v) for v in dem]},
            "Weather / Rainfall": {"collection": "Open-Meteo archive", "count": weather["rows"], "summary": weather},
            "Historical imagery": {"collection": "sentinel-2-l2a historical lookback", "count": len(historical), "items": [_item_summary(v) for v in historical]},
        }
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "aoi": {"latitude": float(latitude), "longitude": float(longitude), "radius_km": float(radius_km), "bbox": bbox},
            "period": {"start": start.isoformat(), "end": end.isoformat(), "historical_start": history_start.isoformat(), "historical_end": history_end.isoformat()},
            "source_groups": groups,
            "all_groups_discovered": all(v["count"] > 0 for v in groups.values()),
            "processing_note": "Discovery confirms availability only. Build analysis-ready, cloud/QA-reviewed and co-registered rasters before multimodal inference; calibrate Sentinel-1 to the representation expected by the selected model.",
        }
    finally:
        if owned:
            session.close()

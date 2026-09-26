"""Automatic Copernicus DEM context for AEROSENTINEL v0.5."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from multisatellite_pipeline import PC_STAC, build_analysis_grid, _read_to_grid, _write_geotiff

DEM_COLLECTION = "cop-dem-glo-30"


class TerrainServiceError(RuntimeError):
    pass


def _sign_items(features, collection, session):
    token_response = session.get(
        f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}", timeout=(10, 30))
    token_response.raise_for_status()
    token = str(token_response.json().get("token") or "").strip()
    if not token:
        raise TerrainServiceError("Planetary Computer returned no DEM access token.")
    for item in features:
        for asset in (item.get("assets") or {}).values():
            href = str(asset.get("href") or "")
            parsed = urlparse(href)
            if parsed.scheme == "https" and parsed.hostname and parsed.hostname.lower().endswith(".blob.core.windows.net"):
                asset["href"] = href + ("&" if "?" in href else "?") + token
    return features


def search_dem(bbox, session=None, limit=16):
    close = False
    if session is None:
        import requests
        session = requests.Session(); close = True
    try:
        response = session.post(PC_STAC + "/search", json={
            "collections": [DEM_COLLECTION], "bbox": list(map(float, bbox)), "limit": int(limit)
        }, timeout=(10, 30))
        response.raise_for_status()
        features = list((response.json() or {}).get("features") or [])
        if not features:
            raise TerrainServiceError("No Copernicus DEM tile intersects the selected area.")
        return _sign_items(features, DEM_COLLECTION, session)
    except Exception as exc:
        if isinstance(exc, TerrainServiceError): raise
        raise TerrainServiceError(f"DEM discovery failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if close: session.close()


def _data_asset(item):
    assets = item.get("assets") or {}
    if "data" in assets and assets["data"].get("href"):
        return assets["data"]
    for asset in assets.values():
        if asset.get("href") and "tiff" in str(asset.get("type") or "").lower():
            return asset
    raise TerrainServiceError(f"DEM item {item.get('id')} has no raster data asset.")


def fetch_terrain(bbox, output_dir, grid_pixels=512, searcher=search_dem):
    grid = build_analysis_grid(bbox, max_pixels=min(768, max(128, int(grid_pixels))))
    items = searcher(bbox)
    accum = np.zeros((grid["height"], grid["width"]), dtype="float64")
    count = np.zeros((grid["height"], grid["width"]), dtype="uint16")
    used = []
    for item in items:
        arr, valid = _read_to_grid(_data_asset(item), grid)
        good = valid & np.isfinite(arr)
        if good.any():
            accum[good] += arr[good]
            count[good] += 1
            used.append(str(item.get("id") or "unknown"))
    elevation = np.full(accum.shape, np.nan, dtype="float32")
    good = count > 0
    if not good.any():
        raise TerrainServiceError("DEM tiles were found but did not yield valid pixels for the selected area.")
    elevation[good] = (accum[good] / count[good]).astype("float32")
    fill = float(np.nanmedian(elevation))
    work = np.where(np.isfinite(elevation), elevation, fill)
    res = float(grid.get("resolution_m") or 30.0)
    gy, gx = np.gradient(work, res, res)
    slope = np.degrees(np.arctan(np.sqrt(gx*gx + gy*gy))).astype("float32")
    slope[~good] = np.nan
    values = elevation[good]
    slopes = slope[good]
    summary = {
        "mean_elevation_m": float(np.mean(values)),
        "p05_elevation_m": float(np.percentile(values, 5)),
        "p95_elevation_m": float(np.percentile(values, 95)),
        "relief_p05_p95_m": float(np.percentile(values, 95) - np.percentile(values, 5)),
        "mean_slope_deg": float(np.mean(slopes)),
        "p90_slope_deg": float(np.percentile(slopes, 90)),
        "valid_percent": float(good.mean()*100.0),
    }
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    tif = _write_geotiff(out/"copernicus_dem_context.tif", grid, [elevation, slope], ["surface_elevation_m", "slope_deg"])
    from PIL import Image
    ev = elevation.copy(); sv = slope.copy()
    elo,ehi=np.nanpercentile(ev,[2,98]); eunit=np.clip((ev-elo)/max(float(ehi-elo),1e-6),0,1)
    sunit=np.clip(sv/30.0,0,1)
    rgb=np.stack([np.nan_to_num(sunit),np.nan_to_num(eunit),np.nan_to_num(1-sunit)],axis=-1)
    preview=out/"copernicus_dem_preview.png";Image.fromarray((rgb*255).astype('uint8')).save(preview)
    return {
        "provider": "Microsoft Planetary Computer / Copernicus DEM GLO-30",
        "collection": DEM_COLLECTION,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "bbox": list(map(float,bbox)),
        "grid_crs": str(grid["crs"]),
        "grid_resolution_m": res,
        "tile_ids": used,
        "summary": summary,
        "raster_path": str(tif),
        "preview_path": str(preview),
        "interpretation": "Static terrain/surface context for resilience screening; Copernicus DEM is a DSM, not surveyed airfield elevation data.",
    }

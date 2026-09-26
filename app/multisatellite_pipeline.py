"""Concurrent multi-satellite acquisition and analysis for AEROSENTINEL.

The pipeline is intentionally explicit about two different meanings of
"simultaneous":

* computation is concurrent -- independent satellite workers overlap search,
  remote COG reads, preprocessing and per-scene analysis;
* acquisition times are whatever the satellites actually observed.  Fusion
  records the cross-sensor acquisition skew and penalizes temporal mismatch.

Primary public catalog: Microsoft Planetary Computer STAC. The REST search and
anonymous SAS-signing APIs are used directly through requests, so no extra STAC
client dependency is required by the desktop runtime.

Outputs are research screening layers, not calibrated hazard probabilities.
Sentinel-1 GRD is amplitude data; its low-backscatter layer is therefore a
relative proxy unless the source collection itself is radiometrically/terrain
corrected.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np

from spectral_indices import compute_optical_indices, screening_layers
from image_processing import enhance_rgb, lee_filter, gradient_texture
from quality_control import image_quality_metrics
from change_detection import composite_change, registration_aware_composite_change
from thermal_processing import thermal_diagnostics
from reliability_fusion import scene_reliability, weighted_fusion, counterfactual_source_contribution
from robustness_profile import image_corruption_profile

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass(frozen=True)
class SatelliteSpec:
    key: str
    label: str
    collections: tuple[str, ...]
    kind: str
    cloud_filter: bool


SATELLITES = {
    "sentinel-1": SatelliteSpec(
        "sentinel-1", "Sentinel-1 SAR", ("sentinel-1-grd",), "sar", False
    ),
    "sentinel-2": SatelliteSpec(
        "sentinel-2", "Sentinel-2 Optical", ("sentinel-2-l2a",), "optical", True
    ),
    "landsat-8-9": SatelliteSpec(
        "landsat-8-9", "Landsat 8/9 Optical + Thermal (TIRS)", ("landsat-c2-l2",), "optical", True
    ),
}

DEFAULT_SATELLITES = ("sentinel-1", "sentinel-2", "landsat-8-9")

OPTICAL_ALIASES = {
    "red": ("red", "B04", "SR_B4"),
    "green": ("green", "B03", "SR_B3"),
    "blue": ("blue", "B02", "SR_B2"),
    "nir": ("nir", "nir08", "B08", "SR_B5"),
    "swir16": ("swir16", "B11", "SR_B6"),
    "swir22": ("swir22", "B12", "SR_B7"),
    "thermal": ("lwir11", "lwir", "ST_B10", "st_b10", "thermal", "tir"),
    "thermal_qa": ("qa", "st_qa", "ST_QA"),
    "emissivity": ("emis", "st_emis", "ST_EMIS"),
    "emissivity_std": ("emsd", "st_emsd", "ST_EMSD"),
    "cloud_distance": ("cdist", "st_cdist", "ST_CDIST"),
    "atmos_transmittance": ("atran", "st_atran", "ST_ATRAN"),
    "qa_radsat": ("qa_radsat", "QA_RADSAT"),
    "scl": ("SCL", "scl"),
    # Do not alias generic "qa" here: in Planetary Computer Landsat items,
    # "qa" is ST_QA (surface-temperature uncertainty), not QA_PIXEL.
    "qa_pixel": ("qa_pixel", "QA_PIXEL"),
}


class MultisatelliteError(RuntimeError):
    pass


def _iso(value) -> str:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    else:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("Missing acquisition timestamp.")
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def item_time(item: Mapping) -> datetime:
    props = item.get("properties") or {}
    raw = props.get("datetime") or props.get("start_datetime") or props.get("end_datetime")
    return datetime.fromisoformat(_iso(raw))


def _collection_query(spec: SatelliteSpec, max_cloud: float):
    return {"eo:cloud_cover": {"lte": float(max_cloud)}} if spec.cloud_filter else None


def search_planetary_computer(spec: SatelliteSpec, bbox, start, end, max_items=1,
                              max_cloud=80.0):
    """Search Planetary Computer with requests and sign Azure assets.

    Planetary Computer documents anonymous SAS-token access for hosted datasets.
    Using the REST API directly keeps the desktop dependency footprint small.
    """
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": "AEROSENTINEL/2.3 concurrent-multisatellite"})
    query = _collection_query(spec, max_cloud)
    last = None
    try:
        for collection in spec.collections:
            try:
                payload = {
                    "collections": [collection], "bbox": list(map(float, bbox)),
                    "datetime": f"{str(start)[:10]}T00:00:00Z/{str(end)[:10]}T23:59:59Z",
                    "limit": max(1, int(max_items) * 6),
                    "sortby": [{"field": "datetime", "direction": "desc"}],
                }
                if query:
                    payload["query"] = query
                response = session.post(PC_STAC + "/search", json=payload, timeout=(10, 30))
                if getattr(response, "status_code", 200) == 400:
                    # Compatibility fallback if a STAC deployment disables sorting.
                    payload.pop("sortby", None)
                    response = session.post(PC_STAC + "/search", json=payload, timeout=(10, 30))
                response.raise_for_status()
                body = response.json()
                features = list(body.get("features") or [])
                features.sort(key=item_time, reverse=True)
                features = features[: int(max_items)]
                if not features:
                    return []
                token_response = session.get(
                    f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}",
                    timeout=(10, 30),
                )
                token_response.raise_for_status()
                token = str(token_response.json().get("token") or "").strip()
                if not token:
                    raise MultisatelliteError(f"Planetary Computer returned no SAS token for {collection}.")
                for item in features:
                    for asset in (item.get("assets") or {}).values():
                        href = str(asset.get("href") or "")
                        parsed = urlparse(href)
                        if parsed.scheme == "https" and parsed.hostname and parsed.hostname.lower().endswith(".blob.core.windows.net"):
                            asset["href"] = href + ("&" if "?" in href else "?") + token
                return features
            except Exception as exc:
                last = exc
        if last is not None:
            raise MultisatelliteError(f"{spec.label} search failed: {last}") from last
        return []
    finally:
        session.close()


def _asset_common_names(asset: Mapping) -> set[str]:
    names = set()
    for field in ("eo:bands", "raster:bands"):
        for band in asset.get(field) or []:
            for key in ("common_name", "name"):
                if band.get(key):
                    names.add(str(band[key]).lower())
    return names


def _find_asset(item: Mapping, aliases: Sequence[str], common_names: Sequence[str] = ()):
    assets = item.get("assets") or {}
    for key in aliases:
        if key in assets and assets[key].get("href"):
            return key, assets[key]
    lower = {str(k).lower(): (k, v) for k, v in assets.items()}
    for key in aliases:
        if str(key).lower() in lower and lower[str(key).lower()][1].get("href"):
            return lower[str(key).lower()]
    wanted = {str(v).lower() for v in common_names}
    if wanted:
        for key, asset in assets.items():
            if asset.get("href") and (_asset_common_names(asset) & wanted):
                return key, asset
    return None, None


def select_assets(spec: SatelliteSpec, item: Mapping):
    """Map provider-specific asset keys into AEROSENTINEL canonical bands."""
    if spec.kind == "optical":
        selected = {}
        for canonical, common in (("red", ("red",)), ("green", ("green",)),
                                  ("blue", ("blue",)), ("nir", ("nir", "nir08"))):
            key, asset = _find_asset(item, OPTICAL_ALIASES[canonical], common)
            if asset is None:
                raise MultisatelliteError(f"{spec.label} item {item.get('id')} has no {canonical} band.")
            selected[canonical] = (key, asset)
        for canonical, common in (("swir16", ("swir16",)), ("swir22", ("swir22",))):
            key, asset = _find_asset(item, OPTICAL_ALIASES[canonical], common)
            if asset is not None:
                selected[canonical] = (key, asset)
        # Landsat Collection 2 Level-2 exposes the surface-temperature product
        # as the LWIR11/ST_B10 asset. Keep it optional because some scenes may
        # lack the auxiliary data required for a Level-2 temperature product.
        if spec.key == "landsat-8-9":
            key, asset = _find_asset(item, OPTICAL_ALIASES["thermal"], ("lwir11", "lwir", "thermal", "tir"))
            if asset is not None:
                selected["thermal"] = (key, asset)
            # Landsat Level-2 thermal auxiliary layers are optional scene by
            # scene, but materially improve interpretation when present.
            for canonical in ("thermal_qa", "emissivity", "emissivity_std",
                              "cloud_distance", "atmos_transmittance", "qa_radsat"):
                key, asset = _find_asset(item, OPTICAL_ALIASES[canonical])
                if asset is not None:
                    selected[canonical] = (key, asset)
        # Sensor-specific QA, optional but strongly preferred.
        qa_name = "scl" if spec.key == "sentinel-2" else "qa_pixel"
        key, asset = _find_asset(item, OPTICAL_ALIASES[qa_name])
        if asset is not None:
            selected[qa_name] = (key, asset)
        return selected

    # Sentinel-1 GRD: use a co-pol channel and, when available, cross-pol.
    assets = item.get("assets") or {}
    for copol, cross in (("vv", "vh"), ("hh", "hv")):
        ckey, casset = _find_asset(item, (copol,), (copol,))
        if casset is not None:
            out = {"copol": (ckey, casset)}
            xkey, xasset = _find_asset(item, (cross,), (cross,))
            if xasset is not None:
                out["crosspol"] = (xkey, xasset)
            out["polarization"] = (copol, cross if xasset is not None else None)
            return out
    raise MultisatelliteError(f"{spec.label} item {item.get('id')} has no supported SAR polarization asset.")


def _validated_remote_href(href: str) -> str:
    parsed = urlparse(str(href))
    if parsed.scheme != "https" or not parsed.hostname:
        raise MultisatelliteError("Satellite asset is not a signed HTTPS URL.")
    host = parsed.hostname.lower()
    allowed = (
        host == "planetarycomputer.microsoft.com"
        or host.endswith(".blob.core.windows.net")
        or host.endswith(".azureedge.net")
    )
    if not allowed:
        raise MultisatelliteError(f"Unexpected satellite asset host: {host}")
    return href


def build_analysis_grid(bbox, max_pixels=512):
    """Build one local UTM grid shared by every satellite in a concurrent run."""
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds

    west, south, east, north = map(float, bbox)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("Invalid WGS84 AOI bounds.")
    lon = (west + east) / 2.0
    lat = (south + north) / 2.0
    zone = max(1, min(60, int((lon + 180) // 6) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    crs = CRS.from_epsg(epsg)
    mb = transform_bounds("EPSG:4326", crs, west, south, east, north, densify_pts=21)
    mw, ms, me, mn = mb
    span_x, span_y = max(me - mw, 1.0), max(mn - ms, 1.0)
    max_pixels = int(max(128, min(1024, max_pixels)))
    resolution = max(span_x, span_y) / max_pixels
    width = max(1, int(math.ceil(span_x / resolution)))
    height = max(1, int(math.ceil(span_y / resolution)))
    transform = from_bounds(mw, ms, me, mn, width, height)
    return {
        "crs": crs, "transform": transform, "width": width, "height": height,
        "bounds_wgs84": [west, south, east, north], "bounds_projected": list(mb),
        "resolution_m": float(resolution),
    }


def _scale_offset(asset: Mapping):
    bands = asset.get("raster:bands") or []
    band = bands[0] if bands else {}
    scale = band.get("scale", 1.0)
    offset = band.get("offset", 0.0)
    try:
        return float(scale), float(offset)
    except (TypeError, ValueError):
        return 1.0, 0.0


def _read_to_grid(asset: Mapping, grid, *, categorical=False):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    href = _validated_remote_href(asset["href"])
    resampling = Resampling.nearest if categorical else Resampling.bilinear
    env = dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        VSI_CACHE="TRUE", VSI_CACHE_SIZE=32_000_000,
    )
    with rasterio.Env(**env):
        with rasterio.open(href) as src:
            with WarpedVRT(
                src, crs=grid["crs"], transform=grid["transform"],
                width=grid["width"], height=grid["height"],
                resampling=resampling,
            ) as vrt:
                arr = vrt.read(1, masked=True).astype("float32")
    values = arr.filled(np.nan).astype("float32")
    if not categorical:
        scale, offset = _scale_offset(asset)
        values = values * scale + offset
    return values, ~np.ma.getmaskarray(arr) & np.isfinite(values)




def _read_landsat_surface_temperature(asset: Mapping, grid):
    """Read Landsat Collection-2 Level-2 surface temperature in Kelvin.

    Planetary Computer normally publishes STAC raster scale/offset metadata.
    If those metadata are absent and the values still look like raw uint16 DN,
    apply the documented USGS Collection-2 factor: DN * 0.00341802 + 149.0.
    """
    kelvin, valid = _read_to_grid(asset, grid)
    finite = kelvin[valid & np.isfinite(kelvin)]
    scale_note = "STAC raster scale/offset metadata"
    if finite.size and float(np.nanmedian(finite)) > 1000.0:
        kelvin = kelvin * 0.00341802 + 149.0
        scale_note = "USGS Collection-2 fallback scale 0.00341802 + 149.0"
    plausible = valid & np.isfinite(kelvin) & (kelvin >= 150.0) & (kelvin <= 400.0)
    kelvin = np.where(plausible, kelvin, np.nan).astype("float32")
    return kelvin, plausible, scale_note


def _box_mean(arr, valid=None, radius=1):
    """Small NaN-aware box mean used as a transparent SAR speckle-screening smoother."""
    data=np.asarray(arr,dtype="float32")
    if valid is None:
        valid=np.isfinite(data)
    valid=np.asarray(valid,dtype=bool)&np.isfinite(data)
    fill=np.where(valid,data,0.0)
    weight=valid.astype("float32")
    r=max(1,int(radius));k=2*r+1
    def integ(a):
        pad=np.pad(a,r,mode="constant")
        return np.pad(pad,((1,0),(1,0)),mode="constant").cumsum(0).cumsum(1)
    si=integ(fill);wi=integ(weight);h,w=data.shape
    sums=si[k:k+h,k:k+w]-si[:h,k:k+w]-si[k:k+h,:w]+si[:h,:w]
    counts=wi[k:k+h,k:k+w]-wi[:h,k:k+w]-wi[k:k+h,:w]+wi[:h,:w]
    out=np.full_like(data,np.nan,dtype="float32")
    ok=counts>0
    out[ok]=sums[ok]/counts[ok]
    return out

def _safe_divide(a, b):
    out = np.full_like(a, np.nan, dtype="float32")
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-9)
    out[valid] = a[valid] / b[valid]
    return out


def _percentile_unit(arr, valid):
    out = np.full_like(arr, np.nan, dtype="float32")
    values = arr[valid & np.isfinite(arr)]
    if values.size == 0:
        return out
    lo, hi = np.percentile(values, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out[valid] = 0.5
    else:
        out[valid] = np.clip((arr[valid] - lo) / (hi - lo), 0, 1)
    return out


def _cloud_mask(spec: SatelliteSpec, selected, grid):
    """Return True where QA permits quantitative optical screening."""
    shape = (grid["height"], grid["width"])
    if "scl" in selected:
        scl, valid = _read_to_grid(selected["scl"][1], grid, categorical=True)
        scl_i = np.where(np.isfinite(scl), np.rint(scl), -999).astype("int16")
        blocked = np.isin(scl_i, [0, 1, 3, 8, 9, 10, 11])
        return valid & ~blocked, "Sentinel-2 SCL"
    if "qa_pixel" in selected:
        qa, valid = _read_to_grid(selected["qa_pixel"][1], grid, categorical=True)
        qa_i = np.where(np.isfinite(qa), np.rint(qa), 1).astype("uint16")
        # Landsat Collection-2 QA_PIXEL: fill, dilated cloud, cirrus,
        # cloud, cloud shadow and snow are blocked for spectral indices.
        mask_bits = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)
        return valid & ((qa_i & mask_bits) == 0), "Landsat QA_PIXEL"
    return np.ones(shape, dtype=bool), "No pixel QA asset; provider scene cloud metadata only"


def _write_geotiff(path: Path, grid, bands: Sequence[np.ndarray], names: Sequence[str]):
    import rasterio
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff", width=grid["width"], height=grid["height"], count=len(bands),
        dtype="float32", crs=grid["crs"], transform=grid["transform"],
        nodata=np.nan, compress="deflate", tiled=True,
        blockxsize=256 if grid["width"] >= 256 else max(16, (grid["width"] // 16) * 16),
        blockysize=256 if grid["height"] >= 256 else max(16, (grid["height"] // 16) * 16),
    )
    # GTiff tile dimensions must be multiples of 16. Very small AOIs use strips.
    if grid["width"] < 16 or grid["height"] < 16:
        profile.pop("tiled", None); profile.pop("blockxsize", None); profile.pop("blockysize", None)
    with rasterio.open(path, "w", **profile) as dst:
        for idx, (band, name) in enumerate(zip(bands, names), start=1):
            dst.write(np.asarray(band, dtype="float32"), idx)
            dst.set_band_description(idx, str(name))
    return path


def _rgb_preview(path: Path, red, green, blue, valid):
    from PIL import Image
    chans = []
    for arr in (red, green, blue):
        unit = _percentile_unit(arr, valid)
        chans.append(np.nan_to_num(unit, nan=0.0))
    rgb = np.stack(chans, axis=-1)
    rgb = np.clip(np.power(rgb, 0.75) * 255, 0, 255).astype("uint8")
    Image.fromarray(rgb).save(path)


def _sar_preview(path: Path, copol, crosspol, valid):
    from PIL import Image
    c = np.nan_to_num(_percentile_unit(np.log1p(np.maximum(copol, 0)), valid), nan=0.0)
    if crosspol is not None:
        x = np.nan_to_num(_percentile_unit(np.log1p(np.maximum(crosspol, 0)), valid), nan=0.0)
        ratio = np.nan_to_num(_percentile_unit(_safe_divide(crosspol, np.maximum(copol, 1e-6)), valid), nan=0.0)
    else:
        x = c; ratio = 1.0 - c
    rgb = np.stack([c, x, ratio], axis=-1)
    from PIL import Image
    Image.fromarray(np.clip(rgb * 255, 0, 255).astype("uint8")).save(path)


def _summary(arr, valid):
    values = arr[valid & np.isfinite(arr)]
    if values.size == 0:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def process_item(spec: SatelliteSpec, item: Mapping, grid, output_dir: Path,
                 *, event: Callable[[dict], None] | None = None):
    """Stream one scene's AOI bands and analyze them on the common grid."""
    event = event or (lambda _e: None)
    started = time.time()
    item_id = str(item.get("id") or "unknown")
    acquired = _iso((item.get("properties") or {}).get("datetime")
                    or (item.get("properties") or {}).get("start_datetime"))
    safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in item_id)[:120]
    folder = Path(output_dir) / spec.key / safe_id
    folder.mkdir(parents=True, exist_ok=True)
    selected = select_assets(spec, item)
    event({"type": "scene_started", "satellite": spec.key, "label": spec.label,
           "scene": item_id, "acquired_at": acquired})

    if spec.kind == "optical":
        bands, valid_all = {}, None
        for canonical in ("red", "green", "blue", "nir", "swir16", "swir22"):
            if canonical not in selected:
                continue
            event({"type": "asset_read", "satellite": spec.key, "scene": item_id,
                   "asset": canonical})
            arr, band_valid = _read_to_grid(selected[canonical][1], grid)
            bands[canonical] = arr
            valid_all = band_valid if valid_all is None else (valid_all & band_valid)
        qa_valid, qa_source = _cloud_mask(spec, selected, grid)
        valid = (valid_all if valid_all is not None else qa_valid) & qa_valid
        red, green, blue, nir = (bands[k] for k in ("red", "green", "blue", "nir"))

        indices = compute_optical_indices(bands, valid)
        layers = screening_layers(indices)
        ndvi = indices["ndvi"]
        ndwi = indices["ndwi"]
        nbr = indices.get("nbr", np.full_like(ndvi, np.nan))
        ndbi = indices.get("ndbi", np.full_like(ndvi, np.nan))
        water = layers["water"]
        vegetation = layers["vegetation"]
        built_surface = layers["built_surface"]
        texture = gradient_texture(nir, valid)

        thermal_kelvin = thermal_celsius = heat_screening = None
        thermal_valid = None
        thermal_scale_note = None
        thermal_diag = None
        thermal_aux = {}
        if "thermal" in selected:
            event({"type": "asset_read", "satellite": spec.key, "scene": item_id,
                   "asset": "thermal"})
            thermal_kelvin, thermal_valid, thermal_scale_note = _read_landsat_surface_temperature(
                selected["thermal"][1], grid
            )
            thermal_valid = thermal_valid & qa_valid

            # QA_RADSAT bit 11 is Landsat 8/9 terrain occlusion. It is not a
            # TIRS saturation flag, but excluding terrain-occluded pixels avoids
            # interpreting temperatures where the desired ground is not visible.
            if "qa_radsat" in selected:
                event({"type": "asset_read", "satellite": spec.key, "scene": item_id,
                       "asset": "qa_radsat"})
                qars, qars_valid = _read_to_grid(selected["qa_radsat"][1], grid, categorical=True)
                qars_i = np.where(np.isfinite(qars), np.rint(qars), (1 << 11)).astype("uint16")
                thermal_valid &= qars_valid & ((qars_i & (1 << 11)) == 0)
                thermal_aux["terrain_visibility_source"] = "Landsat QA_RADSAT bit 11"

            for canonical in ("thermal_qa", "emissivity", "emissivity_std",
                              "cloud_distance", "atmos_transmittance"):
                if canonical not in selected:
                    continue
                event({"type": "asset_read", "satellite": spec.key, "scene": item_id,
                       "asset": canonical})
                arr, arr_valid = _read_to_grid(selected[canonical][1], grid)
                arr[~arr_valid] = np.nan
                thermal_aux[canonical] = arr

            thermal_kelvin[~thermal_valid] = np.nan
            thermal_diag = thermal_diagnostics(
                thermal_kelvin, thermal_valid,
                uncertainty_k=thermal_aux.get("thermal_qa"),
                emissivity=thermal_aux.get("emissivity"),
                emissivity_std=thermal_aux.get("emissivity_std"),
                cloud_distance_km=thermal_aux.get("cloud_distance"),
                atmospheric_transmittance=thermal_aux.get("atmos_transmittance"),
            )
            thermal_celsius = thermal_diag["surface_temp_c"]
            # Keep the legacy relative layer for fusion, but derive it from
            # positive robust/uncertainty-aware anomaly evidence rather than
            # raw scene temperature ranking alone.
            score_source = thermal_diag["uncertainty_normalized_anomaly"]
            if not np.any(np.isfinite(score_source[thermal_valid])):
                score_source = thermal_diag["robust_anomaly_z"]
            positive = np.where(np.isfinite(score_source), np.maximum(score_source, 0.0), np.nan)
            heat_screening = _percentile_unit(positive, thermal_valid)
            heat_screening[~thermal_valid] = np.nan

        stack_bands = [red, green, blue, nir]
        stack_names = [
            "red_reflectance", "green_reflectance", "blue_reflectance", "nir_reflectance"
        ]
        for band_name in ("swir16", "swir22"):
            if band_name in bands:
                stack_bands.append(bands[band_name]); stack_names.append(f"{band_name}_reflectance")
        for name in ("ndvi", "ndwi", "mndwi", "ndmi", "savi", "nbr", "nbr2", "ndbi"):
            if name in indices:
                stack_bands.append(indices[name]); stack_names.append(name)
        stack_bands.extend([water, vegetation, built_surface, texture])
        stack_names.extend([
            "relative_water_screening", "relative_vegetation_screening",
            "relative_built_surface_screening", "relative_gradient_texture_screening",
        ])
        if thermal_kelvin is not None:
            stack_bands.extend([thermal_kelvin, thermal_celsius, heat_screening])
            stack_names.extend([
                "landsat_surface_temperature_kelvin",
                "landsat_surface_temperature_celsius",
                "relative_heat_screening",
            ])
            aux_layers = (
                ("thermal_qa", "landsat_st_qa_uncertainty_kelvin"),
                ("emissivity", "landsat_surface_emissivity"),
                ("emissivity_std", "landsat_emissivity_standard_deviation"),
                ("cloud_distance", "landsat_distance_to_cloud_km"),
                ("atmos_transmittance", "landsat_atmospheric_transmittance"),
            )
            for key, label in aux_layers:
                if key in thermal_aux:
                    stack_bands.append(thermal_aux[key]); stack_names.append(label)
            diagnostic_layers = (
                ("temperature_anomaly_c", "thermal_temperature_anomaly_celsius"),
                ("robust_anomaly_z", "thermal_robust_anomaly_z"),
                ("uncertainty_normalized_anomaly", "thermal_uncertainty_normalized_anomaly"),
                ("inverse_variance_weight", "thermal_inverse_variance_weight"),
                ("elevated_uncertainty_flag", "thermal_elevated_uncertainty_flag"),
                ("near_cloud_flag", "thermal_near_cloud_flag"),
                ("thermal_anomaly_candidate", "thermal_anomaly_candidate_flag"),
            )
            for key, label in diagnostic_layers:
                arr = thermal_diag[key]
                if arr.dtype == bool:
                    arr = arr.astype("float32")
                stack_bands.append(arr); stack_names.append(label)
        stack_path = folder / "analysis_stack.tif"
        _write_geotiff(stack_path, grid, stack_bands, stack_names)
        preview = folder / "preview.png"
        _rgb_preview(preview, red, green, blue, valid)
        enhanced_preview = folder / "preview_enhanced.png"
        enhanced_rgb = enhance_rgb(red, green, blue, valid)
        from PIL import Image
        Image.fromarray(enhanced_rgb).save(enhanced_preview)
        props = item.get("properties") or {}
        quality = image_quality_metrics(
            enhanced_rgb, valid, cloud_cover_percent=props.get("eo:cloud_cover")
        )
        robustness = image_corruption_profile(
            enhanced_rgb, valid, cloud_cover_percent=props.get("eo:cloud_cover")
        )
        analysis = {
            "ndvi": _summary(ndvi, valid), "ndwi": _summary(ndwi, valid),
            "nbr": _summary(nbr, valid), "ndbi": _summary(ndbi, valid),
            "water_screening": _summary(water, valid),
            "vegetation_screening": _summary(vegetation, valid),
            "built_surface_screening": _summary(built_surface, valid),
            "gradient_texture_screening": _summary(texture, valid),
            "qa_source": qa_source,
            "processing_quality": quality,
            "robustness_stress": robustness,
            "processing_modules": [
                "robust percentile stretch", "CLAHE local contrast", "mild unsharp visualization",
                "gradient texture", "expanded spectral indices", "processing-quality screening",
                "uncertainty-aware Landsat thermal diagnostics",
            ],
            "enhancement_note": (
                "Visualization enhancement changes contrast/noise presentation only; it does not create "
                "new spatial detail or improve the satellite's native ground sampling distance."
            ),
        }
        for name in ("mndwi", "ndmi", "savi", "nbr2"):
            if name in indices:
                analysis[name] = _summary(indices[name], valid)
        arrays = {
            "water": water, "ndvi": ndvi, "built_surface": built_surface,
            "texture": texture, "valid": valid
        }
        if thermal_kelvin is not None:
            analysis.update({
                "surface_temperature_kelvin": _summary(thermal_kelvin, thermal_valid),
                "surface_temperature_celsius": _summary(thermal_celsius, thermal_valid),
                "relative_heat_screening": _summary(heat_screening, thermal_valid),
                "temperature_anomaly_celsius": _summary(thermal_diag["temperature_anomaly_c"], thermal_valid),
                "thermal_robust_anomaly_z": _summary(thermal_diag["robust_anomaly_z"], thermal_valid),
                "thermal_uncertainty_normalized_anomaly": _summary(thermal_diag["uncertainty_normalized_anomaly"], thermal_valid),
                "thermal_quality": thermal_diag["stats"],
                "thermal_auxiliary_assets": sorted(k for k in thermal_aux if k != "terrain_visibility_source"),
                "thermal_terrain_visibility_source": thermal_aux.get("terrain_visibility_source"),
                "thermal_scale": thermal_scale_note,
                "thermal_native_sampling_note": (
                    "Landsat 8/9 TIRS Band 10 native sampling is 100 m; the Level-2 product is distributed "
                    "on a 30 m grid. Do not interpret neighboring 30 m thermal pixels as independent 30 m measurements."
                ),
                "thermal_note": (
                    "USGS Landsat Collection-2 Level-2 land-surface temperature. ST_QA uncertainty, "
                    "emissivity, cloud-distance and atmospheric-transmittance diagnostics are retained when available. "
                    "AEROSENTINEL anomaly flags are research screening aids, not 2-m air temperature or a certified fire alarm."
                ),
            })
            if "thermal_qa" in thermal_aux:
                analysis["surface_temperature_uncertainty_kelvin"] = _summary(thermal_aux["thermal_qa"], thermal_valid)
            if "emissivity" in thermal_aux:
                analysis["surface_emissivity"] = _summary(thermal_aux["emissivity"], thermal_valid)
            if "cloud_distance" in thermal_aux:
                analysis["distance_to_cloud_km"] = _summary(thermal_aux["cloud_distance"], thermal_valid)
            arrays.update({
                "surface_temp_c": thermal_celsius,
                "heat": heat_screening,
                "thermal_valid": thermal_valid,
                "thermal_anomaly": thermal_diag["temperature_anomaly_c"],
                "thermal_anomaly_z": thermal_diag["uncertainty_normalized_anomaly"],
            })
    else:
        event({"type": "asset_read", "satellite": spec.key, "scene": item_id, "asset": "copol"})
        copol, valid = _read_to_grid(selected["copol"][1], grid)
        cross = None
        if "crosspol" in selected:
            event({"type": "asset_read", "satellite": spec.key, "scene": item_id, "asset": "crosspol"})
            cross, cross_valid = _read_to_grid(selected["crosspol"][1], grid)
            valid &= cross_valid
        log_c = np.log1p(np.maximum(copol, 0))
        log_c_filtered = lee_filter(log_c, valid, radius=2)
        filtered_copol = np.expm1(np.maximum(log_c_filtered, 0)).astype("float32")
        low_backscatter = 1.0 - _percentile_unit(log_c_filtered, valid)
        low_backscatter[~valid] = np.nan
        ratio = (_safe_divide(cross, np.maximum(copol, 1e-6)) if cross is not None
                 else np.full_like(copol, np.nan))
        sar_texture = gradient_texture(log_c_filtered, valid)
        stack_bands = [copol, log_c_filtered, low_backscatter, sar_texture]
        stack_names = [
            f"{selected['polarization'][0]}_amplitude", "lee_filtered_log_copol",
            "relative_low_backscatter_screening", "relative_sar_texture_screening",
        ]
        if cross is not None:
            log_x = np.log1p(np.maximum(cross, 0))
            log_x_filtered = lee_filter(log_x, valid, radius=2)
            filtered_cross = np.expm1(np.maximum(log_x_filtered, 0)).astype("float32")
            stack_bands.extend([cross, log_x_filtered, ratio])
            stack_names.extend([
                f"{selected['polarization'][1]}_amplitude",
                "lee_filtered_log_crosspol", "cross_to_copol_ratio",
            ])
        else:
            filtered_cross = None
            copol_unit = _percentile_unit(log_c, valid)
            stack_bands.append(copol_unit); stack_names.append("relative_copol_unit")
        stack_path = folder / "analysis_stack.tif"
        _write_geotiff(stack_path, grid, stack_bands, stack_names)
        preview = folder / "preview.png"
        _sar_preview(preview, copol, cross, valid)
        enhanced_preview = folder / "preview_enhanced.png"
        _sar_preview(enhanced_preview, filtered_copol, filtered_cross, valid)
        from PIL import Image
        enhanced_rgb = np.asarray(Image.open(enhanced_preview).convert("RGB"))
        quality = image_quality_metrics(enhanced_rgb, valid)
        robustness = image_corruption_profile(enhanced_rgb, valid)
        analysis = {
            "copol_amplitude": _summary(copol, valid),
            "crosspol_amplitude": _summary(cross, valid) if cross is not None else None,
            "low_backscatter_screening": _summary(low_backscatter, valid),
            "sar_texture_screening": _summary(sar_texture, valid),
            "polarizations": [v for v in selected["polarization"] if v],
            "radiometry_note": "Relative amplitude screening from GRD; not sigma0/gamma0 unless the upstream collection states otherwise.",
            "noise_filter": "Adaptive Lee-style filter on log-amplitude for research speckle-sensitivity reduction",
            "processing_quality": quality,
            "robustness_stress": robustness,
            "processing_modules": [
                "Lee-style SAR speckle reduction", "gradient texture",
                "robust percentile preview stretch", "processing-quality screening",
            ],
            "enhancement_note": (
                "SAR filtering reduces speckle sensitivity for screening but does not convert GRD amplitude "
                "into calibrated sigma0/gamma0 or improve native spatial resolution."
            ),
        }
        arrays = {
            "water": low_backscatter.astype("float32"),
            "texture": sar_texture, "valid": valid
        }

    props = item.get("properties") or {}
    result = {
        "satellite": spec.key, "label": spec.label, "kind": spec.kind,
        "scene_id": item_id, "collection": item.get("collection"),
        "platform": props.get("platform") or props.get("constellation") or spec.label,
        "acquired_at": acquired, "cloud_cover_percent": props.get("eo:cloud_cover"),
        "native_gsd_m": props.get("gsd") or (10.0 if spec.key in {"sentinel-1","sentinel-2"} else 30.0 if spec.key=="landsat-8-9" else None),
        "valid_percent": float(np.mean(valid) * 100.0),
        "stack_path": str(stack_path), "preview_path": str(preview),
        "enhanced_preview_path": str(enhanced_preview),
        "processing_quality": quality,
        "analysis": analysis, "asset_keys": {k: v[0] if isinstance(v, tuple) else v for k, v in selected.items()},
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.time() - started),
        "_arrays": arrays,
    }
    event({"type": "scene_finished", "satellite": spec.key, "label": spec.label,
           "scene": item_id, "elapsed_seconds": result["elapsed_seconds"]})
    return result


def _latest_by_satellite(results):
    latest = {}
    for result in results:
        key = result["satellite"]
        if key not in latest or result["acquired_at"] > latest[key]["acquired_at"]:
            latest[key] = result
    return latest


def _temporal_change(results):
    """Registration-aware temporal change with optional multi-date persistence.

    The latest scene is compared with up to three earlier scenes from the same
    satellite.  With only two scenes this reduces to a normal pairwise change
    screen.  With longer history the returned map is the temporal median of the
    registered pairwise screens, reducing one-off noise and pseudo-change.
    """
    by_sat = {}
    for result in results:
        by_sat.setdefault(result["satellite"], []).append(result)
    maps, summaries = [], {}
    for key, rows in by_sat.items():
        rows.sort(key=lambda r: r["acquired_at"], reverse=True)
        if len(rows) < 2:
            continue
        latest = rows[0]
        priors = rows[1:min(len(rows), 4)]
        pair_maps = []
        pair_summaries = []
        primary_legacy = None
        primary_valid = None

        for pidx, previous in enumerate(priors):
            composite, composite_summary = registration_aware_composite_change(
                latest["_arrays"], previous["_arrays"], max_shift_px=6.0
            )
            reg = composite_summary.get("registration") or {}
            prev_water = previous["_arrays"].get("water")
            prev_valid = previous["_arrays"].get("valid")
            if reg.get("accepted") and prev_water is not None:
                from registration_quality import warp_translation
                prev_water, _ = warp_translation(
                    prev_water, reg.get("shift_x_px", 0.0), reg.get("shift_y_px", 0.0)
                )
                prev_valid_float, prev_mask = warp_translation(
                    np.asarray(prev_valid, dtype="float32"), reg.get("shift_x_px", 0.0),
                    reg.get("shift_y_px", 0.0), categorical=True
                )
                prev_valid = prev_mask & (np.nan_to_num(prev_valid_float, nan=0.0) >= 0.5)
            a = latest["_arrays"].get("water")
            b = prev_water
            valid = (latest["_arrays"].get("valid") & np.asarray(prev_valid, dtype=bool)
                     & np.isfinite(a) & np.isfinite(b))
            legacy_change = np.full_like(a, np.nan, dtype="float32")
            if valid.any():
                legacy_change[valid] = np.clip(np.abs(a[valid] - b[valid]), 0, 1)
            if pidx == 0:
                primary_legacy = legacy_change
                primary_valid = valid
            if composite is not None:
                pair_maps.append(composite)
            elif valid.any():
                pair_maps.append(legacy_change)
            pair_summaries.append({
                "previous": previous["acquired_at"],
                "registration": reg,
                "registration_gate": composite_summary.get("registration_gate"),
                "geometrically_ambiguous": bool(composite_summary.get("geometrically_ambiguous")),
                "multi_signal": composite_summary,
            })

        if not pair_maps:
            continue
        hist_stack = np.stack(pair_maps)
        valid_count = np.isfinite(hist_stack).sum(axis=0)
        temporal_map = np.full(hist_stack.shape[1:], np.nan, dtype="float32")
        if np.any(valid_count > 0):
            # np.nanmedian can warn for all-NaN slices; operate pixelwise via a
            # masked array to keep the packaged test output clean.
            masked = np.ma.masked_invalid(hist_stack)
            temporal_map = np.ma.median(masked, axis=0).filled(np.nan).astype("float32")
        persistence = np.zeros(hist_stack.shape[1:], dtype="float32")
        high = np.where(np.isfinite(hist_stack), hist_stack >= 0.60, False)
        np.divide(high.sum(axis=0), valid_count, out=persistence, where=valid_count>0)
        persistence[valid_count == 0] = np.nan
        maps.append(temporal_map)

        primary = pair_summaries[0]
        finite_temporal = temporal_map[np.isfinite(temporal_map)]
        finite_persistence = persistence[np.isfinite(persistence)]
        summaries[key] = {
            "latest": latest["acquired_at"], "previous": priors[0]["acquired_at"],
            "mean_absolute_screening_change": (
                float(np.nanmean(primary_legacy)) if primary_valid is not None and primary_valid.any() else None
            ),
            "valid_percent": float(primary_valid.mean() * 100.0) if primary_valid is not None else 0.0,
            "multi_signal": primary["multi_signal"],
            "registration": primary["registration"],
            "registration_gate": primary["registration_gate"],
            "geometrically_ambiguous": primary["geometrically_ambiguous"],
            "history_comparisons": len(pair_maps),
            "registration_history": pair_summaries,
            "mean_temporal_median_change": float(np.mean(finite_temporal)) if finite_temporal.size else None,
            "mean_change_persistence": float(np.mean(finite_persistence)) if finite_persistence.size else None,
            "persistent_high_change_fraction": (
                float(np.mean((finite_temporal >= 0.60) & (finite_persistence >= 0.66)))
                if finite_temporal.size and finite_persistence.size else None
            ),
            "temporal_method": "latest versus up to three registered historical scenes; median change + persistence",
        }
    if maps:
        stack = np.stack(maps)
        valid_count = np.isfinite(stack).sum(axis=0)
        summed = np.nansum(stack, axis=0)
        fused = np.divide(summed, valid_count, out=np.full(stack.shape[1:], np.nan, dtype="float32"), where=valid_count>0).astype("float32")
    else:
        fused = None
    return fused, summaries

def fuse_results(results, selected_satellites, grid, output_dir: Path,
                 temporal_tolerance_hours=72.0):
    """Reliability-aware fusion of the latest available scene from each satellite.

    Scene quality, cloud burden and acquisition skew control how strongly a
    source can influence the fused surface.  This reduces the risk that a weak
    modality dominates merely because it is present.
    """
    if not results:
        raise MultisatelliteError("No satellite scene completed successfully; fusion cannot run.")
    latest = _latest_by_satellite(results)
    sources = [latest[k] for k in selected_satellites if k in latest]
    if not sources:
        raise MultisatelliteError("No requested satellite produced an analyzable scene.")

    times = [item_time({"properties": {"datetime": r["acquired_at"]}}) for r in sources]
    reference_time = max(times)
    tolerance = max(float(temporal_tolerance_hours), 1e-6)
    source_quality = {
        r["satellite"]: scene_reliability(
            r, reference_time=reference_time, temporal_tolerance_hours=tolerance
        ) for r in sources
    }
    weights = [source_quality[r["satellite"]]["weight"] for r in sources]
    signals, valid_masks = [], []
    for row in sources:
        arr = row["_arrays"]["water"]
        valid = row["_arrays"]["valid"] & np.isfinite(arr)
        signals.append(np.where(valid, arr, np.nan).astype("float32"))
        valid_masks.append(valid)

    consensus, disagreement, effective_n, fusion_quality = weighted_fusion(signals, weights)
    stack = np.stack(signals)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    expected = max(1, len(selected_satellites))
    coverage = (count / expected).astype("float32")

    skew_hours = (max(times) - min(times)).total_seconds() / 3600.0 if len(times) > 1 else 0.0
    alignment = float(math.exp(-skew_hours / tolerance))
    missing = 1.0 - coverage
    # Reliability-aware uncertainty: missing modalities, source disagreement,
    # acquisition skew and the quality of the contributing sources are distinct.
    uncertainty = np.clip(
        0.40 * missing
        + 0.25 * np.nan_to_num(disagreement, nan=1.0)
        + 0.15 * (1.0 - alignment)
        + 0.20 * (1.0 - np.clip(fusion_quality, 0, 1)),
        0, 1,
    ).astype("float32")
    uncertainty[count == 0] = 1.0

    def weighted_optional(array_key):
        maps, ws = [], []
        for row in sources:
            arr = row["_arrays"].get(array_key)
            if arr is None or not np.isfinite(arr).any():
                continue
            maps.append(arr); ws.append(source_quality[row["satellite"]]["weight"])
        if not maps:
            return np.full_like(consensus, np.nan), np.full_like(consensus, np.nan), 0
        mean, std, _eff, _q = weighted_fusion(maps, ws)
        return mean.astype("float32"), std.astype("float32"), len(maps)

    mean_ndvi, ndvi_disagreement, _ = weighted_optional("ndvi")
    mean_built_surface, _built_disagreement, _ = weighted_optional("built_surface")
    mean_surface_temp_c, _temp_disagreement, thermal_source_count = weighted_optional("surface_temp_c")
    mean_heat_screening, _heat_disagreement, _ = weighted_optional("heat")

    change, change_summary = _temporal_change(results)
    if change is None:
        change = np.full_like(consensus, np.nan)

    reg_confidences = []
    reg_uncertainties = []
    ambiguous_pairs = 0
    for row in change_summary.values():
        reg = row.get("registration") or {}
        if reg.get("confidence") is not None:
            reg_confidences.append(float(reg.get("confidence") or 0.0))
        if reg.get("uncertainty_px") is not None:
            reg_uncertainties.append(float(reg["uncertainty_px"]))
        if row.get("geometrically_ambiguous"):
            ambiguous_pairs += 1
    registration_summary = {
        "pairs_evaluated": len(change_summary),
        "mean_confidence": float(np.mean(reg_confidences)) if reg_confidences else None,
        "max_uncertainty_px": float(np.max(reg_uncertainties)) if reg_uncertainties else None,
        "ambiguous_pairs": int(ambiguous_pairs),
        "status": ("NO_TEMPORAL_PAIR" if not change_summary else
                   "REVIEW_GEOMETRY" if ambiguous_pairs else "ACCEPTABLE"),
    }

    contributions = counterfactual_source_contribution(
        signals, weights, [r["satellite"] for r in sources]
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion_path = output_dir / "multisatellite_fusion.tif"
    _write_geotiff(
        fusion_path, grid,
        [consensus, uncertainty, coverage, effective_n, fusion_quality,
         mean_ndvi, ndvi_disagreement, mean_built_surface,
         mean_surface_temp_c, mean_heat_screening, change],
        ["multisensor_wetness_screening", "fusion_uncertainty_screening",
         "source_coverage_fraction", "effective_source_count", "fusion_source_reliability",
         "mean_optical_ndvi", "optical_ndvi_disagreement", "mean_optical_built_surface_screening",
         "mean_land_surface_temperature_celsius", "mean_relative_heat_screening",
         "temporal_change_screening"],
    )

    from PIL import Image
    wet = np.nan_to_num(consensus, nan=0.0)
    unc = np.nan_to_num(uncertainty, nan=1.0)
    cov = np.nan_to_num(coverage, nan=0.0)
    preview_rgb = np.stack([unc, wet, cov], axis=-1)
    preview_path = output_dir / "multisatellite_fusion_preview.png"
    Image.fromarray(np.clip(preview_rgb * 255, 0, 255).astype("uint8")).save(preview_path)

    available = [r["satellite"] for r in sources]
    return {
        "fusion_path": str(fusion_path), "preview_path": str(preview_path),
        "available_satellites": available,
        "requested_satellites": list(selected_satellites),
        "source_count": len(available), "requested_source_count": len(selected_satellites),
        "acquisition_skew_hours": float(skew_hours),
        "temporal_tolerance_hours": float(temporal_tolerance_hours),
        "temporal_alignment_score": alignment,
        "mean_source_coverage": float(np.nanmean(coverage)),
        "mean_effective_source_count": float(np.nanmean(effective_n[count > 0])) if np.any(count > 0) else 0.0,
        "mean_fusion_source_reliability": float(np.nanmean(fusion_quality[count > 0])) if np.any(count > 0) else 0.0,
        "source_reliability_weights": source_quality,
        "counterfactual_source_contribution": contributions,
        "registration_quality": registration_summary,
        "mean_wetness_screening": float(np.nanmean(consensus)) if np.isfinite(consensus).any() else None,
        "mean_fusion_uncertainty": float(np.nanmean(uncertainty)),
        "mean_optical_ndvi": float(np.nanmean(mean_ndvi)) if np.isfinite(mean_ndvi).any() else None,
        "mean_optical_built_surface_screening": float(np.nanmean(mean_built_surface)) if np.isfinite(mean_built_surface).any() else None,
        "mean_land_surface_temperature_celsius": (
            float(np.nanmean(mean_surface_temp_c)) if np.isfinite(mean_surface_temp_c).any() else None
        ),
        "mean_relative_heat_screening": (
            float(np.nanmean(mean_heat_screening)) if np.isfinite(mean_heat_screening).any() else None
        ),
        "thermal_source_count": thermal_source_count,
        "temporal_change": change_summary,
        "interpretation": (
            "Reliability-aware cross-sensor fusion uses scene validity, processing readiness, optical cloud burden "
            "and acquisition-time alignment to control each source's influence. Residual registration is estimated "
            "before temporal change scoring when a suitable signal is available. Counterfactual source-ablation "
            "scores show how much each modality affects the fused screen. Landsat surface temperature remains land-"
            "surface temperature, not air temperature. All outputs are research screens, not calibrated event probabilities."
        ),
    }


class ParallelTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def enter(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def leave(self):
        with self.lock:
            self.active = max(0, self.active - 1)


def run_concurrent_multisatellite(*, bbox, start_date, end_date, output_dir,
                                  selected_satellites=DEFAULT_SATELLITES,
                                  max_scenes_per_satellite=1, max_cloud=80.0,
                                  workers=3, grid_pixels=512,
                                  temporal_tolerance_hours=72.0,
                                  searcher=search_planetary_computer,
                                  processor=process_item,
                                  event: Callable[[dict], None] | None = None,
                                  cancelled: Callable[[], bool] | None = None):
    """Search, stream/crop, preprocess and analyze multiple satellites concurrently.

    Each scene task performs remote COG reading plus per-sensor analysis. Tasks
    are submitted to one bounded ThreadPoolExecutor, so independent satellites
    overlap in time. Fusion runs after the available tasks finish.
    """
    event = event or (lambda _e: None)
    cancelled = cancelled or (lambda: False)
    selected = list(dict.fromkeys(selected_satellites))
    if len(selected) < 2:
        raise ValueError("Select at least two satellite sources for concurrent multi-satellite analysis.")
    unknown = [k for k in selected if k not in SATELLITES]
    if unknown:
        raise ValueError(f"Unknown satellite source(s): {', '.join(unknown)}")
    workers = int(max(2, min(8, workers)))
    max_scenes = int(max(1, min(3, max_scenes_per_satellite)))
    grid = build_analysis_grid(bbox, grid_pixels)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    errors = {}
    found = {}

    event({"type": "run_started", "satellites": selected, "workers": workers})
    # Search concurrently too; one slow catalog query does not block others.
    with ThreadPoolExecutor(max_workers=min(workers, len(selected)), thread_name_prefix="stac") as pool:
        futures = {
            pool.submit(searcher, SATELLITES[key], bbox, start_date, end_date,
                        max_scenes, max_cloud): key for key in selected
        }
        for future in as_completed(futures):
            key = futures[future]
            if cancelled():
                raise MultisatelliteError("Concurrent satellite run was cancelled.")
            try:
                found[key] = future.result()
                event({"type": "search_finished", "satellite": key, "count": len(found[key])})
            except Exception as exc:
                found[key] = []
                errors[key] = f"{type(exc).__name__}: {exc}"
                event({"type": "source_error", "satellite": key, "error": errors[key]})

    jobs = []
    for key in selected:
        for item in found.get(key, []):
            jobs.append((SATELLITES[key], item))
    if not jobs:
        raise MultisatelliteError("No selected satellite returned a usable scene in the requested period.")

    tracker = ParallelTracker()
    results = []

    def do_job(spec, item):
        if cancelled():
            raise MultisatelliteError("Concurrent satellite run was cancelled.")
        tracker.enter()
        try:
            return processor(spec, item, grid, root / "scenes", event=event)
        finally:
            tracker.leave()

    with ThreadPoolExecutor(max_workers=min(workers, len(jobs)), thread_name_prefix="satellite") as pool:
        futures = {pool.submit(do_job, spec, item): (spec, item) for spec, item in jobs}
        for future in as_completed(futures):
            spec, item = futures[future]
            if cancelled():
                raise MultisatelliteError("Concurrent satellite run was cancelled.")
            try:
                results.append(future.result())
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                errors[f"{spec.key}:{item.get('id')}"] = message
                event({"type": "source_error", "satellite": spec.key,
                       "scene": item.get("id"), "error": message})

    if not results:
        raise MultisatelliteError("All satellite processing workers failed. See per-source errors.")
    fusion = fuse_results(results, selected, grid, root / "fusion", temporal_tolerance_hours)
    serializable_results = []
    for row in sorted(results, key=lambda r: (r["satellite"], r["acquired_at"]), reverse=False):
        clean = {k: v for k, v in row.items() if k != "_arrays"}
        serializable_results.append(clean)
    completed = time.time()
    run = {
        "version": "AEROSENTINEL multi-satellite concurrent pipeline 0.6",
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished_at": datetime.fromtimestamp(completed, timezone.utc).isoformat(),
        "elapsed_seconds": float(completed - started),
        "concurrency": {
            "configured_workers": workers, "peak_scene_workers": tracker.peak,
            "searches_concurrent": True, "scene_processing_concurrent": True,
            "pipeline_semantics": "download/remote-read + preprocess + analyze per satellite overlap; fusion follows completed workers",
        },
        "aoi": {"bbox": list(map(float, bbox)), "grid_crs": str(grid["crs"]),
                "grid_width": grid["width"], "grid_height": grid["height"],
                "grid_resolution_m": grid["resolution_m"]},
        "period": {"start": str(start_date)[:10], "end": str(end_date)[:10]},
        "requested_satellites": selected,
        "search_counts": {k: len(found.get(k, [])) for k in selected},
        "scene_results": serializable_results,
        "fusion": fusion,
        "errors": errors,
        "simultaneous_computation": tracker.peak >= 2,
        "simultaneous_acquisition_claimed": False,
    }
    event({"type": "run_finished", "scene_count": len(results),
           "peak_workers": tracker.peak, "fusion": fusion})
    return run


def write_run_json(run, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, allow_nan=False), encoding="utf-8")
    return path

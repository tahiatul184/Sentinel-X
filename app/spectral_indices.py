"""Robust multispectral indices for AEROSENTINEL.

All outputs are float32 arrays with NaN outside the supplied valid mask.  The
module deliberately exposes physically interpretable indices rather than
inventing spatial detail or class labels.
"""
from __future__ import annotations

from typing import Mapping
import numpy as np


def normalized_difference(a, b, valid=None, eps: float = 1e-9):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    if a.shape != b.shape:
        raise ValueError("Normalized-difference inputs must have the same shape.")
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a + b) > float(eps))
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    out = np.full(a.shape, np.nan, dtype="float32")
    out[ok] = (a[ok] - b[ok]) / (a[ok] + b[ok])
    return np.clip(out, -1.0, 1.0).astype("float32")


def _mask(arr, valid):
    out = np.asarray(arr, dtype="float32").copy()
    out[~np.asarray(valid, dtype=bool)] = np.nan
    return out


def compute_optical_indices(bands: Mapping[str, np.ndarray], valid):
    """Compute an index suite from whichever canonical bands are available.

    Required: red, green, nir. Optional: swir16 (preferred for moisture/built
    indices) and swir22 (burn severity / dry-surface contrast).
    """
    missing = [k for k in ("red", "green", "nir") if k not in bands]
    if missing:
        raise ValueError("Missing optical bands: " + ", ".join(missing))
    valid = np.asarray(valid, dtype=bool)
    red = bands["red"]
    green = bands["green"]
    nir = bands["nir"]
    out = {
        "ndvi": normalized_difference(nir, red, valid),
        "ndwi": normalized_difference(green, nir, valid),
    }

    denom = np.asarray(nir, dtype="float32") + np.asarray(red, dtype="float32") + 0.5
    savi = np.full_like(np.asarray(nir, dtype="float32"), np.nan, dtype="float32")
    ok = valid & np.isfinite(nir) & np.isfinite(red) & (np.abs(denom) > 1e-9)
    savi[ok] = 1.5 * (np.asarray(nir)[ok] - np.asarray(red)[ok]) / denom[ok]
    out["savi"] = np.clip(savi, -1.5, 1.5).astype("float32")

    swir16 = bands.get("swir16")
    swir22 = bands.get("swir22")
    if swir16 is not None:
        out["ndmi"] = normalized_difference(nir, swir16, valid)
        out["mndwi"] = normalized_difference(green, swir16, valid)
        out["ndbi"] = normalized_difference(swir16, nir, valid)
    elif swir22 is not None:
        # Compatibility fallback. SWIR1 is preferred for NDBI; retain a
        # transparent SWIR2-based proxy if SWIR1 is unavailable.
        out["ndbi"] = normalized_difference(swir22, nir, valid)

    if swir22 is not None:
        out["nbr"] = normalized_difference(nir, swir22, valid)
    if swir16 is not None and swir22 is not None:
        out["nbr2"] = normalized_difference(swir16, swir22, valid)

    for key, value in list(out.items()):
        out[key] = _mask(value, valid)
    return out


def screening_layers(indices: Mapping[str, np.ndarray]):
    """Map signed indices to transparent 0..1 screening layers."""
    ndvi = np.asarray(indices["ndvi"], dtype="float32")
    water_index = np.asarray(indices.get("mndwi", indices["ndwi"]), dtype="float32")
    ndbi = np.asarray(indices.get("ndbi", np.full_like(ndvi, np.nan)), dtype="float32")
    return {
        "water": np.clip((water_index + 1.0) / 2.0, 0, 1).astype("float32"),
        "vegetation": np.clip((ndvi + 1.0) / 2.0, 0, 1).astype("float32"),
        "built_surface": np.clip((ndbi + 1.0) / 2.0, 0, 1).astype("float32"),
    }

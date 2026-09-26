"""Data/image quality screening metrics for AEROSENTINEL.

Scores describe processing readiness, not sensor accuracy, hazard probability,
or fitness for safety-critical decisions.
"""
from __future__ import annotations

import math
import numpy as np


def image_quality_metrics(rgb_uint8, valid=None, cloud_cover_percent=None):
    import cv2

    rgb = np.asarray(rgb_uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Quality metrics require an RGB image.")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype("uint8")
    h, w = rgb.shape[:2]
    ok = np.ones((h, w), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if ok.shape != (h, w):
        raise ValueError("valid mask shape does not match image.")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    values = gray[ok]
    valid_fraction = float(ok.mean()) if ok.size else 0.0
    if not values.size:
        return {
            "valid_fraction": valid_fraction, "dynamic_range_fraction": 0.0,
            "entropy_bits": 0.0, "laplacian_variance": 0.0,
            "clipped_fraction": 1.0, "processing_readiness_score": 0.0,
            "cloud_cover_percent": cloud_cover_percent,
        }
    p02, p98 = np.percentile(values, [2, 98])
    dynamic = float(max(0.0, p98 - p02) / 255.0)
    hist = np.bincount(values.astype("uint8"), minlength=256).astype("float64")
    prob = hist / max(1.0, hist.sum())
    nz = prob > 0
    entropy = float(-(prob[nz] * np.log2(prob[nz])).sum())
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_var = float(np.var(lap[ok])) if ok.any() else 0.0
    clipped = float(np.mean((values <= 2) | (values >= 253)))
    texture = 1.0 - math.exp(-max(0.0, lap_var) / 250.0)
    entropy_unit = min(1.0, entropy / 8.0)
    cloud_factor = 1.0
    if cloud_cover_percent is not None:
        try:
            cloud_factor = np.clip(1.0 - float(cloud_cover_percent) / 100.0, 0.0, 1.0)
        except (TypeError, ValueError):
            cloud_factor = 1.0
    score = (0.40 * valid_fraction + 0.18 * min(1.0, dynamic * 2.0)
             + 0.14 * entropy_unit + 0.13 * texture
             + 0.10 * (1.0 - clipped) + 0.05 * cloud_factor)
    return {
        "valid_fraction": valid_fraction,
        "dynamic_range_fraction": dynamic,
        "entropy_bits": entropy,
        "laplacian_variance": lap_var,
        "clipped_fraction": clipped,
        "cloud_cover_percent": None if cloud_cover_percent is None else float(cloud_cover_percent),
        "processing_readiness_score": float(np.clip(score, 0.0, 1.0)),
        "score_note": "Processing/readability screen only; not sensor accuracy or operational reliability.",
    }


def array_validity_metrics(arrays):
    """Summarize finite-data coverage across a mapping of numeric arrays."""
    rows = {}
    for name, arr in arrays.items():
        data = np.asarray(arr)
        if not np.issubdtype(data.dtype, np.number):
            continue
        finite = np.isfinite(data)
        values = data[finite]
        rows[name] = {
            "finite_fraction": float(finite.mean()) if finite.size else 0.0,
            "min": float(np.min(values)) if values.size else None,
            "max": float(np.max(values)) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
        }
    return rows

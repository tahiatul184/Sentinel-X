"""Pixel-domain enhancement helpers for AEROSENTINEL.

These functions improve visualization robustness and reduce noise sensitivity.
They do not claim super-resolution and never create new sensor measurements.
"""
from __future__ import annotations

import numpy as np


def robust_unit(arr, valid=None, low: float = 2.0, high: float = 98.0):
    arr = np.asarray(arr, dtype="float32")
    ok = np.isfinite(arr)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    out = np.full(arr.shape, np.nan, dtype="float32")
    values = arr[ok]
    if not values.size:
        return out
    lo, hi = np.percentile(values, [float(low), float(high)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out[ok] = 0.5
    else:
        out[ok] = np.clip((arr[ok] - lo) / (hi - lo), 0.0, 1.0)
    return out


def enhance_rgb(red, green, blue, valid=None, clip_limit: float = 2.0):
    """Return an 8-bit RGB visualization with robust stretch + CLAHE + mild sharpening."""
    import cv2

    arrays = [np.asarray(v, dtype="float32") for v in (red, green, blue)]
    if arrays[0].shape != arrays[1].shape or arrays[0].shape != arrays[2].shape:
        raise ValueError("RGB channels must have the same shape.")
    if valid is None:
        valid = np.isfinite(arrays[0]) & np.isfinite(arrays[1]) & np.isfinite(arrays[2])
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(arrays[0]) & np.isfinite(arrays[1]) & np.isfinite(arrays[2])

    rgb = np.stack([np.nan_to_num(robust_unit(v, valid), nan=0.0) for v in arrays], axis=-1)
    # Mild gamma lift is visualization-only and applied after quantitative indices.
    rgb8 = np.clip(np.power(rgb, 0.82) * 255.0, 0, 255).astype("uint8")
    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=max(0.5, float(clip_limit)), tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.8)
    enhanced = cv2.addWeighted(enhanced, 1.18, blur, -0.18, 0)
    enhanced[~valid] = 0
    return enhanced.astype("uint8")


def lee_filter(arr, valid=None, radius: int = 2, noise_quantile: float = 35.0):
    """Adaptive Lee-style speckle reducer for a single 2-D SAR screening band."""
    import cv2

    data = np.asarray(arr, dtype="float32")
    if data.ndim != 2:
        raise ValueError("Lee filter expects a 2-D array.")
    ok = np.isfinite(data) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(data))
    if not ok.any():
        return np.full_like(data, np.nan, dtype="float32")
    fill_value = float(np.nanmedian(data[ok]))
    filled = np.where(ok, data, fill_value).astype("float32")
    k = max(3, 2 * int(radius) + 1)
    mean = cv2.boxFilter(filled, cv2.CV_32F, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_sq = cv2.boxFilter(filled * filled, cv2.CV_32F, (k, k), normalize=True, borderType=cv2.BORDER_REFLECT)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    local_var = var[ok]
    noise_var = float(np.percentile(local_var, np.clip(noise_quantile, 1.0, 99.0))) if local_var.size else 0.0
    weight = np.maximum(var - noise_var, 0.0) / np.maximum(var, 1e-9)
    filtered = mean + weight * (filled - mean)
    filtered = filtered.astype("float32")
    filtered[~ok] = np.nan
    return filtered


def gradient_texture(arr, valid=None):
    """Return a robust 0..1 gradient-magnitude texture screen."""
    data = np.asarray(arr, dtype="float32")
    ok = np.isfinite(data) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(data))
    if not ok.any():
        return np.full_like(data, np.nan, dtype="float32")
    fill = np.where(ok, data, float(np.nanmedian(data[ok])))
    gy, gx = np.gradient(fill)
    mag = np.sqrt(gx * gx + gy * gy).astype("float32")
    out = robust_unit(mag, ok, 2, 98)
    out[~ok] = np.nan
    return out


def denoise_rgb(rgb_uint8, strength: int = 6):
    """Conservative non-local means denoising for uploaded/local RGB imagery."""
    import cv2

    arr = np.asarray(rgb_uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("RGB image must have shape HxWx3.")
    arr = np.clip(arr, 0, 255).astype("uint8")
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    den = cv2.fastNlMeansDenoisingColored(bgr, None, max(1, int(strength)), max(1, int(strength)), 7, 21)
    return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)

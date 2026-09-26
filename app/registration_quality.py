"""Residual image-registration quality assessment for AEROSENTINEL.

The common projected grid removes most geometric mismatch, but resampling and
orthorectification do not guarantee sub-pixel agreement.  This module estimates
an optional residual translation and exposes confidence/uncertainty so temporal
change can abstain when geometry is ambiguous.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def _finite_pair(a, b, valid=None):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Registration inputs must be same-shape 2-D arrays.")
    ok = np.isfinite(a) & np.isfinite(b)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    return a, b, ok


def estimate_translation(reference, moving, valid=None, max_shift_px=6.0) -> dict:
    """Estimate the translation needed to align ``moving`` to ``reference``.

    OpenCV phase correlation is used when available because it provides a
    sub-pixel translation estimate and a response score.  This is a residual
    translation screen, not a replacement for rigorous sensor orthorectification.
    """
    ref, mov, ok = _finite_pair(reference, moving, valid)
    valid_fraction = float(ok.mean()) if ok.size else 0.0
    if ok.sum() < max(64, int(ok.size * 0.02)):
        return {
            "status": "INSUFFICIENT_OVERLAP", "method": "phase_correlation",
            "shift_x_px": 0.0, "shift_y_px": 0.0, "magnitude_px": 0.0,
            "response": 0.0, "confidence": 0.0, "uncertainty_px": None,
            "valid_overlap_fraction": valid_fraction, "accepted": False,
        }

    ref_fill = float(np.nanmedian(ref[ok])); mov_fill = float(np.nanmedian(mov[ok]))
    r = np.where(ok, ref, ref_fill).astype("float32")
    m = np.where(ok, mov, mov_fill).astype("float32")
    r -= float(np.mean(r[ok])); m -= float(np.mean(m[ok]))
    # Avoid unstable registration on nearly textureless rasters.
    texture = min(float(np.std(r[ok])), float(np.std(m[ok])))
    if texture < 1e-5:
        return {
            "status": "LOW_TEXTURE", "method": "phase_correlation",
            "shift_x_px": 0.0, "shift_y_px": 0.0, "magnitude_px": 0.0,
            "response": 0.0, "confidence": 0.0, "uncertainty_px": None,
            "valid_overlap_fraction": valid_fraction, "accepted": False,
        }

    try:
        import cv2
        window = cv2.createHanningWindow((r.shape[1], r.shape[0]), cv2.CV_32F)
        # phaseCorrelate(src1, src2) returns the shift that maps src1 -> src2.
        (sx, sy), response = cv2.phaseCorrelate(m, r, window)
        method = "opencv_phase_correlation"
    except Exception:
        # Integer-pixel fallback using FFT cross-correlation.
        fa = np.fft.fft2(m); fb = np.fft.fft2(r)
        cross = fa * np.conj(fb)
        cross /= np.maximum(np.abs(cross), 1e-12)
        corr = np.abs(np.fft.ifft2(cross))
        y, x = np.unravel_index(int(np.argmax(corr)), corr.shape)
        if x > corr.shape[1] // 2: x -= corr.shape[1]
        if y > corr.shape[0] // 2: y -= corr.shape[0]
        sx, sy = float(-x), float(-y)
        peak = float(np.max(corr)); response = peak / max(float(np.mean(corr)) * 8.0, 1e-9)
        response = float(np.clip(response, 0.0, 1.0))
        method = "numpy_fft_phase_correlation"

    sx = float(sx); sy = float(sy); response = float(np.clip(response, 0.0, 1.0))
    magnitude = float(math.hypot(sx, sy))
    within_limit = magnitude <= float(max_shift_px)
    confidence = float(np.clip(response * min(1.0, valid_fraction / 0.35), 0.0, 1.0))
    accepted = bool(within_limit and confidence >= 0.15)
    # A conservative screening uncertainty; not a calibrated geodetic covariance.
    uncertainty = float(np.clip(0.20 + 1.80 * (1.0 - confidence), 0.20, 2.0))
    return {
        "status": "ACCEPTED" if accepted else ("SHIFT_EXCEEDS_LIMIT" if not within_limit else "LOW_CONFIDENCE"),
        "method": method, "shift_x_px": sx, "shift_y_px": sy,
        "magnitude_px": magnitude, "response": response, "confidence": confidence,
        "uncertainty_px": uncertainty, "valid_overlap_fraction": valid_fraction,
        "accepted": accepted, "max_shift_px": float(max_shift_px),
        "note": "Residual translation screen after map-grid reprojection; not a full sensor-geometry solution.",
    }


def warp_translation(arr, shift_x_px: float, shift_y_px: float, *, categorical=False):
    """Shift a 2-D array by a fractional translation, returning NaN at new edges."""
    data = np.asarray(arr)
    if data.ndim != 2:
        raise ValueError("warp_translation expects a 2-D array.")
    try:
        import cv2
        src = data.astype("float32")
        valid = np.isfinite(src).astype("uint8")
        fill = float(np.nanmedian(src[np.isfinite(src)])) if np.isfinite(src).any() else 0.0
        src = np.where(np.isfinite(src), src, fill).astype("float32")
        matrix = np.array([[1.0, 0.0, float(shift_x_px)], [0.0, 1.0, float(shift_y_px)]], dtype="float32")
        interp = cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR
        out = cv2.warpAffine(src, matrix, (src.shape[1], src.shape[0]), flags=interp,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        v = cv2.warpAffine(valid, matrix, (src.shape[1], src.shape[0]), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        out = out.astype("float32"); out[~v] = np.nan
        return out, v
    except Exception:
        # Safe integer fallback.
        ix = int(round(float(shift_x_px))); iy = int(round(float(shift_y_px)))
        out = np.roll(np.roll(data.astype("float32"), iy, axis=0), ix, axis=1)
        v = np.isfinite(out)
        if iy > 0: v[:iy, :] = False
        elif iy < 0: v[iy:, :] = False
        if ix > 0: v[:, :ix] = False
        elif ix < 0: v[:, ix:] = False
        out[~v] = np.nan
        return out, v


def registration_quality_gate(registration: Mapping | None, min_confidence=0.35, max_uncertainty_px=1.5) -> dict:
    reg = dict(registration or {})
    confidence = float(reg.get("confidence") or 0.0)
    uncertainty = reg.get("uncertainty_px")
    accepted = bool(reg.get("accepted")) and confidence >= float(min_confidence)
    if uncertainty is not None:
        accepted = accepted and float(uncertainty) <= float(max_uncertainty_px)
    return {
        "pass": accepted,
        "confidence": confidence,
        "uncertainty_px": uncertainty,
        "reason": "registration quality acceptable" if accepted else "registration uncertainty too high for confident change attribution",
    }

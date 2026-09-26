"""Transparent multi-signal temporal change screening for AEROSENTINEL."""
from __future__ import annotations

from typing import Mapping, Sequence
import numpy as np

from registration_quality import estimate_translation, warp_translation, registration_quality_gate

DEFAULT_SIGNALS = ("water", "ndvi", "built_surface", "heat")


def _signal_change(a, b, valid):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    ok = np.asarray(valid, dtype=bool) & np.isfinite(a) & np.isfinite(b)
    raw = np.full_like(a, np.nan, dtype="float32")
    score = np.full_like(a, np.nan, dtype="float32")
    if not ok.any():
        return raw, score, None
    raw[ok] = np.abs(a[ok] - b[ok])
    vals = raw[ok]
    median = float(np.median(vals))
    q25, q75 = np.percentile(vals, [25, 75])
    robust_scale = max(float(q75 - q25), median, float(np.percentile(vals, 75)) * 0.25, 1e-6)
    score[ok] = 1.0 - np.exp(-raw[ok] / robust_scale)
    summary = {
        "mean_absolute_change": float(np.mean(vals)),
        "median_absolute_change": median,
        "p95_absolute_change": float(np.percentile(vals, 95)),
        "robust_scale": robust_scale,
        "high_change_screening_fraction": float(np.mean(score[ok] >= 0.75)),
        "valid_percent": float(ok.mean() * 100.0),
    }
    return raw, score.astype("float32"), summary


def composite_change(latest: Mapping[str, np.ndarray], previous: Mapping[str, np.ndarray],
                     signals: Sequence[str] = DEFAULT_SIGNALS):
    """Combine common change signals without assigning a semantic event label."""
    base_valid = np.asarray(latest.get("valid"), dtype=bool) & np.asarray(previous.get("valid"), dtype=bool)
    maps = []
    summaries = {}
    for name in signals:
        if latest.get(name) is None or previous.get(name) is None:
            continue
        _raw, score, summary = _signal_change(latest[name], previous[name], base_valid)
        if summary is not None:
            maps.append(score)
            summaries[name] = summary
    if not maps:
        return None, {"signals": {}, "available_signal_count": 0}
    stack = np.stack(maps)
    valid_count = np.isfinite(stack).sum(axis=0)
    summed = np.nansum(stack, axis=0)
    fused = np.divide(summed, valid_count, out=np.full(stack.shape[1:], np.nan, dtype="float32"), where=valid_count>0).astype("float32")
    finite = fused[np.isfinite(fused)]
    overall = {
        "signals": summaries,
        "available_signal_count": len(summaries),
        "mean_composite_change_screening": float(np.mean(finite)) if finite.size else None,
        "p95_composite_change_screening": float(np.percentile(finite, 95)) if finite.size else None,
        "high_change_screening_fraction": float(np.mean(finite >= 0.75)) if finite.size else None,
        "note": "Relative multi-signal change screen; not automatic event identification.",
    }
    return fused, overall


def _registration_anchor(latest, previous, signals):
    # Prefer continuous, structurally rich signals before threshold-like screens.
    priority = ("ndvi", "built_surface", "water", "heat")
    for name in priority:
        if name in signals and latest.get(name) is not None and previous.get(name) is not None:
            a=np.asarray(latest[name]); b=np.asarray(previous[name])
            if np.isfinite(a).sum() >= 64 and np.isfinite(b).sum() >= 64:
                return name
    return None


def registration_aware_composite_change(latest: Mapping[str, np.ndarray], previous: Mapping[str, np.ndarray],
                                        signals: Sequence[str] = DEFAULT_SIGNALS,
                                        max_shift_px: float = 6.0):
    """Estimate residual translation, align the previous observation, then score change.

    If registration confidence is insufficient, change is still computed on the
    map grid but the summary is explicitly marked geometrically ambiguous. This
    prevents a small geometric offset from being silently interpreted as a real
    semantic change.
    """
    anchor = _registration_anchor(latest, previous, signals)
    base_valid = np.asarray(latest.get("valid"), dtype=bool) & np.asarray(previous.get("valid"), dtype=bool)
    if anchor is None:
        fused, summary = composite_change(latest, previous, signals)
        summary["registration"] = {
            "status":"NO_SUITABLE_ANCHOR","confidence":0.0,"accepted":False,
            "note":"No continuous common signal was available for residual registration assessment."
        }
        summary["registration_gate"] = registration_quality_gate(summary["registration"])
        summary["geometrically_ambiguous"] = True
        return fused, summary

    reg = estimate_translation(latest[anchor], previous[anchor], valid=base_valid, max_shift_px=max_shift_px)
    gate = registration_quality_gate(reg)
    aligned = dict(previous)
    aligned_valid = np.asarray(previous.get("valid"), dtype=bool)
    if reg.get("accepted"):
        warped_valid = None
        for name in set(signals) | {"valid"}:
            if name == "valid" or previous.get(name) is None:
                continue
            warped, v = warp_translation(previous[name], reg["shift_x_px"], reg["shift_y_px"])
            aligned[name] = warped
            warped_valid = v if warped_valid is None else (warped_valid & v)
        if warped_valid is not None:
            aligned_valid = aligned_valid & warped_valid
        # Also warp the source validity mask so newly exposed edges stay invalid.
        valid_float, valid_warp = warp_translation(np.asarray(previous.get("valid"),dtype="float32"),
                                                   reg["shift_x_px"], reg["shift_y_px"], categorical=True)
        aligned_valid = valid_warp & (np.nan_to_num(valid_float, nan=0.0) >= 0.5)
        aligned["valid"] = aligned_valid

    fused, summary = composite_change(latest, aligned, signals)
    summary["registration_anchor"] = anchor
    summary["registration"] = reg
    summary["registration_gate"] = gate
    summary["geometrically_ambiguous"] = not gate["pass"]
    if not gate["pass"]:
        summary["note"] = (
            "Change is shown for review, but registration quality is insufficient for confident semantic attribution."
        )
    else:
        summary["note"] = (
            "Residual translation was estimated and the previous observation aligned before multi-signal change screening."
        )
    return fused, summary

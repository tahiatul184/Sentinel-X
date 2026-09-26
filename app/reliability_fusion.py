"""Reliability-aware multi-sensor fusion utilities for AEROSENTINEL."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Mapping, Sequence
import numpy as np


def _clip(x, lo=0.0, hi=1.0):
    try: x = float(x)
    except (TypeError, ValueError): return lo
    if not math.isfinite(x): return lo
    return float(np.clip(x, lo, hi))


def _dt(value):
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def scene_reliability(row: Mapping, *, reference_time=None, temporal_tolerance_hours=72.0) -> dict:
    """Build a transparent scalar reliability weight for one scene.

    The weight is a fusion-control heuristic, not a probability that the scene is
    correct.  It only determines how strongly one sensor can influence a fused
    screening surface.
    """
    valid = _clip(float(row.get("valid_percent") or 0.0) / 100.0)
    processing = _clip((row.get("processing_quality") or {}).get("processing_readiness_score", 0.65))
    cloud = row.get("cloud_cover_percent")
    cloud_factor = 1.0
    if str(row.get("kind") or "").lower() == "optical" and cloud is not None:
        cloud_factor = 1.0 - _clip(float(cloud) / 100.0)
    temporal = 1.0
    age_hours = 0.0
    if reference_time is not None and row.get("acquired_at"):
        age_hours = abs((_dt(reference_time) - _dt(row["acquired_at"])).total_seconds()) / 3600.0
        temporal = math.exp(-age_hours / max(float(temporal_tolerance_hours), 1e-6))
    # Geometric mean prevents one excellent metric from fully hiding a poor one.
    raw = max(1e-6, valid * processing * max(cloud_factor, 0.05) * max(temporal, 0.05)) ** 0.25
    weight = float(np.clip(raw, 0.05, 1.0))
    return {
        "weight": weight, "valid_factor": valid, "processing_factor": processing,
        "cloud_factor": cloud_factor, "temporal_factor": temporal,
        "age_from_reference_hours": float(age_hours),
        "note": "Transparent fusion weight; not a calibrated sensor-accuracy probability.",
    }


def weighted_fusion(signals: Sequence[np.ndarray], weights: Sequence[float]):
    if len(signals) != len(weights) or not signals:
        raise ValueError("signals and weights must be non-empty and have equal length")
    stack = np.stack([np.asarray(s, dtype="float32") for s in signals])
    w = np.asarray(weights, dtype="float32").reshape((-1, 1, 1))
    finite = np.isfinite(stack)
    ww = np.where(finite, w, 0.0)
    denom = np.sum(ww, axis=0)
    num = np.nansum(np.where(finite, stack * w, 0.0), axis=0)
    consensus = np.full(stack.shape[1:], np.nan, dtype="float32")
    ok = denom > 0
    consensus[ok] = num[ok] / denom[ok]
    variance = np.full_like(consensus, np.nan)
    if ok.any():
        diff2 = np.where(finite, (stack - consensus[None, ...]) ** 2, 0.0)
        variance[ok] = np.sum(diff2 * ww, axis=0)[ok] / denom[ok]
    disagreement = np.sqrt(np.maximum(variance, 0.0)).astype("float32")
    denom2 = np.sum(np.square(ww), axis=0)
    effective_n = np.zeros_like(consensus, dtype="float32")
    eff_ok = denom2 > 0
    effective_n[eff_ok] = (denom[eff_ok] ** 2) / denom2[eff_ok]
    count = finite.sum(axis=0).astype("float32")
    mean_reliability = np.zeros_like(consensus, dtype="float32")
    c_ok = count > 0
    mean_reliability[c_ok] = denom[c_ok] / count[c_ok]
    return consensus, disagreement, effective_n, mean_reliability


def counterfactual_source_contribution(signals: Sequence[np.ndarray], weights: Sequence[float], names: Sequence[str]):
    if len(signals) < 2:
        return {str(names[0]): 1.0} if names else {}
    full, _d, _n, _q = weighted_fusion(signals, weights)
    raw = {}
    for idx, name in enumerate(names):
        sub_s = [s for j, s in enumerate(signals) if j != idx]
        sub_w = [w for j, w in enumerate(weights) if j != idx]
        if not sub_s:
            raw[str(name)] = 0.0; continue
        alt, _d2, _n2, _q2 = weighted_fusion(sub_s, sub_w)
        ok = np.isfinite(full) & np.isfinite(alt)
        raw[str(name)] = float(np.mean(np.abs(full[ok] - alt[ok]))) if ok.any() else 0.0
    total = sum(raw.values())
    if total <= 1e-12:
        return {k: 0.0 for k in raw}
    return {k: float(v / total) for k, v in raw.items()}

"""Reliability-aware multimodal temporal pipeline for AEROSENTINEL v0.5.

Architecture implemented here:
Multi-source EO -> multimodal foundation representation -> temporal representation
-> change/anomaly/land-surface branches -> uncertainty -> reliability audit
-> current/future/risk -> predictive digital twin -> decision-support outputs.

Two representation backends are supported:
1) TerraMind-backed hybrid foundation stage when timestamped TerraMind
   embeddings are supplied. TerraMind provides the learned spatial EO backbone
   for Sentinel-1 + Sentinel-2 + DEM; weather/rainfall and historical-imagery
   context are fused through explicit context adapters inside the same overall
   foundation-representation stage before temporal processing.
2) A deterministic robust source-fusion fallback when no learned embedding is
   supplied. The fallback is intentionally labelled as such and is not claimed
   to be a foundation model.

All scores are research screening indicators, not calibrated event/failure
probabilities or operational decisions.
"""
from __future__ import annotations

import csv
import io
import json
import math
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np


SOURCE_GROUPS = {
    "Sentinel-1 SAR": ["s1_vv_db", "s1_vh_db"],
    "Sentinel-2 Optical": ["s2_ndvi", "s2_ndwi", "s2_nbr"],
    "DEM": ["dem_elevation_m", "dem_slope_deg"],
    "Weather / Rainfall": ["rain_mm_h", "rain_24h_mm", "forecast_rain_24h_mm"],
    "Historical imagery": [
        "historical_ndvi", "historical_ndwi", "historical_nbr",
        "historical_change_score",
    ],
}
FEATURE_COLUMNS = [name for values in SOURCE_GROUPS.values() for name in values]
BOUNDED_FEATURES = {
    "s2_ndvi": (-1.0, 1.0), "s2_ndwi": (-1.0, 1.0), "s2_nbr": (-1.0, 1.0),
    "historical_ndvi": (-1.0, 1.0), "historical_ndwi": (-1.0, 1.0),
    "historical_nbr": (-1.0, 1.0), "historical_change_score": (0.0, 1.0),
}
NONNEGATIVE_FEATURES = {
    "dem_slope_deg", "rain_mm_h", "rain_24h_mm", "forecast_rain_24h_mm",
}
DOMAIN_KEYS = (
    "airfield_risk_screening", "disaster_risk_screening",
    "infrastructure_resilience_screening",
)


def _timestamp(value: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Each multimodal row needs a timestamp.")
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid ISO timestamp: {raw}") from exc
    if dt.tzinfo is None:
        raise ValueError("Multimodal timestamps must include a timezone.")
    return dt.astimezone(timezone.utc)



def _season_label(value: str) -> str:
    dt = _timestamp(value)
    month = dt.month
    if month in (12,1,2): return "dry"
    if month in (3,4,5): return "pre-monsoon"
    if month in (6,7,8,9): return "monsoon"
    return "post-monsoon"

def _optional_float(value, column):
    if value is None or str(value).strip() == "" or (
        isinstance(value, (float, np.floating)) and np.isnan(value)
    ):
        return np.nan
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric or blank.") from exc
    if not math.isfinite(x):
        raise ValueError(f"{column} must be finite when supplied.")
    if column in BOUNDED_FEATURES:
        lo, hi = BOUNDED_FEATURES[column]
        if not lo <= x <= hi:
            raise ValueError(f"{column} must be between {lo:g} and {hi:g}.")
    if column in NONNEGATIVE_FEATURES and x < 0:
        raise ValueError(f"{column} cannot be negative.")
    if column == "dem_slope_deg" and x > 90:
        raise ValueError("dem_slope_deg cannot exceed 90 degrees.")
    if column in {"rain_mm_h", "rain_24h_mm", "forecast_rain_24h_mm"} and x > 2000:
        raise ValueError(f"{column} exceeds the research safety bound (2000 mm).")
    return x


def parse_multimodal_csv(text: str):
    """Parse timestamp-aligned multi-source features with explicit missing cells."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "timestamp" not in reader.fieldnames:
        raise ValueError("CSV must contain a timestamp column.")
    available = [c for c in FEATURE_COLUMNS if c in reader.fieldnames]
    if not available:
        raise ValueError("CSV has no recognized multimodal feature columns.")
    rows = []
    seen = set()
    for idx, raw in enumerate(reader, start=2):
        dt = _timestamp(raw.get("timestamp"))
        stamp = dt.isoformat()
        if stamp in seen:
            raise ValueError(f"Duplicate timestamp on CSV line {idx}: {stamp}")
        seen.add(stamp)
        row = {"timestamp": stamp}
        for col in FEATURE_COLUMNS:
            row[col] = _optional_float(raw.get(col), col) if col in reader.fieldnames else np.nan
        if not any(np.isfinite(row[col]) for col in FEATURE_COLUMNS):
            raise ValueError(f"CSV line {idx} contains no usable source values.")
        rows.append(row)
    if not 2 <= len(rows) <= 10000:
        raise ValueError("Provide between 2 and 10,000 timestamped multimodal rows.")
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def parse_foundation_records(records: Sequence[Mapping] | Mapping | str | bytes | None):
    """Validate TerraMind/foundation embedding records.

    Accepted forms are a record, a list of records, or JSON text/bytes. Each
    record must contain ``acquired_at`` or ``timestamp`` and a finite embedding
    vector. The runner in ``terramind_job.py`` produces this contract.
    """
    if records is None:
        return []
    if isinstance(records, bytes):
        records = records.decode("utf-8")
    if isinstance(records, str):
        records = json.loads(records)
    if isinstance(records, Mapping):
        if "records" in records:
            records = records["records"]
        else:
            records = [records]
    out = []
    seen = set()
    for rec in records:
        stamp = _timestamp(rec.get("timestamp") or rec.get("acquired_at")).isoformat()
        if stamp in seen:
            raise ValueError(f"Duplicate foundation embedding timestamp: {stamp}")
        seen.add(stamp)
        emb = np.asarray(rec.get("embedding"), dtype=float)
        if emb.ndim != 1 or not 8 <= emb.size <= 16384 or not np.isfinite(emb).all():
            raise ValueError("Foundation embedding must be a finite 1-D vector of 8-16,384 values.")
        backend = str(rec.get("model") or rec.get("backend") or "unknown").strip()
        out.append({
            "timestamp": stamp,
            "embedding": emb,
            "backend": backend,
            "modalities": list(rec.get("modalities") or []),
            "input_sha256": rec.get("input_sha256"),
        })
    out.sort(key=lambda r: r["timestamp"])
    dimensions = {r["embedding"].size for r in out}
    if len(dimensions) > 1:
        raise ValueError("All foundation embeddings in one run must have the same dimension.")
    return out


def _robust_standardize(matrix):
    """Column-wise robust scaling, preserving missing cells as NaN."""
    arr = np.asarray(matrix, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        values = col[finite]
        center = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        scale = float(q75 - q25)
        if scale < 1e-9:
            scale = float(np.std(values))
        if scale < 1e-9:
            scale = 1.0
        out[finite, j] = (values - center) / scale
    return np.clip(out, -8, 8)


def _source_matrix(z):
    result = {}
    offset = 0
    for source, cols in SOURCE_GROUPS.items():
        section = z[:, offset:offset + len(cols)]
        offset += len(cols)
        valid = np.isfinite(section)
        count = valid.sum(axis=1)
        summed = np.nansum(section, axis=1)
        mean = np.divide(summed, count, out=np.full(len(section), np.nan), where=count > 0)
        result[source] = mean
    return result


def _to_unit(value, scale=2.0):
    value = np.asarray(value, dtype=float)
    return 1.0 - np.exp(-np.maximum(value, 0.0) / scale)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _row_mean(values, fallback=None):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else fallback


def _score_from_named(zrow, feature_index, names, signs=None):
    signs = signs or [1.0] * len(names)
    values = []
    for name, sign in zip(names, signs):
        idx = feature_index[name]
        if np.isfinite(zrow[idx]):
            values.append(sign * zrow[idx])
    return float(_sigmoid(np.mean(values))) if values else None


def _historical_contrast(row, current, historical, *, invert=False):
    a = row.get(current)
    b = row.get(historical)
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return None
    delta = float(a - b)
    if invert:
        delta = -delta
    # Indices naturally live in [-1,1]; a ~0.2 deviation is already material.
    return float(np.clip(_sigmoid(delta / 0.20), 0, 1))


def _align_foundation(rows, foundation_records):
    """Align exact-timestamp embeddings and reduce them to stable PCs.

    Exact timestamps are deliberate: learned features should not silently be
    borrowed from a different acquisition. Missing embeddings are retained.
    """
    n = len(rows)
    if not foundation_records:
        return np.full((n, 0), np.nan), [None] * n, None
    by_time = {r["timestamp"]: r for r in foundation_records}
    dim = foundation_records[0]["embedding"].size
    matrix = np.full((n, dim), np.nan, dtype=float)
    meta = [None] * n
    for i, row in enumerate(rows):
        rec = by_time.get(row["timestamp"])
        if rec is not None:
            matrix[i] = rec["embedding"]
            meta[i] = {k: v for k, v in rec.items() if k != "embedding"}
    present = np.isfinite(matrix).all(axis=1)
    if present.sum() < 2:
        # One embedding can prove backend use but cannot support learned temporal change.
        return np.full((n, 0), np.nan), meta, foundation_records[0]["backend"]
    x = matrix[present]
    center = x.mean(axis=0)
    x0 = x - center
    scale = float(np.sqrt(np.mean(x0 ** 2)))
    if scale < 1e-9:
        scale = 1.0
    x0 = x0 / scale
    try:
        _, _, vt = np.linalg.svd(x0, full_matrices=False)
        k = min(4, vt.shape[0])
        projected = x0 @ vt[:k].T
    except np.linalg.LinAlgError:
        k = min(4, x0.shape[1])
        projected = x0[:, :k]
    pcs = np.full((n, k), np.nan, dtype=float)
    pcs[present] = projected
    pcs = _robust_standardize(pcs)
    backend = foundation_records[0]["backend"]
    return pcs, meta, backend


def _temporal_gap_penalty(stamps):
    times = np.array([_timestamp(s).timestamp() for s in stamps], dtype=float)
    if len(times) < 3:
        return np.zeros(len(times), dtype=float)
    gaps = np.diff(times) / 3600.0
    baseline = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 24.0
    baseline = max(baseline, 1e-6)
    penalties = np.zeros(len(times), dtype=float)
    penalties[1:] = np.clip((gaps / baseline - 1.0) / 3.0, 0, 1)
    return penalties


def _rolling(values, end, window=3):
    start = max(0, end - window + 1)
    arr = [float(v) for v in values[start:end + 1] if v is not None and np.isfinite(v)]
    return float(np.mean(arr)) if arr else 0.0


def _state_projection(current_scores, rows, index, horizon_steps):
    """Short-horizon bounded state projection with optional rain forcing.

    This is a transparent state-space sensitivity projection, not a calibrated
    meteorological or hazard forecast.
    """
    window = np.asarray(current_scores[max(0, index - 5):index + 1], dtype=float)
    if len(window) <= 1:
        slope = 0.0
    else:
        x = np.arange(len(window), dtype=float)
        # Give recent points more weight without introducing an opaque model.
        w = np.linspace(0.6, 1.0, len(window))
        xm = np.average(x, weights=w)
        ym = np.average(window, weights=w)
        denom = np.sum(w * (x - xm) ** 2)
        slope = float(np.sum(w * (x - xm) * (window - ym)) / denom) if denom > 1e-12 else 0.0
    row = rows[index]
    rain_now = row.get("rain_24h_mm")
    rain_forecast = row.get("forecast_rain_24h_mm")
    rain_force = 0.0
    if np.isfinite(rain_forecast):
        reference = float(rain_now) if np.isfinite(rain_now) else 0.0
        rain_force = float(np.clip((float(rain_forecast) - reference) / 250.0, -0.20, 0.30))
    future = float(np.clip(current_scores[index] + slope * horizon_steps + rain_force, 0, 1))
    return future, slope, rain_force


def _perturbation_sensitivity(base_parts, coverage, rng, n=64):
    """Sensitivity ensemble for transparent uncertainty estimation.

    Noise magnitude grows as source coverage falls. This quantifies local
    score sensitivity only; it is not a posterior predictive distribution.
    """
    parts = np.asarray(base_parts, dtype=float)
    parts = parts[np.isfinite(parts)]
    if not len(parts):
        return 1.0, (0.0, 0.5, 1.0)
    sigma = 0.04 + 0.16 * (1.0 - float(coverage))
    draws = np.clip(parts[None, :] + rng.normal(0, sigma, size=(n, len(parts))), 0, 1)
    scores = draws.mean(axis=1)
    sensitivity = float(np.clip(np.std(scores) / 0.18, 0, 1))
    q10, q50, q90 = np.quantile(scores, [0.10, 0.50, 0.90])
    return sensitivity, (float(q10), float(q50), float(q90))


def _domain_scores(surface, change, anomaly, rain_pressure, uncertainty):
    wet = surface.get("wetness_proxy")
    veg = surface.get("vegetation_stress_proxy")
    terrain = surface.get("terrain_exposure_proxy")
    hist = surface.get("historical_deviation_proxy")
    wet = 0.5 if wet is None else wet
    veg = 0.5 if veg is None else veg
    terrain = 0.5 if terrain is None else terrain
    hist = 0.5 if hist is None else hist
    airfield = float(np.clip(0.34 * wet + 0.20 * rain_pressure + 0.18 * change + 0.12 * anomaly + 0.10 * terrain + 0.06 * uncertainty, 0, 1))
    disaster = float(np.clip(0.27 * wet + 0.24 * rain_pressure + 0.21 * change + 0.14 * anomaly + 0.08 * hist + 0.06 * uncertainty, 0, 1))
    infra_risk = float(np.clip(0.23 * wet + 0.17 * rain_pressure + 0.18 * change + 0.17 * anomaly + 0.16 * terrain + 0.09 * uncertainty, 0, 1))
    resilience = float(np.clip(1.0 - infra_risk, 0, 1))
    return airfield, disaster, resilience


def run_multimodal_pipeline(rows: Iterable[dict], forecast_horizon_steps=1,
                            abstain_reliability=0.45, foundation_records=None,
                            ensemble_seed=2026):
    """Run the complete AEROSENTINEL v2.8.0 architecture on timestamp-aligned inputs.

    Learned TerraMind embeddings are optional at runtime because users may want
    to exercise the deterministic pipeline before installing the large model
    environment. ``architecture_compliance`` makes that distinction explicit.
    """
    rows = list(rows)
    if not 2 <= len(rows) <= 10000:
        raise ValueError("Provide between 2 and 10,000 multimodal rows.")
    horizon = int(forecast_horizon_steps)
    if not 1 <= horizon <= 24:
        raise ValueError("Forecast horizon must be 1-24 feature-table steps.")
    abstain_reliability = float(abstain_reliability)
    if not 0 <= abstain_reliability <= 1:
        raise ValueError("Abstention reliability threshold must be 0-1.")
    rng = np.random.default_rng(int(ensemble_seed))

    normalized_rows = []
    seen = set()
    for raw in rows:
        stamp = _timestamp(raw.get("timestamp")).isoformat()
        if stamp in seen:
            raise ValueError(f"Duplicate timestamp: {stamp}")
        seen.add(stamp)
        record = {"timestamp": stamp}
        for col in FEATURE_COLUMNS:
            record[col] = _optional_float(raw.get(col), col)
        if not any(np.isfinite(record[col]) for col in FEATURE_COLUMNS):
            raise ValueError(f"No usable source values at {stamp}.")
        normalized_rows.append(record)
    normalized_rows.sort(key=lambda r: r["timestamp"])

    foundation_records = parse_foundation_records(foundation_records)
    fpcs, foundation_meta, foundation_backend = _align_foundation(normalized_rows, foundation_records)

    matrix = np.array([[r[c] for c in FEATURE_COLUMNS] for r in normalized_rows], dtype=float)
    z = _robust_standardize(matrix)
    sources = _source_matrix(z)
    source_names = list(SOURCE_GROUPS)
    source_stack = np.column_stack([sources[s] for s in source_names])
    source_feature_completeness = np.zeros_like(source_stack, dtype=float)
    offset = 0
    for j, source in enumerate(source_names):
        width = len(SOURCE_GROUPS[source])
        source_feature_completeness[:, j] = np.isfinite(matrix[:, offset:offset+width]).mean(axis=1)
        offset += width
    source_present = np.isfinite(source_stack)
    source_count = source_present.sum(axis=1)
    source_coverage = source_count / len(source_names)
    # Partially populated source groups cannot influence the shared representation
    # as strongly as complete groups. This is a deterministic reliability gate,
    # not a learned claim of sensor accuracy.
    gated_source_stack = source_stack * np.sqrt(np.clip(source_feature_completeness, 0.0, 1.0))

    # Hybrid multimodal foundation stage: all five architecture source groups
    # enter one representation. TerraMind supplies learned S1/S2/DEM spatial
    # features; weather/rainfall and historical imagery enter via transparent
    # context adapters before temporal processing.
    representation_matrix = np.column_stack([gated_source_stack, fpcs]) if fpcs.shape[1] else gated_source_stack.copy()
    representation = []
    for i, row in enumerate(normalized_rows):
        rec = {
            "timestamp": row["timestamp"],
            "source_vector": {
                s: (None if not np.isfinite(source_stack[i, j]) else float(source_stack[i, j]))
                for j, s in enumerate(source_names)
            },
            "source_reliability_weights": {
                s: float(source_feature_completeness[i, j]) for j, s in enumerate(source_names)
            },
            "foundation_backend": foundation_meta[i]["backend"] if foundation_meta[i] else None,
            "foundation_components": [float(v) for v in fpcs[i] if np.isfinite(v)] if fpcs.shape[1] else [],
        }
        representation.append(rec)

    feature_index = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}
    gap_penalty = _temporal_gap_penalty([r["timestamp"] for r in normalized_rows])
    change_rows, anomaly_rows, surface_rows = [], [], []
    uncertainty_rows, temporal_rows = [], []
    current_scores = []
    branch_intervals = []
    seasons = [_season_label(r["timestamp"]) for r in normalized_rows]

    for i, row in enumerate(normalized_rows):
        rep_valid = np.isfinite(representation_matrix[i])
        if i == 0:
            change = 0.0
            source_changes = {s: None for s in source_names}
            fm_change = None
        else:
            pair = rep_valid & np.isfinite(representation_matrix[i - 1])
            distance = float(np.sqrt(np.mean((representation_matrix[i, pair] - representation_matrix[i - 1, pair]) ** 2))) if pair.any() else 0.0
            change = float(_to_unit(distance, scale=1.6))
            source_changes = {}
            for j, source in enumerate(source_names):
                a, b = source_stack[i, j], source_stack[i - 1, j]
                source_changes[source] = float(_to_unit(abs(a - b), 1.4)) if np.isfinite(a) and np.isfinite(b) else None
            if fpcs.shape[1] and np.isfinite(fpcs[i]).any() and np.isfinite(fpcs[i - 1]).any():
                p = np.isfinite(fpcs[i]) & np.isfinite(fpcs[i - 1])
                fm_change = float(_to_unit(np.sqrt(np.mean((fpcs[i, p] - fpcs[i - 1, p]) ** 2)), 1.4)) if p.any() else None
            else:
                fm_change = None
        change_rows.append({
            "timestamp": row["timestamp"], "change_score": change,
            "source_change": source_changes, "foundation_change": fm_change,
        })

        # Causal anomaly baseline: only preceding observations, avoiding leakage
        # from the current point into its own anomaly reference.
        if i == 0:
            anomaly = 0.0
            source_anomaly = {s: None for s in source_names}
        else:
            same_season_idx = [j for j in range(i) if seasons[j] == seasons[i]]
            if len(same_season_idx) >= 3:
                history = representation_matrix[same_season_idx]
                anomaly_baseline_strategy = "SAME_SEASON"
            else:
                history = representation_matrix[:i]
                anomaly_baseline_strategy = "ALL_HISTORY_FALLBACK"
            components = []
            source_anomaly = {}
            for j in range(representation_matrix.shape[1]):
                val = representation_matrix[i, j]
                hist = history[:, j]
                finite = hist[np.isfinite(hist)]
                if not np.isfinite(val) or len(finite) == 0:
                    continue
                med = float(np.median(finite))
                mad = float(np.median(np.abs(finite - med)))
                scale = max(1.4826 * mad, float(np.std(finite)), 0.5)
                components.append(abs(float(val) - med) / scale)
            anomaly = float(_to_unit(np.median(components), scale=1.7)) if components else 0.0
            for j, source in enumerate(source_names):
                val = source_stack[i, j]
                hist = source_stack[:i, j]
                finite = hist[np.isfinite(hist)]
                if np.isfinite(val) and len(finite):
                    med = float(np.median(finite)); mad = float(np.median(np.abs(finite - med)))
                    scale = max(1.4826 * mad, float(np.std(finite)), 0.5)
                    source_anomaly[source] = float(_to_unit(abs(float(val)-med)/scale, 1.7))
                else:
                    source_anomaly[source] = None
        if i == 0:
            anomaly_baseline_strategy = "INSUFFICIENT_HISTORY"
            anomaly_baseline_rows = 0
        else:
            anomaly_baseline_rows = int(history.shape[0])
        anomaly_rows.append({"timestamp": row["timestamp"], "anomaly_score": anomaly, "source_anomaly": source_anomaly,
                             "season":seasons[i], "baseline_strategy":anomaly_baseline_strategy,
                             "baseline_rows":anomaly_baseline_rows})

        wetness = _score_from_named(z[i], feature_index, ["s2_ndwi", "rain_mm_h", "rain_24h_mm"])
        # Lower-than-usual SAR backscatter is often compatible with smooth open
        # water, but this remains only a relative proxy without calibrated RTC.
        sar_water = _score_from_named(z[i], feature_index, ["s1_vv_db", "s1_vh_db"], signs=[-1.0, -1.0])
        if wetness is None:
            wetness = sar_water
        elif sar_water is not None:
            wetness = float(0.72 * wetness + 0.28 * sar_water)
        vegetation_stress = _score_from_named(z[i], feature_index, ["s2_ndvi", "s2_nbr"], signs=[-1.0, -1.0])
        terrain_exposure = _score_from_named(z[i], feature_index, ["dem_elevation_m", "dem_slope_deg"], signs=[-1.0, 1.0])
        hist_direct = row.get("historical_change_score")
        hist_contrasts = [
            _historical_contrast(row, "s2_ndwi", "historical_ndwi"),
            _historical_contrast(row, "s2_ndvi", "historical_ndvi", invert=True),
            _historical_contrast(row, "s2_nbr", "historical_nbr", invert=True),
        ]
        historical_deviation = _row_mean(
            [float(hist_direct) if np.isfinite(hist_direct) else None, *hist_contrasts], fallback=None
        )
        surface_score = _row_mean([wetness, vegetation_stress, terrain_exposure, historical_deviation], fallback=anomaly)
        if wetness is not None and wetness >= 0.67:
            label = "wetness-elevated"
        elif vegetation_stress is not None and vegetation_stress >= 0.67:
            label = "vegetation-stress-elevated"
        elif historical_deviation is not None and historical_deviation >= 0.67:
            label = "historical-deviation-elevated"
        else:
            label = "no-dominant-proxy"
        surface = {
            "timestamp": row["timestamp"], "wetness_proxy": wetness,
            "sar_water_proxy": sar_water,
            "vegetation_stress_proxy": vegetation_stress,
            "terrain_exposure_proxy": terrain_exposure,
            "historical_deviation_proxy": historical_deviation,
            "surface_condition_score": float(surface_score),
            "relative_surface_state": label,
        }
        surface_rows.append(surface)

        present_values = source_stack[i, source_present[i]]
        disagreement = float(np.std(present_values)) if len(present_values) > 1 else 1.0
        disagreement_unit = float(_to_unit(disagreement, scale=1.5))
        missing_penalty = 1.0 - float(source_coverage[i])
        rain_pressure = _score_from_named(z[i], feature_index, ["rain_mm_h", "rain_24h_mm", "forecast_rain_24h_mm"])
        rain_pressure = 0.5 if rain_pressure is None else rain_pressure

        # Preliminary state without uncertainty, then assess sensitivity to its branches.
        preliminary_parts = [change, anomaly, float(surface_score), rain_pressure]
        sensitivity, interval = _perturbation_sensitivity(preliminary_parts, source_coverage[i], rng)
        temporal_support_penalty = 0.25 if i == 0 else (0.12 if i == 1 else 0.0)
        present_quality = source_feature_completeness[i, source_present[i]]
        source_quality_penalty = 1.0 - float(np.mean(present_quality)) if present_quality.size else 1.0
        fm_penalty = 0.0
        if foundation_records:
            fm_penalty = 0.0 if foundation_meta[i] else 0.10
        uncertainty = float(np.clip(
            0.35 * missing_penalty + 0.18 * disagreement_unit +
            0.17 * sensitivity + 0.10 * gap_penalty[i] +
            0.08 * temporal_support_penalty + 0.12 * source_quality_penalty + fm_penalty,
            0, 1,
        ))
        reliability = float(1.0 - uncertainty)
        reasons = []
        if source_count[i] < 5: reasons.append(f"{5-int(source_count[i])} source group(s) missing")
        if source_quality_penalty > 0.35: reasons.append("one or more source groups are only partially populated")
        if disagreement_unit > 0.55: reasons.append("high cross-source disagreement")
        if sensitivity > 0.55: reasons.append("high perturbation sensitivity")
        if gap_penalty[i] > 0.35: reasons.append("large temporal gap")
        if foundation_records and not foundation_meta[i]: reasons.append("foundation embedding missing at timestamp")
        if source_count[i] < 2 or reliability < abstain_reliability:
            audit = "ABSTAIN"
        elif source_count[i] < 4 or reliability < 0.72 or sensitivity > 0.65:
            audit = "REVIEW"
        else:
            audit = "PASS"
        if not reasons and audit == "PASS": reasons.append("prototype checks satisfied")
        uncertainty_rows.append({
            "timestamp": row["timestamp"], "source_count": int(source_count[i]),
            "source_coverage": float(source_coverage[i]),
            "source_disagreement": disagreement_unit,
            "perturbation_sensitivity": sensitivity,
            "temporal_gap_penalty": float(gap_penalty[i]),
            "uncertainty_score": uncertainty, "reliability_score": reliability,
            "audit": audit, "audit_reasons": reasons,
        })
        branch_intervals.append(interval)

        state = float(np.clip(0.31 * change + 0.28 * anomaly + 0.27 * float(surface_score) + 0.14 * rain_pressure, 0, 1))
        current_scores.append(state)
        prev = current_scores[i - 1] if i else state
        velocity = float(state - prev)
        previous_velocity = temporal_rows[i - 1]["state_velocity"] if i else 0.0
        temporal_rows.append({
            "timestamp": row["timestamp"], "current_state_score": state,
            "delta_from_previous": velocity,
            "state_velocity": velocity,
            "state_acceleration": float(velocity - previous_velocity),
            "rolling_state_3": _rolling(current_scores, i, 3),
            "rolling_state_6": _rolling(current_scores, i, 6),
            "representation_dimension": int(np.isfinite(representation_matrix[i]).sum()),
        })

    outputs = []
    forecast_rows = []
    for i, row in enumerate(normalized_rows):
        future, slope, rain_force = _state_projection(current_scores, normalized_rows, i, horizon)
        u = uncertainty_rows[i]["uncertainty_score"]
        current = current_scores[i]
        surface = surface_rows[i]
        rain_pressure = _score_from_named(z[i], feature_index, ["rain_mm_h", "rain_24h_mm", "forecast_rain_24h_mm"])
        rain_pressure = 0.5 if rain_pressure is None else rain_pressure
        airfield, disaster, resilience = _domain_scores(surface, change_rows[i]["change_score"], anomaly_rows[i]["anomaly_score"], rain_pressure, u)
        # Future domain movement is bounded by the state change; resilience moves inversely.
        delta_future = future - current
        airfield_future = float(np.clip(airfield + 0.65 * delta_future, 0, 1))
        disaster_future = float(np.clip(disaster + 0.75 * delta_future, 0, 1))
        resilience_future = float(np.clip(resilience - 0.60 * delta_future, 0, 1))
        risk = float(np.clip(0.36 * current + 0.34 * future + 0.18 * max(airfield, disaster) + 0.12 * u, 0, 1))
        spread = 0.06 + 0.22 * u
        risk_p10 = float(np.clip(risk - spread, 0, 1))
        risk_p90 = float(np.clip(risk + spread, 0, 1))
        supported = uncertainty_rows[i]["audit"] != "ABSTAIN"
        outputs.append({
            "timestamp": row["timestamp"],
            "current_state": current,
            "future_prediction": future,
            "forecast_trend_per_step": slope,
            "forecast_rain_forcing": rain_force,
            "risk_screening": risk,
            "risk_sensitivity_p10": risk_p10,
            "risk_sensitivity_p90": risk_p90,
            "airfield_risk_screening": airfield,
            "airfield_future_screening": airfield_future,
            "disaster_risk_screening": disaster,
            "disaster_future_screening": disaster_future,
            "infrastructure_resilience_screening": resilience,
            "infrastructure_future_resilience": resilience_future,
            "reliability": uncertainty_rows[i]["reliability_score"],
            "audit": uncertainty_rows[i]["audit"],
            "supported_for_review": supported,
        })
        forecast_rows.append({
            "timestamp": row["timestamp"], "horizon_steps": horizon,
            "projected_state": future, "trend_per_step": slope,
            "rain_forcing": rain_force, "sensitivity_p10": risk_p10,
            "sensitivity_p90": risk_p90,
            "note": "Transparent short-horizon state projection; not a calibrated weather/hazard forecast.",
        })

    source_summary = {}
    for source, cols in SOURCE_GROUPS.items():
        indices = [FEATURE_COLUMNS.index(c) for c in cols]
        any_present = np.isfinite(matrix[:, indices]).any(axis=1)
        source_summary[source] = {
            "rows_available": int(any_present.sum()),
            "coverage": float(any_present.mean()),
            "features": cols,
        }

    learned_rows = sum(m is not None for m in foundation_meta)
    embedding_active = learned_rows >= 2 and foundation_backend is not None
    required_tm_modalities = {"S2L2A", "S1GRD", "DEM"}
    strict_foundation_rows = sum(
        bool(m)
        and str(m.get("backend", "")).lower().startswith("terramind")
        and required_tm_modalities.issubset(set(m.get("modalities") or []))
        for m in foundation_meta
    )
    terramind_active = strict_foundation_rows >= 2
    all_source_groups_seen = all(v["rows_available"] > 0 for v in source_summary.values())
    complete_source_coverage = all(abs(v["coverage"] - 1.0) < 1e-12 for v in source_summary.values())
    strict_foundation_coverage = strict_foundation_rows == len(normalized_rows)
    strict_run = bool(complete_source_coverage and strict_foundation_coverage and len(normalized_rows) >= 2)
    architecture_compliance = {
        "multi_source_earth_observation": "IMPLEMENTED" if all_source_groups_seen else "IMPLEMENTED_WITH_MISSING_RUNTIME_SOURCES",
        "sentinel_1_sar": "IMPLEMENTED",
        "sentinel_2_optical": "IMPLEMENTED",
        "dem": "IMPLEMENTED",
        "weather_rainfall": "IMPLEMENTED",
        "historical_imagery": "IMPLEMENTED",
        "multimodal_foundation_model": (
            "ACTIVE_TERRAMIND_HYBRID" if terramind_active else
            ("EXTERNAL_EMBEDDING_AUGMENTED_FALLBACK" if embedding_active else "FALLBACK_TRANSPARENT_FUSION")
        ),
        "temporal_representation": "IMPLEMENTED",
        "change_detection": "IMPLEMENTED",
        "anomaly_detection": "IMPLEMENTED",
        "land_surface_understanding": "IMPLEMENTED_AS_RELATIVE_PROXIES",
        "uncertainty_estimation": "IMPLEMENTED_SENSITIVITY_ENSEMBLE",
        "reliability_audit": "IMPLEMENTED_PASS_REVIEW_ABSTAIN",
        "current_state": "IMPLEMENTED",
        "future_prediction": "IMPLEMENTED_SHORT_HORIZON_STATE_PROJECTION",
        "risk": "IMPLEMENTED_SCREENING_NOT_CALIBRATED_PROBABILITY",
        "predictive_digital_twin": "IMPLEMENTED",
        "decision_support_map": "IMPLEMENTED_AS_ASSET_PRIORITY_SURFACE",
        "airfield_risk": "IMPLEMENTED_SCREENING",
        "disaster_risk": "IMPLEMENTED_SCREENING",
        "infrastructure_resilience": "IMPLEMENTED_SCREENING",
        "strict_architecture_run": strict_run,
        "strict_source_coverage": complete_source_coverage,
        "strict_terramind_coverage": strict_foundation_coverage,
    }

    latest = outputs[-1]
    if terramind_active:
        encoder_text = (
            f"Hybrid multimodal foundation stage: TerraMind spatial backbone ({foundation_backend}) "
            "for Sentinel-1 + Sentinel-2 + DEM, with weather/rainfall and historical-imagery "
            "context adapters fused before temporal processing."
        )
    elif embedding_active:
        encoder_text = (
            f"External embedding backend ({foundation_backend}) is present, but the strict TerraMind "
            "S1GRD+S2L2A+DEM contract is not satisfied; the run remains non-strict."
        )
    else:
        encoder_text = (
            "Transparent robust five-source fusion fallback; run TerraMind for learned "
            "Sentinel-1 + Sentinel-2 + DEM embeddings to activate the strict foundation path."
        )
    foundation_stage = {
        "mode": architecture_compliance["multimodal_foundation_model"],
        "learned_backbone": foundation_backend if terramind_active else None,
        "learned_backbone_modalities": ["S1GRD", "S2L2A", "DEM"],
        "context_adapter_inputs": ["Weather / Rainfall", "Historical imagery"],
        "all_architecture_input_groups": source_names,
        "representation_contract": "all five source groups fused before temporal representation",
        "strict_requirement": "complete five-source coverage and timestamp-matched TerraMind S1GRD+S2L2A+DEM embeddings for every row",
    }
    return {
        "architecture": "AEROSENTINEL multimodal spatio-temporal predictive twin v2.8.0",
        "architecture_compliance": architecture_compliance,
        "encoder": encoder_text,
        "foundation_records_used": learned_rows,
        "strict_terramind_records_used": strict_foundation_rows,
        "foundation_backend": foundation_backend,
        "score_scope": "Relative research screening indicators and sensitivity bands; not calibrated event/failure probabilities or operational decisions.",
        "multimodal_foundation_stage": foundation_stage,
        "forecast_horizon_steps": horizon,
        "source_summary": source_summary,
        "representation": representation,
        "temporal_representation": temporal_rows,
        "change_detection": change_rows,
        "anomaly_detection": anomaly_rows,
        "land_surface_understanding": surface_rows,
        "uncertainty_estimation": uncertainty_rows,
        "reliability_audit": {
            "pass": sum(v["audit"] == "PASS" for v in uncertainty_rows),
            "review": sum(v["audit"] == "REVIEW" for v in uncertainty_rows),
            "abstain": sum(v["audit"] == "ABSTAIN" for v in uncertainty_rows),
            "latest": uncertainty_rows[-1],
        },
        "future_prediction": forecast_rows,
        "predictive_digital_twin": outputs,
        "latest": latest,
    }


def _asset_domain(asset_type):
    text = str(asset_type or "").lower()
    if any(k in text for k in ("runway", "taxi", "apron", "airfield", "aviation")):
        return "airfield"
    if any(k in text for k in ("shelter", "evac", "emergency", "disaster")):
        return "disaster"
    return "infrastructure"


def asset_decision_rows(assets, latest):
    """Attach domain-aware latest screening to registered assets."""
    rows = []
    for asset in assets:
        payload = asset.get("payload", asset)
        criticality = int(round(float(payload.get("criticality", 3))))
        domain = _asset_domain(payload.get("type"))
        if domain == "airfield":
            hazard = float(latest["airfield_risk_screening"])
            future = float(latest.get("airfield_future_screening", hazard))
        elif domain == "disaster":
            hazard = float(latest["disaster_risk_screening"])
            future = float(latest.get("disaster_future_screening", hazard))
        else:
            hazard = float(1.0 - latest["infrastructure_resilience_screening"])
            future = float(1.0 - latest.get("infrastructure_future_resilience", latest["infrastructure_resilience_screening"]))
        criticality_factor = 0.70 + 0.075 * np.clip(criticality, 1, 5)
        priority = float(np.clip(max(hazard, future) * criticality_factor, 0, 1))
        rows.append({
            "asset": asset.get("title") or payload.get("name") or "Asset",
            "site": payload.get("site"), "type": payload.get("type"),
            "domain": domain,
            "latitude": float(payload.get("latitude")),
            "longitude": float(payload.get("longitude")),
            "criticality": criticality,
            "screening_priority": priority,
            "risk_screening": hazard,
            "current_domain_screening": hazard,
            "future_domain_screening": future,
            "reliability": float(latest["reliability"]),
            "audit": latest["audit"],
            "supported_for_review": bool(latest.get("supported_for_review", latest["audit"] != "ABSTAIN")),
        })
    return rows


def decision_surface_points(decisions, grid_size=17, padding_fraction=0.20):
    """Create an IDW *review-priority* surface for the decision-support map.

    This is intentionally not called a hazard raster. It visualizes how asset
    review priorities cluster spatially using only registered asset points.
    """
    decisions = list(decisions)
    if not decisions:
        return []
    lats = np.array([float(d["latitude"]) for d in decisions])
    lons = np.array([float(d["longitude"]) for d in decisions])
    weights = np.array([float(d["screening_priority"]) for d in decisions])
    if len(decisions) == 1:
        return [{"latitude": float(lats[0]), "longitude": float(lons[0]), "priority": float(weights[0])}]
    lat_span = max(float(np.ptp(lats)), 0.002)
    lon_span = max(float(np.ptp(lons)), 0.002)
    lat_min, lat_max = lats.min() - lat_span * padding_fraction, lats.max() + lat_span * padding_fraction
    lon_min, lon_max = lons.min() - lon_span * padding_fraction, lons.max() + lon_span * padding_fraction
    size = int(np.clip(grid_size, 5, 41))
    points = []
    for lat in np.linspace(lat_min, lat_max, size):
        for lon in np.linspace(lon_min, lon_max, size):
            # Longitude is scaled by latitude to avoid obvious east-west distortion.
            dx = (lons - lon) * max(math.cos(math.radians(float(lat))), 0.2)
            dy = lats - lat
            d2 = dx * dx + dy * dy
            if np.min(d2) < 1e-12:
                value = float(weights[np.argmin(d2)])
            else:
                inv = 1.0 / d2
                value = float(np.sum(inv * weights) / np.sum(inv))
            points.append({"latitude": float(lat), "longitude": float(lon), "priority": value})
    return points


def template_csv():
    return """timestamp,s1_vv_db,s1_vh_db,s2_ndvi,s2_ndwi,s2_nbr,dem_elevation_m,dem_slope_deg,rain_mm_h,rain_24h_mm,forecast_rain_24h_mm,historical_ndvi,historical_ndwi,historical_nbr,historical_change_score
2026-09-10T06:00:00Z,-11.2,-18.4,0.54,0.08,0.38,7.2,1.8,0.0,12.0,18.0,0.56,0.06,0.41,0.12
2026-09-11T06:00:00Z,-12.8,-19.7,0.49,0.16,0.31,7.2,1.8,18.0,42.0,60.0,0.56,0.06,0.41,0.24
2026-09-12T06:00:00Z,-15.1,-22.0,0.42,0.31,0.20,7.2,1.8,46.0,108.0,125.0,0.56,0.06,0.41,0.51
"""

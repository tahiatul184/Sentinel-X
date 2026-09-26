"""AEROSENTINEL research early-warning engine.

This module implements the thesis/research architecture on top of AEROSENTINEL's
open EO pipeline. It is deliberately conservative: outputs are AOI-level
research screening indicators for human review. They are not target labels,
not calibrated threat probabilities, and never infer an "enemy UAV".
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

FEATURES = (
    "wetness_screening",
    "temporal_change_screening",
    "fusion_uncertainty",
    "source_coverage",
    "temporal_alignment",
    "optical_ndvi",
    "rain_pressure",
    "wind_pressure",
    "visibility_pressure",
    "convective_pressure",
    "airfield_environment_screening",
    "disaster_response_screening",
    "infrastructure_continuity_screening",
)

STATUS_ORDER = ("NORMAL", "MONITOR", "ELEVATED", "INVESTIGATE")

def bangladesh_season(value) -> str:
    """Simple climatological grouping used only for baseline stratification."""
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        month = dt.month
    except Exception:
        return "unknown"
    if month in (12, 1, 2):
        return "dry"
    if month in (3, 4, 5):
        return "pre-monsoon"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post-monsoon"



def _clip(value, lo=0.0, hi=1.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, lo, hi))


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _mean_change(fusion: Mapping) -> float:
    rows = (fusion or {}).get("temporal_change") or {}
    vals = []
    for row in rows.values():
        val = row.get("mean_absolute_screening_change") if isinstance(row, Mapping) else None
        if val is not None:
            vals.append(_clip(val))
    return float(np.mean(vals)) if vals else 0.0


def _pressure(value, scale, *, invert=False):
    if value is None:
        return 0.0
    p = _clip(_number(value) / max(float(scale), 1e-9))
    return 1.0 - p if invert else p


def build_observation(multisatellite_run: Mapping | None, weather: Mapping | None,
                      terrain: Mapping | None, assessment: Mapping | None,
                      *, area_name="Selected area") -> dict:
    """Create one timestamped, label-free EO/context observation.

    The feature vector intentionally uses transparent screening variables so it
    can be audited even when a foundation-model embedding is unavailable.
    """
    run = multisatellite_run or {}
    fusion = run.get("fusion") or {}
    wsum = (weather or {}).get("summary") or {}
    scores = (assessment or {}).get("scores") or {}
    sats = fusion.get("available_satellites") or []
    requested = fusion.get("requested_satellites") or run.get("requested_satellites") or []
    scene_results = run.get("scene_results") or []

    cloud_vals = [
        _number(r.get("cloud_cover_percent")) for r in scene_results
        if r.get("cloud_cover_percent") is not None and r.get("kind") == "optical"
    ]
    valid_vals = [_number(r.get("valid_percent"), 0.0) for r in scene_results if r.get("valid_percent") is not None]
    gsd_vals = [_number(r.get("native_gsd_m"), 0.0) for r in scene_results if r.get("native_gsd_m")]
    processing_vals = [
        _number((r.get("processing_quality") or {}).get("processing_readiness_score"), np.nan)
        for r in scene_results
        if (r.get("processing_quality") or {}).get("processing_readiness_score") is not None
    ]
    registration = fusion.get("registration_quality") or {}

    features = {
        "wetness_screening": _clip(fusion.get("mean_wetness_screening")),
        "temporal_change_screening": _mean_change(fusion),
        "fusion_uncertainty": _clip(fusion.get("mean_fusion_uncertainty"), 0, 1),
        "source_coverage": _clip(fusion.get("mean_source_coverage"), 0, 1),
        "temporal_alignment": _clip(fusion.get("temporal_alignment_score"), 0, 1),
        "optical_ndvi": _clip((_number(fusion.get("mean_optical_ndvi"), 0.0) + 1.0) / 2.0),
        "rain_pressure": _pressure(wsum.get("rain_next_24h_mm"), 80.0),
        "wind_pressure": _pressure(wsum.get("max_gust_next_6h_kmh"), 80.0),
        "visibility_pressure": 1.0 - _clip(_number(wsum.get("min_visibility_next_6h_m"), 10000.0) / 10000.0),
        "convective_pressure": _clip(_number(wsum.get("thunderstorm_hours_next_24h"), 0.0) / 6.0),
        "airfield_environment_screening": _clip(scores.get("airfield_environment_screening")),
        "disaster_response_screening": _clip(scores.get("disaster_response_screening")),
        "infrastructure_continuity_screening": _clip(scores.get("infrastructure_continuity_screening")),
    }
    timestamp = run.get("finished_at") or (assessment or {}).get("generated_at") or datetime.now(timezone.utc).isoformat()
    quality = {
        "source_count": len(sats),
        "requested_source_count": len(requested),
        "source_coverage": features["source_coverage"],
        "temporal_alignment": features["temporal_alignment"],
        "mean_optical_cloud_percent": float(np.mean(cloud_vals)) if cloud_vals else None,
        "mean_valid_percent": float(np.mean(valid_vals)) if valid_vals else None,
        "acquisition_skew_hours": fusion.get("acquisition_skew_hours"),
        "best_native_gsd_m": min(gsd_vals) if gsd_vals else None,
        "mean_processing_readiness": float(np.mean(processing_vals)) if processing_vals else None,
        "registration_confidence": registration.get("mean_confidence"),
        "registration_max_uncertainty_px": registration.get("max_uncertainty_px"),
        "registration_ambiguous_pairs": registration.get("ambiguous_pairs"),
        "fusion_source_reliability": fusion.get("mean_fusion_source_reliability"),
        "effective_source_count": fusion.get("mean_effective_source_count"),
        "source_reliability_weights": fusion.get("source_reliability_weights") or {},
        "counterfactual_source_contribution": fusion.get("counterfactual_source_contribution") or {},
        "weather_available": bool(wsum),
        "terrain_available": bool((terrain or {}).get("summary")),
    }
    observed = {
        "satellites": list(sats),
        "scene_count": len(scene_results),
        "wetness_screening": features["wetness_screening"],
        "temporal_change_screening": features["temporal_change_screening"],
        "mean_optical_ndvi_unit": features["optical_ndvi"],
        "mean_optical_built_surface_screening": fusion.get("mean_optical_built_surface_screening"),
        "rain_next_24h_mm": wsum.get("rain_next_24h_mm"),
        "max_gust_next_6h_kmh": wsum.get("max_gust_next_6h_kmh"),
        "min_visibility_next_6h_m": wsum.get("min_visibility_next_6h_m"),
        "terrain_summary": (terrain or {}).get("summary") or {},
    }
    body = {"timestamp": timestamp, "season": bangladesh_season(timestamp),
            "features": features, "quality": quality, "area_name": area_name}
    signature = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return {**body, "observed_evidence": observed, "signature": signature}


def _matrix(observations: Sequence[Mapping]):
    arr = np.array([[ _number((o.get("features") or {}).get(k), np.nan) for k in FEATURES] for o in observations], dtype=float)
    return arr


def _robust_stats(matrix):
    med = np.nanmedian(matrix, axis=0)
    q25 = np.nanpercentile(matrix, 25, axis=0)
    q75 = np.nanpercentile(matrix, 75, axis=0)
    scale = (q75 - q25) / 1.349
    mad = np.nanmedian(np.abs(matrix - med), axis=0) * 1.4826
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, mad)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 0.10)
    return med, scale


def _standardize(matrix, med, scale):
    z = (matrix - med) / scale
    return np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0)


def unlabelled_representation(observations: Sequence[Mapping], latest: Mapping,
                              foundation_record: Mapping | None = None) -> dict:
    """Return an unlabelled representation with optional TerraMind augmentation."""
    all_obs = list(observations)
    if not all_obs or all_obs[-1].get("signature") != latest.get("signature"):
        all_obs.append(latest)
    matrix = _matrix(all_obs)
    med, scale = _robust_stats(matrix)
    z = _standardize(matrix, med, scale)
    n = len(all_obs)
    # Unlabelled SVD is an honest fallback representation baseline; it is not
    # called a foundation model. TerraMind, when present, remains separately named.
    if n >= 3:
        centered = z - np.mean(z, axis=0, keepdims=True)
        try:
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
            dims = max(1, min(6, vt.shape[0]))
            vector = (centered[-1] @ vt[:dims].T).astype(float)
        except np.linalg.LinAlgError:
            vector = z[-1, :6].astype(float)
        backend = "UNLABELLED_SVD_BASELINE"
    else:
        vector = z[-1, :6].astype(float)
        backend = "TRANSPARENT_FEATURE_BASELINE"

    tm = None
    if foundation_record:
        embedding = np.asarray(foundation_record.get("embedding") or [], dtype=float)
        modalities = {str(x).upper() for x in foundation_record.get("modalities") or []}
        if embedding.size and np.isfinite(embedding).all() and {"S1GRD", "S2L2A", "DEM"}.issubset(modalities):
            # Keep vectors separate instead of pretending a task-trained fusion head exists.
            tm = {
                "model": foundation_record.get("model") or "TerraMind",
                "dimension": int(embedding.size),
                "modalities": sorted(modalities),
                "pooled_norm": float(np.linalg.norm(embedding) / math.sqrt(max(1, embedding.size))),
                "acquired_at": foundation_record.get("acquired_at"),
            }
            backend = "TERRAMIND_PLUS_CONTEXT"
    return {
        "backend": backend,
        "context_embedding": vector.tolist(),
        "context_dimension": int(vector.size),
        "foundation": tm,
        "no_manual_labels_required": True,
        "history_rows": n,
        "feature_names": list(FEATURES),
    }


def _score_against_baseline(observation: Mapping, baseline: Sequence[Mapping]):
    if not baseline:
        return 0.0, 0.0, {}, True
    base = _matrix(baseline)
    current = _matrix([observation])[0]
    med, scale = _robust_stats(base)
    z = np.abs((current - med) / scale)
    z = np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=8.0)
    # Robust open-world score: both overall distance and strongest deviations matter.
    rms = float(np.sqrt(np.mean(np.square(np.clip(z, 0, 8)))))
    top = float(np.mean(np.sort(z)[-min(3, len(z)) :]))
    raw = 0.6 * rms + 0.4 * top
    score = 1.0 - math.exp(-raw / 2.8)
    ood = 1.0 - math.exp(-max(0.0, float(np.max(z)) - 2.0) / 3.0)
    by_feature = {name: float(value) for name, value in zip(FEATURES, z)}
    return _clip(score), _clip(ood), by_feature, len(baseline) < 4


def _seasonal_baseline(current: Mapping, baseline: Sequence[Mapping], minimum_same_season=3):
    """Prefer same-season history when enough rows exist; otherwise use all history."""
    current_season = current.get("season") or bangladesh_season(current.get("timestamp"))
    same = [
        row for row in baseline
        if (row.get("season") or bangladesh_season(row.get("timestamp"))) == current_season
    ]
    if current_season != "unknown" and len(same) >= int(minimum_same_season):
        return same, "SAME_SEASON"
    return list(baseline), "ALL_HISTORY_FALLBACK"


def anomaly_trajectory(observations: Sequence[Mapping], minimum_baseline=3):
    obs = list(observations)
    rows = []
    for idx, current in enumerate(obs):
        prior = obs[:idx]
        if len(prior) < minimum_baseline:
            rows.append({"timestamp": current.get("timestamp"), "season": current.get("season") or bangladesh_season(current.get("timestamp")),
                         "anomaly_score": 0.0, "ood_score": 0.0, "baseline_rows": len(prior),
                         "baseline_strategy":"INSUFFICIENT_HISTORY", "baseline_immature": True})
            continue
        baseline, strategy = _seasonal_baseline(current, prior, minimum_same_season=minimum_baseline)
        score, ood, _z, immature = _score_against_baseline(current, baseline)
        rows.append({"timestamp": current.get("timestamp"), "season": current.get("season") or bangladesh_season(current.get("timestamp")),
                     "anomaly_score": score, "ood_score": ood, "baseline_rows": len(baseline),
                     "total_prior_rows":len(prior), "baseline_strategy":strategy, "baseline_immature": immature})
    return rows

def temporal_reasoning(trajectory: Sequence[Mapping]) -> dict:
    usable = [r for r in trajectory if not r.get("baseline_immature")]
    if not usable:
        return {"persistence": "INSUFFICIENT HISTORY", "persistence_score": 0.0,
                "consecutive_anomalies": 0, "repeated_anomalies": 0, "trend": "UNKNOWN",
                "trend_slope": 0.0, "forecast": [0.0, 0.0, 0.0], "seasonal_baseline": "INSUFFICIENT HISTORY"}
    values = np.array([_clip(r.get("anomaly_score")) for r in usable], dtype=float)
    threshold = 0.50
    consecutive = 0
    for value in values[::-1]:
        if value >= threshold:
            consecutive += 1
        else:
            break
    repeated = int(np.sum(values >= threshold))
    if len(values) >= 2:
        x = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(x[-min(6, len(x)):], values[-min(6, len(values)):], 1)[0])
    else:
        slope = 0.0
    if slope > 0.05:
        trend = "INCREASING"
    elif slope < -0.05:
        trend = "DECREASING"
    else:
        trend = "STABLE"
    persistence_score = _clip(0.20 * repeated + 0.25 * consecutive)
    persistence = "HIGH" if consecutive >= 3 or persistence_score >= 0.75 else "MODERATE" if consecutive >= 2 or persistence_score >= 0.45 else "LOW"
    last = float(values[-1])
    forecast = [_clip(last + slope * step) for step in (1, 2, 3)]
    seasons = [r.get("season") or bangladesh_season(r.get("timestamp")) for r in usable]
    current_season = seasons[-1] if seasons else "unknown"
    same_season = sum(1 for x in seasons[:-1] if x == current_season)
    distinct_seasons = len(set(seasons) - {"unknown"})
    seasonal = "AVAILABLE" if same_season >= 3 and len(usable) >= 8 and distinct_seasons >= 2 else "INSUFFICIENT HISTORY"
    return {"persistence": persistence, "persistence_score": persistence_score,
            "consecutive_anomalies": consecutive, "repeated_anomalies": repeated,
            "trend": trend, "trend_slope": slope, "forecast": forecast,
            "current_season": current_season, "same_season_history_rows": same_season,
            "seasonal_baseline": seasonal}


def fit_temperature(scores: Sequence[float], labels: Sequence[int]) -> dict:
    """Fit one-parameter probability temperature using a small validation set."""
    s = np.clip(np.asarray(scores, dtype=float), 1e-5, 1 - 1e-5)
    y = np.asarray(labels, dtype=float)
    if s.size != y.size or s.size < 12 or not {0.0, 1.0}.issubset(set(y.tolist())):
        raise ValueError("Calibration needs at least 12 labelled rows containing both 0 and 1 outcomes.")
    logits = np.log(s / (1 - s))
    candidates = np.geomspace(0.25, 4.0, 121)
    losses = []
    for temp in candidates:
        p = 1.0 / (1.0 + np.exp(-logits / temp))
        loss = -np.mean(y * np.log(np.clip(p, 1e-8, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-8, 1)))
        losses.append(float(loss))
    idx = int(np.argmin(losses))
    return {"method": "temperature_scaling", "temperature": float(candidates[idx]),
            "validation_rows": int(s.size), "nll": float(losses[idx]),
            "fitted_at": datetime.now(timezone.utc).isoformat()}


def apply_calibration(score: float, calibration: Mapping | None):
    score = _clip(score, 1e-5, 1 - 1e-5)
    if not calibration or calibration.get("method") != "temperature_scaling":
        return score, "UNCALIBRATED"
    temp = max(0.05, _number(calibration.get("temperature"), 1.0))
    logit = math.log(score / (1.0 - score))
    return _clip(1.0 / (1.0 + math.exp(-logit / temp))), "TEMPERATURE_SCALED"


def resolution_feasibility(gsd_m: float | None, nominal_object_m=2.0, min_pixels_across=4.0,
                           mtf50=None, snr_db=None) -> dict:
    """Physics-aware resolvability screen using GSD plus optional MTF/SNR metadata.

    The score is deliberately a detectability *screen*, not a detection probability.
    If MTF/SNR metadata are unavailable, the result falls back to the conservative
    pixel-footprint rule used by earlier AEROSENTINEL versions.
    """
    if not gsd_m or gsd_m <= 0:
        return {"status": "UNKNOWN RESOLUTION", "physical_state":"NOT RESOLVABLE", "feasible": False, "pixels_across": None,
                "effective_pixels_across":None, "detectability_screening_score":0.0,
                "reason": "Ground sampling distance is unavailable."}
    pixels = float(nominal_object_m) / float(gsd_m)
    mtf_factor = 1.0
    snr_factor = 1.0
    if mtf50 is not None:
        mtf_factor = float(np.clip(_number(mtf50,0.0) / 0.25, 0.0, 1.0))
    if snr_db is not None:
        snr_factor = float(np.clip((_number(snr_db,0.0)-3.0)/17.0, 0.0, 1.0))
    information_factor = math.sqrt(max(0.0, mtf_factor * snr_factor))
    effective_pixels = pixels * information_factor
    feasible = effective_pixels >= float(min_pixels_across)
    score = _clip(effective_pixels / max(float(min_pixels_across),1e-9))
    return {"status": "FEASIBLE FOR CANDIDATE SCREENING" if feasible else "INSUFFICIENT RESOLUTION",
            "physical_state":"DETECTABLE_CANDIDATE_REGIME" if feasible else "NOT RESOLVABLE",
            "feasible": feasible, "pixels_across": pixels, "effective_pixels_across":effective_pixels,
            "detectability_screening_score":score, "gsd_m": float(gsd_m),
            "nominal_object_m": float(nominal_object_m), "minimum_pixels_across": float(min_pixels_across),
            "mtf50":None if mtf50 is None else float(mtf50), "snr_db":None if snr_db is None else float(snr_db),
            "reason": ("The nominal object retains enough effective source information for research candidate screening; identity still requires validation."
                       if feasible else "The nominal object occupies too little effective source information for a defensible direct-detection claim."),
            "note":"Heuristic physical resolvability screen, not a calibrated detection probability."}

def direct_path(multisatellite_run: Mapping | None, highres_evidence: Mapping | None = None) -> dict:
    scene_results = (multisatellite_run or {}).get("scene_results") or []
    gsd = [r.get("native_gsd_m") for r in scene_results if r.get("native_gsd_m")]
    best = min(map(float, gsd)) if gsd else None
    standard = resolution_feasibility(best)
    highres = highres_evidence or {}
    highres_feas = highres.get("feasibility") or {}
    if highres_feas.get("feasible"):
        candidate_count = int(highres.get("candidate_count") or 0)
        max_saliency = _clip(highres.get("max_candidate_saliency"))
        evidence_score = _clip(max_saliency * min(1.0, candidate_count / 3.0))
        return {"path": "DIRECT_SMALL_OBJECT_CANDIDATE", "status": "CANDIDATES FOR HUMAN REVIEW" if candidate_count else "NO CANDIDATE ABOVE SCREENING THRESHOLD",
                "available": True, "evidence_score": evidence_score, "candidate_count": candidate_count,
                "candidate_identity": "UNKNOWN", "high_resolution": highres,
                "standard_satellite_feasibility": standard,
                "interpretation": "Candidate saliency only. The system does not identify an enemy UAV or make a threat determination."}
    return {"path": "DIRECT_SMALL_OBJECT_CANDIDATE", "status": standard["status"], "available": False,
            "evidence_score": 0.0, "candidate_count": 0, "candidate_identity": "UNRESOLVED",
            "standard_satellite_feasibility": standard,
            "interpretation": "Direct small-object identification is unavailable at current resolution. Use the anomaly/temporal path instead."}


def _quality_reliability(current: Mapping, history_rows: int, foundation: Mapping | None,
                         calibration: Mapping | None):
    q = current.get("quality") or {}
    source = _clip(q.get("source_coverage"))
    alignment = _clip(q.get("temporal_alignment"))
    registration = q.get("registration_confidence")
    registration = alignment if registration is None else _clip(registration)
    valid = _clip(_number(q.get("mean_valid_percent"), 50.0) / 100.0)
    processing = q.get("mean_processing_readiness")
    processing = 0.65 if processing is None else _clip(processing)
    fusion_quality = q.get("fusion_source_reliability")
    fusion_quality = source if fusion_quality is None else _clip(fusion_quality)
    cloud = q.get("mean_optical_cloud_percent")
    cloud_quality = 0.7 if cloud is None else 1.0 - _clip(_number(cloud) / 100.0)
    history = _clip(history_rows / 12.0)
    weather = 1.0 if q.get("weather_available") else 0.5
    terrain = 1.0 if q.get("terrain_available") else 0.7
    foundation_bonus = 1.0 if foundation else 0.85
    calibration_bonus = 1.0 if calibration else 0.90
    reliability = _clip((
        0.16*source + 0.10*alignment + 0.12*registration + 0.10*valid +
        0.10*processing + 0.10*fusion_quality + 0.07*cloud_quality + 0.14*history +
        0.04*weather + 0.03*terrain + 0.02*foundation_bonus + 0.02*calibration_bonus
    ))
    source_count = int(q.get("source_count") or 0)
    if source_count <= 0:
        reliability = min(reliability, 0.20)
    elif source_count == 1:
        reliability = min(reliability, 0.40)
    if int(q.get("registration_ambiguous_pairs") or 0) > 0:
        reliability = min(reliability, 0.55)
    return reliability

def _status(score, abstain):
    if abstain:
        return "ABSTAIN / HUMAN REVIEW"
    if score < 0.25:
        return "NORMAL"
    if score < 0.50:
        return "MONITOR"
    if score < 0.75:
        return "ELEVATED"
    return "INVESTIGATE"


def run_reliability_stress_suite(current: Mapping, baseline: Sequence[Mapping], base_reliability: float):
    """Sensitivity audit for the thesis 'Trust' layer.

    This is a stress/sensitivity suite, not external validation. Each row states
    exactly what degradation is simulated.
    """
    feature = dict(current.get("features") or {})
    cases = []

    def case(name, modified, note):
        obs = dict(current); obs["features"] = modified
        score, ood, _z, immature = _score_against_baseline(obs, baseline)
        missing = sum(1 for k in FEATURES if k not in modified or modified.get(k) is None) / len(FEATURES)
        rel = _clip(base_reliability * (1 - 0.45*missing) * (1 - 0.25*ood))
        cases.append({"condition": name, "anomaly_score": score, "ood_score": ood,
                      "reliability": rel, "baseline_immature": immature, "simulation": note})

    cloud = dict(feature); cloud["optical_ndvi"] = None; cloud["source_coverage"] = _clip(feature.get("source_coverage", 0)-0.25)
    case("Cloud contamination / missing optical", cloud, "Remove optical NDVI and reduce source coverage.")
    sar = dict(feature); sar["wetness_screening"] = _clip(feature.get("wetness_screening",0)+0.25); sar["fusion_uncertainty"] = _clip(feature.get("fusion_uncertainty",0)+0.25)
    case("SAR noise sensitivity", sar, "Perturb wetness proxy and increase fusion uncertainty.")
    sensor = dict(feature); sensor["source_coverage"] = _clip(feature.get("source_coverage",0)-0.34); sensor["fusion_uncertainty"] = _clip(feature.get("fusion_uncertainty",0)+0.20)
    case("Sensor difference / one source lost", sensor, "Reduce source coverage and increase fusion uncertainty.")
    domain = {k: _clip(_number(v)+0.18 if k not in {"source_coverage","temporal_alignment"} else _number(v)-0.10) for k,v in feature.items()}
    case("Domain shift / unseen context", domain, "Shift contextual features and slightly reduce alignment/coverage.")
    season = dict(feature); season["optical_ndvi"] = _clip(feature.get("optical_ndvi",0)+0.20); season["rain_pressure"] = _clip(feature.get("rain_pressure",0)+0.20)
    case("Seasonal change sensitivity", season, "Shift vegetation and rainfall context together.")
    return cases


def latest_terramind_record(root: str | Path, target_time: str | None = None, max_age_hours=168) -> dict | None:
    root = Path(root)
    candidates = sorted((root / "data/foundation").rglob("*.terramind.json"), key=lambda p: p.stat().st_mtime, reverse=True) if (root/"data/foundation").exists() else []
    target = None
    if target_time:
        try:
            target = datetime.fromisoformat(str(target_time).replace("Z", "+00:00"))
            if target.tzinfo is None: target = target.replace(tzinfo=timezone.utc)
        except Exception:
            target = None
    for path in candidates[:20]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            modalities = {str(x).upper() for x in data.get("modalities") or []}
            if not {"S1GRD", "S2L2A", "DEM"}.issubset(modalities):
                continue
            if target and data.get("acquired_at"):
                acquired = datetime.fromisoformat(str(data["acquired_at"]).replace("Z", "+00:00"))
                if acquired.tzinfo is None: acquired=acquired.replace(tzinfo=timezone.utc)
                if abs((target - acquired).total_seconds()) > max_age_hours*3600:
                    continue
            data["_path"] = str(path)
            return data
        except Exception:
            continue
    return None


def run_trust_geo_x(observations: Iterable[Mapping], *, multisatellite_run: Mapping | None = None,
                    highres_evidence: Mapping | None = None, foundation_record: Mapping | None = None,
                    calibration: Mapping | None = None, abstain_threshold=0.45,
                    area_name="Selected area") -> dict:
    obs = list(observations)
    if not obs:
        raise ValueError("At least one observation is required.")
    obs.sort(key=lambda r: str(r.get("timestamp") or ""))
    current = obs[-1]
    baseline = obs[:-1]
    representation = unlabelled_representation(baseline, current, foundation_record)
    anomaly_baseline, baseline_strategy = _seasonal_baseline(current, baseline, minimum_same_season=3)
    score, ood, feature_z, immature = _score_against_baseline(current, anomaly_baseline)
    trajectory = anomaly_trajectory(obs)
    temporal = temporal_reasoning(trajectory)
    direct = direct_path(multisatellite_run, highres_evidence)
    reliability = _quality_reliability(current, len(baseline), representation.get("foundation"), calibration)

    # Uncertainty combines model/data limitations and out-of-distribution evidence.
    history_penalty = 1.0 - _clip(len(baseline)/12.0)
    direct_uncertainty = 0.65 if not direct.get("available") else 0.35
    data_uncertainty = _clip((current.get("features") or {}).get("fusion_uncertainty"))
    uncertainty = _clip(0.35*data_uncertainty + 0.25*ood + 0.20*history_penalty + 0.20*direct_uncertainty)

    persistence = _clip(temporal.get("persistence_score"))
    current_features = current.get("features") or {}
    change = _clip(current_features.get("temporal_change_screening"))
    direct_score = _clip(direct.get("evidence_score"))
    spatial_context = _clip(np.mean([
        _clip(current_features.get("airfield_environment_screening")),
        _clip(current_features.get("infrastructure_continuity_screening")),
        _clip(current_features.get("wetness_screening")),
    ]))
    anomaly_characteristics = _clip(0.70*score + 0.30*ood)
    # Hypothesis score only; direct candidates have intentionally low weight.
    raw_activity = _clip(0.47*score + 0.20*persistence + 0.13*change +
                         0.10*spatial_context + 0.05*ood + 0.05*direct_score)
    calibrated_activity, calibration_status = apply_calibration(raw_activity, calibration)
    confidence = _clip(reliability * (1.0 - uncertainty))
    abstain = bool(reliability < float(abstain_threshold) or (immature and len(baseline) < 3))
    status = _status(calibrated_activity, abstain)

    forecast = temporal.get("forecast") or [score, score, score]
    warning_forecast = [_clip(0.65*v + 0.20*persistence + 0.15*change) for v in forecast]
    stress = run_reliability_stress_suite(current, anomaly_baseline, reliability)

    top = sorted(feature_z.items(), key=lambda kv: kv[1], reverse=True)[:5]
    interpretation = {
        "hypothesis": "AOI-level activity/anomaly relevance for human review",
        "activity_relevance_screening": calibrated_activity,
        "status": status,
        "explanation": (
            "The score combines open-world anomaly evidence, temporal persistence, change trajectory and optional direct small-object candidate saliency. "
            "It does not identify an enemy UAV and is not a calibrated threat probability."
        ),
        "top_deviating_features": [{"feature": k, "robust_z": v} for k,v in top],
    }
    observed = dict(current.get("observed_evidence") or {})
    observed.update({
        "direct_path_status": direct.get("status"),
        "direct_candidate_count": direct.get("candidate_count"),
        "direct_candidate_identity": direct.get("candidate_identity"),
    })

    q = current.get("quality") or {}
    data_quality = {
        "cloud_detection": {"available": q.get("mean_optical_cloud_percent") is not None,
                            "mean_optical_cloud_percent": q.get("mean_optical_cloud_percent")},
        "noise_filtering": "SAR log-amplitude 3x3 local mean screening + optical QA masking",
        "co_registration": ("Common local UTM grid + residual phase-correlation quality gate"
                            if (multisatellite_run or {}).get("fusion") else "No fused grid available"),
        "missing_data_fraction": 1.0 - _clip(q.get("source_coverage")),
        "multimodal_alignment_score": _clip(q.get("temporal_alignment")),
        "acquisition_skew_hours": q.get("acquisition_skew_hours"),
        "valid_pixel_percent": q.get("mean_valid_percent"),
        "processing_readiness": q.get("mean_processing_readiness"),
        "registration_quality": {
            "confidence": q.get("registration_confidence"),
            "max_uncertainty_px": q.get("registration_max_uncertainty_px"),
            "ambiguous_pairs": q.get("registration_ambiguous_pairs"),
        },
        "fusion_source_reliability": q.get("fusion_source_reliability"),
        "effective_source_count": q.get("effective_source_count"),
        "source_reliability_weights": q.get("source_reliability_weights") or {},
        "counterfactual_source_contribution": q.get("counterfactual_source_contribution") or {},
    }
    activity_assessment = {
        "evidence_fusion": score,
        "spatial_context": spatial_context,
        "temporal_context": persistence,
        "anomaly_characteristics": anomaly_characteristics,
        "direct_candidate_evidence": direct_score,
        "uav_activity_relevance_hypothesis": calibrated_activity,
        "interpretation_boundary": "AOI-level research hypothesis only; no hostile/benign or enemy-UAV determination.",
    }

    result = {
        "version": "AEROSENTINEL v2.8.0 thesis research layer",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "area_name": area_name,
        "mode": "satellite-only open-world early-warning research; human-in-the-loop",
        "architecture": {
            "observe": True,
            "data_quality_and_fusion": True,
            "self_supervised_or_foundation_representation": True,
            "direct_candidate_path": True,
            "open_world_anomaly_path": True,
            "temporal_reasoning": True,
            "activity_assessment": True,
            "uncertainty_engine": True,
            "predictive_early_warning": True,
            "decision_map": True,
            "human_review": True,
        },
        "data_quality_and_fusion": data_quality,
        "representation": representation,
        "direct_path": direct,
        "anomaly_path": {
            "current_anomaly_score": score,
            "ood_score": ood,
            "baseline_rows": len(anomaly_baseline),
            "total_history_rows": len(baseline),
            "baseline_strategy": baseline_strategy,
            "baseline_immature": immature,
            "feature_deviation": feature_z,
            "trajectory": trajectory,
        },
        "temporal_reasoning": temporal,
        "uav_activity_assessment": activity_assessment,
        "observed_evidence": observed,
        "interpretation_hypothesis": interpretation,
        "uncertainty_engine": {
            "confidence": confidence,
            "uncertainty": uncertainty,
            "reliability": reliability,
            "ood_score": ood,
            "calibration": calibration_status,
            "abstain": abstain,
            "abstain_threshold": float(abstain_threshold),
        },
        "predictive_early_warning": {
            "score": calibrated_activity,
            "risk_score": calibrated_activity,
            "raw_score": raw_activity,
            "status": status,
            "trend": temporal.get("trend"),
            "persistence": temporal.get("persistence"),
            "forecast_next_3_observations": warning_forecast,
            "recommendation": "HUMAN REVIEW REQUIRED" if status in {"ELEVATED", "INVESTIGATE", "ABSTAIN / HUMAN REVIEW"} else "CONTINUE MONITORING",
        },
        "reliability_stress_suite": stress,
        "limitations": [
            "Small UAVs are generally not directly resolvable in Sentinel/Landsat imagery; direct screening is gated by source GSD.",
            "Open-world anomaly scores indicate deviation from the learned local baseline, not cause or intent.",
            "Activity relevance is a research hypothesis score, not a threat probability or hostile-actor classification.",
            "Calibration is reported as unavailable unless labelled validation data are explicitly supplied.",
            "Seasonal anomaly baselines are used only when enough same-season history exists; otherwise all-history fallback is explicitly reported.",
            "Residual registration is a translation-quality screen, not full bundle adjustment or sensor-geometry correction; ambiguous pairs reduce reliability.",
            "Reliability-aware fusion reduces weak-source dominance but does not prove that multimodal fusion outperforms the best single sensor on every task.",
            "Human analyst review and independent local/official evidence are required before any real-world interpretation.",
        ],
    }
    sig_body = {"obs": current.get("signature"), "score": result["predictive_early_warning"]["score"],
                "status": status, "direct": direct.get("status"), "direct_count": direct.get("candidate_count"),
                "direct_score": direct.get("evidence_score"), "foundation": representation.get("backend")}
    result["signature"] = hashlib.sha256(json.dumps(sig_body, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return result


def resolution_research_table():
    sensors = [
        ("Sentinel-2 optical", 10.0),
        ("Sentinel-1 GRD screening grid (typical order)", 10.0),
        ("Landsat 8/9 optical", 30.0),
        ("Example commercial/open high-resolution 1.0 m", 1.0),
        ("Example commercial/open high-resolution 0.5 m", 0.5),
        ("Example commercial/open high-resolution 0.3 m", 0.3),
    ]
    rows = []
    for sensor, gsd in sensors:
        for size in (1.0, 2.0, 3.0, 5.0):
            f = resolution_feasibility(gsd, nominal_object_m=size)
            rows.append({"sensor": sensor, "gsd_m": gsd, "nominal_object_m": size,
                         "pixels_across": round(f["pixels_across"], 2), "direct_candidate_feasible": f["feasible"]})
    return rows


def domain_reliability_evaluation(rows: Sequence[Mapping], abstain_threshold=0.45) -> dict:
    """Evaluate labelled research cases across region/season/condition groups.

    Required fields: score, label. Optional fields: reliability, region, season,
    condition. This is intended for externally reviewed validation cases, not for
    generating labels from AEROSENTINEL itself.
    """
    clean=[]
    for row in rows:
        try:
            score=float(row.get('score')); label=int(row.get('label'))
            reliability=float(row.get('reliability',1.0))
        except (TypeError,ValueError):
            continue
        if label not in (0,1) or not math.isfinite(score) or not math.isfinite(reliability):
            continue
        clean.append({'score':_clip(score),'label':label,'reliability':_clip(reliability),
                      'region':str(row.get('region') or 'unspecified'),
                      'season':str(row.get('season') or 'unspecified'),
                      'condition':str(row.get('condition') or 'normal')})
    if len(clean)<4:
        raise ValueError('Domain reliability evaluation needs at least 4 valid labelled rows.')

    def metrics(sub):
        y=np.array([r['label'] for r in sub],dtype=float)
        p=np.array([r['score'] for r in sub],dtype=float)
        rel=np.array([r['reliability'] for r in sub],dtype=float)
        report=rel>=float(abstain_threshold)
        coverage=float(report.mean())
        if report.any():
            yy=y[report];pp=p[report]
            acc=float(np.mean((pp>=.5)==yy));brier=float(np.mean((pp-yy)**2))
            bins=np.linspace(0,1,6);ece=0.0
            for lo,hi in zip(bins[:-1],bins[1:]):
                mask=(pp>=lo)&((pp<hi) if hi<1 else (pp<=hi))
                if mask.any():ece+=float(mask.mean())*abs(float(pp[mask].mean())-float(yy[mask].mean()))
        else:
            acc=brier=ece=None
        return {'rows':len(sub),'reported_coverage':coverage,'abstention_rate':1-coverage,
                'accuracy_at_0_5':acc,'brier_score':brier,'ece_5_bins':ece,
                'mean_reliability':float(rel.mean()),'positive_rate':float(y.mean())}

    grouped=[]
    for field in ('region','season','condition'):
        for value in sorted({r[field] for r in clean}):
            sub=[r for r in clean if r[field]==value]
            grouped.append({'group_type':field,'group':value,**metrics(sub)})
    return {'overall':metrics(clean),'groups':grouped,'abstain_threshold':float(abstain_threshold),
            'note':'Metrics require independent human-reviewed labels. Group gaps reveal geographic/seasonal/condition reliability differences.'}


def domain_evaluation_template_csv() -> str:
    return ("region,season,condition,score,label,reliability\n"
            "Bangladesh-site-A,monsoon,normal,0.18,0,0.82\n"
            "Bangladesh-site-A,monsoon,cloud,0.41,0,0.55\n"
            "Bangladesh-site-B,dry,unseen_anomaly,0.78,1,0.69\n"
            "Bangladesh-site-B,dry,sensor_shift,0.52,1,0.44\n")

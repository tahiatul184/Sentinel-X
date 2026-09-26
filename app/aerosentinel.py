"""AEROSENTINEL v2.9.2 satellite-only aviation research layer.

AEROSENTINEL extends the validated AEROSENTINEL research core into a broader
aviation-safety framework. It preserves the conservative human-in-the-loop
boundary while adding explicit cross-sensor robustness, rare-event screening,
spatio-temporal hazard forecasting, explainability and deployment-readiness
instrumentation.

All scores in this module are research screening indicators. They are not
certified aviation minima, flight-clearance decisions or calibrated accident
probabilities unless a separate validation study establishes that claim.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from provenance import build_provenance_manifest
from compute_policy import choose_processing_tier

from trust_geo_x import (
    apply_calibration,
    bangladesh_season,
    build_observation,
    domain_evaluation_template_csv,
    domain_reliability_evaluation,
    fit_temperature,
    latest_terramind_record,
    resolution_research_table,
    run_trust_geo_x,
)

VERSION = "2.9.2"
FRAMEWORK = "AEROSENTINEL"
TITLE = (
    "AEROSENTINEL: A Trustworthy Multimodal Spatio-Temporal AI Framework Using "
    "Satellite Foundation Models for Aviation Safety, Airfield Change Detection, "
    "Hazard Prediction, and Operational Decision Support"
)


def _clip(value, lo=0.0, hi=1.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return float(np.clip(x, lo, hi))


def _mean(values):
    vals = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            vals.append(x)
    return float(np.mean(vals)) if vals else 0.0


def _status(score: float, reliability: float, abstain_threshold: float) -> str:
    if reliability < abstain_threshold:
        return "ABSTAIN / HUMAN REVIEW"
    if score < 0.25:
        return "NORMAL"
    if score < 0.50:
        return "MONITOR"
    if score < 0.75:
        return "ELEVATED"
    return "HIGH REVIEW"


def _file_bytes(paths: Sequence[str | None]) -> int:
    total = 0
    seen = set()
    for raw in paths:
        if not raw:
            continue
        try:
            path = Path(str(raw)).resolve()
        except Exception:
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            if path.is_file():
                total += int(path.stat().st_size)
        except OSError:
            continue
    return total


def rare_event_screen(legacy: Mapping) -> dict:
    anomaly = legacy.get("anomaly_path") or {}
    deviations = anomaly.get("feature_deviation") or {}
    values = [abs(float(v)) for v in deviations.values() if isinstance(v, (int, float)) and math.isfinite(float(v))]
    max_z = max(values) if values else 0.0
    extreme_count = sum(v >= 4.0 for v in values)
    severe_count = sum(v >= 6.0 for v in values)
    tail_score = _clip(1.0 - math.exp(-max(0.0, max_z - 2.0) / 2.5))
    anomaly_score = _clip(anomaly.get("current_anomaly_score"))
    ood = _clip(anomaly.get("ood_score"))
    rows = int(anomaly.get("baseline_rows") or 0)
    score = _clip(0.45 * ood + 0.30 * tail_score + 0.25 * anomaly_score)
    if rows < 4:
        state = "INSUFFICIENT BASELINE"
    elif score >= 0.70 or severe_count:
        state = "RARE-EVENT CANDIDATE"
    elif score >= 0.45 or extreme_count:
        state = "UNUSUAL / REVIEW"
    else:
        state = "WITHIN LEARNED RANGE"
    top = sorted(((str(k), abs(float(v))) for k, v in deviations.items()
                  if isinstance(v, (int, float)) and math.isfinite(float(v))),
                 key=lambda kv: kv[1], reverse=True)[:5]
    return {
        "score": score,
        "state": state,
        "maximum_robust_z": max_z,
        "features_above_4sigma": int(extreme_count),
        "features_above_6sigma": int(severe_count),
        "baseline_rows": rows,
        "top_tail_features": [{"feature": k, "robust_z": v} for k, v in top],
        "method": "Robust-tail + OOD + open-world anomaly screening",
        "interpretation": "Rare-event candidate means statistically unusual relative to the local learned baseline; it does not establish cause, severity or operational consequence.",
    }


def cross_sensor_generalization(multisatellite_run: Mapping | None, legacy: Mapping) -> dict:
    run = multisatellite_run or {}
    fusion = run.get("fusion") or {}
    requested = list(fusion.get("requested_satellites") or run.get("requested_satellites") or [])
    available = list(fusion.get("available_satellites") or [])
    source_fraction = len(available) / max(1, len(requested))
    alignment = _clip(fusion.get("temporal_alignment_score"))
    fusion_uncertainty = _clip(fusion.get("mean_fusion_uncertainty"), 0, 1)
    coverage = _clip(fusion.get("mean_source_coverage"), 0, 1)
    stress = legacy.get("reliability_stress_suite") or []
    source_loss = [r for r in stress if "sensor" in str(r.get("condition", "")).lower() or "source" in str(r.get("condition", "")).lower()]
    base_score = _clip((legacy.get("anomaly_path") or {}).get("current_anomaly_score"))
    deltas = [abs(_clip(r.get("anomaly_score")) - base_score) for r in source_loss]
    source_loss_sensitivity = max(deltas) if deltas else None
    score = _clip(0.30 * source_fraction + 0.25 * coverage + 0.20 * alignment +
                  0.15 * (1.0 - fusion_uncertainty) + 0.10 * (1.0 - _clip(source_loss_sensitivity or 0.0)))
    if len(requested) < 2:
        state = "NOT TESTED"
    elif len(available) < 2:
        state = "LIMITED / SINGLE-SOURCE"
    elif score >= 0.70:
        state = "MULTISENSOR READY FOR VALIDATION"
    else:
        state = "MULTISENSOR REVIEW REQUIRED"
    return {
        "requested_sensors": requested,
        "available_sensors": available,
        "source_fraction": source_fraction,
        "source_coverage": coverage,
        "temporal_alignment": alignment,
        "fusion_uncertainty": fusion_uncertainty,
        "source_loss_sensitivity": source_loss_sensitivity,
        "generalization_readiness_score": score,
        "state": state,
        "note": "This is a cross-sensor robustness screen. True generalization requires independent labelled evaluation on different sensors, regions, seasons and acquisition conditions.",
    }


def aviation_hazard_assessment(legacy: Mapping, rare_event: Mapping, assessment: Mapping | None,
                               abstain_threshold: float) -> dict:
    features = ((legacy.get("anomaly_path") or {}).get("feature_deviation") or {})
    obs = legacy.get("observed_evidence") or {}
    temporal = legacy.get("temporal_reasoning") or {}
    anomaly = legacy.get("anomaly_path") or {}
    uncertainty = legacy.get("uncertainty_engine") or {}
    scores = (assessment or {}).get("scores") or {}

    anomaly_score = _clip(anomaly.get("current_anomaly_score"))
    ood = _clip(anomaly.get("ood_score"))
    persistence = _clip(temporal.get("persistence_score"))
    change = _clip(obs.get("temporal_change_screening"))
    wetness = _clip(obs.get("wetness_screening"))
    airfield = _clip(scores.get("airfield_environment_screening"))
    weather = _clip(scores.get("weather_environment_screening"))
    surface = _clip(scores.get("airfield_surface_screening"))
    infrastructure_continuity = _clip(scores.get("infrastructure_continuity_screening"), 0, 1)
    infrastructure_risk = 1.0 - infrastructure_continuity
    rare = _clip(rare_event.get("score"))
    reliability = _clip(uncertainty.get("reliability"))

    airfield_change = _clip(0.45 * change + 0.35 * anomaly_score + 0.20 * ood)
    surface_hazard = _clip(0.55 * surface + 0.25 * wetness + 0.20 * change)
    met_hazard = weather
    infrastructure_hazard = _clip(0.60 * infrastructure_risk + 0.25 * change + 0.15 * anomaly_score)
    spatiotemporal_hazard = _clip(0.55 * anomaly_score + 0.25 * persistence + 0.20 * change)

    weights = {
        "airfield_change": 0.20,
        "surface_condition": 0.18,
        "weather_environment": 0.18,
        "infrastructure_continuity": 0.12,
        "spatio_temporal_anomaly": 0.17,
        "rare_event": 0.08,
        "out_of_distribution": 0.07,
    }
    domains = {
        "airfield_change": airfield_change,
        "surface_condition": surface_hazard,
        "weather_environment": met_hazard,
        "infrastructure_continuity": infrastructure_hazard,
        "spatio_temporal_anomaly": spatiotemporal_hazard,
        "rare_event": rare,
        "out_of_distribution": ood,
    }
    contributions = {k: weights[k] * domains[k] for k in weights}
    risk = _clip(sum(contributions.values()))
    state = _status(risk, reliability, abstain_threshold)

    top = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
    explainability = {
        "method": "Transparent weighted evidence decomposition",
        "top_risk_drivers": [
            {"driver": key, "screening_value": domains[key], "weight": weights[key],
             "weighted_contribution": value}
            for key, value in top[:5]
        ],
        "top_deviating_features": sorted(
            ({"feature": str(k), "robust_z": float(v)} for k, v in features.items()
             if isinstance(v, (int, float)) and math.isfinite(float(v))),
            key=lambda r: abs(r["robust_z"]), reverse=True)[:5],
        "explanation": "The operational aviation-risk screen is decomposed into visible domain scores and weighted contributions so a reviewer can see why the score changed.",
    }
    return {
        "risk_score": risk,
        "status": state,
        "reliability": reliability,
        "domains": domains,
        "weights": weights,
        "explainability": explainability,
        "recommendation": "HUMAN REVIEW REQUIRED" if state != "NORMAL" else "CONTINUE MONITORING",
        "boundary": "Research decision support only. This score does not provide flight clearance, runway certification, dispatch authority or a calibrated accident probability.",
    }


def hazard_forecast(legacy: Mapping, aviation_risk: Mapping) -> dict:
    temporal = legacy.get("temporal_reasoning") or {}
    uncertainty = legacy.get("uncertainty_engine") or {}
    anomaly_forecast = (legacy.get("predictive_early_warning") or {}).get("forecast_next_3_observations") or []
    current_risk = _clip(aviation_risk.get("risk_score"))
    current_anomaly = _clip((legacy.get("predictive_early_warning") or {}).get("score"))
    delta = current_risk - current_anomaly
    points = [_clip(float(v) + 0.65 * delta) for v in anomaly_forecast[:3]]
    while len(points) < 3:
        points.append(current_risk)
    width = 0.08 + 0.22 * _clip(uncertainty.get("uncertainty"))
    intervals = [{"step": i + 1, "screening": p, "lower": _clip(p - width), "upper": _clip(p + width)}
                 for i, p in enumerate(points)]
    return {
        "trend": temporal.get("trend"),
        "trend_slope": temporal.get("trend_slope"),
        "forecast_next_3_observations": intervals,
        "seasonal_baseline": temporal.get("seasonal_baseline"),
        "method": "Transparent short-horizon spatio-temporal screening projection",
        "note": "Forecast intervals are uncertainty bands for research screening, not calibrated meteorological or operational hazard probabilities.",
    }


def deployment_readiness(multisatellite_run: Mapping | None, *, assessment_latency_ms: float,
                         deployment_context: Mapping | None = None) -> dict:
    run = multisatellite_run or {}
    context = deployment_context or {}
    scene_results = run.get("scene_results") or []
    scene_times = [float(r.get("elapsed_seconds")) for r in scene_results
                   if isinstance(r.get("elapsed_seconds"), (int, float)) and math.isfinite(float(r.get("elapsed_seconds")))]
    paths = []
    fusion = run.get("fusion") or {}
    paths.extend([fusion.get("fusion_path"), fusion.get("preview_path")])
    for row in scene_results:
        paths.extend([row.get("stack_path"), row.get("preview_path")])
    local_bytes = _file_bytes(paths)
    interval_min = max(1.0, float(context.get("collection_interval_minutes") or 30.0))
    output_rate = local_bytes * (60.0 / interval_min)
    sat_elapsed = run.get("elapsed_seconds")
    if isinstance(sat_elapsed, (int, float)) and math.isfinite(float(sat_elapsed)):
        satellite_latency = float(sat_elapsed)
    else:
        satellite_latency = None
    readiness_components = [
        1.0 if satellite_latency is not None else 0.4,
        1.0 if local_bytes > 0 else 0.5,
        1.0 if assessment_latency_ms < 2000 else 0.7 if assessment_latency_ms < 5000 else 0.4,
    ]
    readiness = _clip(float(np.mean(readiness_components)))
    return {
        "assessment_compute_latency_ms": float(assessment_latency_ms),
        "satellite_pipeline_elapsed_seconds": satellite_latency,
        "mean_scene_processing_seconds": _mean(scene_times) if scene_times else None,
        "max_scene_processing_seconds": max(scene_times) if scene_times else None,
        "configured_scene_workers": ((run.get("concurrency") or {}).get("configured_workers")),
        "peak_scene_workers": ((run.get("concurrency") or {}).get("peak_scene_workers")),
        "local_analytical_output_bytes": int(local_bytes),
        "local_analytical_output_megabytes": float(local_bytes / (1024 ** 2)),
        "scheduled_collection_interval_minutes": interval_min,
        "output_storage_rate_bytes_per_hour": float(output_rate),
        "deployment_readiness_score": readiness,
        "real_time_statement": "Satellite observations are not true real-time because acquisition/revisit is externally constrained. Aircraft observations are limited to satellite acquisition times.",
        "bandwidth_statement": "Remote source-transfer bytes are provider/COG dependent and are not fully instrumented in this build; AEROSENTINEL reports the measurable local analytical data footprint instead.",
        "compute_cost_statement": "Processing elapsed time and worker concurrency are reported as transparent computational-cost proxies; energy use and hardware-normalized FLOPs are not yet instrumented.",
    }


def run_aerosentinel(observations, *, multisatellite_run: Mapping | None = None,
                     highres_evidence: Mapping | None = None, foundation_record: Mapping | None = None,
                     calibration: Mapping | None = None, abstain_threshold=0.45,
                     area_name="Selected area", assessment: Mapping | None = None,
                     deployment_context: Mapping | None = None) -> dict:
    started = time.perf_counter()
    observations = list(observations)
    legacy = run_trust_geo_x(
        observations,
        multisatellite_run=multisatellite_run,
        highres_evidence=highres_evidence,
        foundation_record=foundation_record,
        calibration=calibration,
        abstain_threshold=abstain_threshold,
        area_name=area_name,
    )
    rare = rare_event_screen(legacy)
    cross_sensor = cross_sensor_generalization(multisatellite_run, legacy)
    aviation = aviation_hazard_assessment(legacy, rare, assessment, float(abstain_threshold))
    forecast = hazard_forecast(legacy, aviation)
    latency_ms = (time.perf_counter() - started) * 1000.0
    deployment = deployment_readiness(
        multisatellite_run,
        assessment_latency_ms=latency_ms,
        deployment_context=deployment_context,
    )

    # Preserve validated lower-level records for backward compatibility, but make
    # the user-facing interpretation explicitly aviation-safety oriented.
    legacy["version"] = f"AEROSENTINEL v{VERSION} aviation-safety research layer"
    legacy["framework"] = FRAMEWORK
    legacy["title"] = TITLE
    legacy["mode"] = "multimodal spatio-temporal aviation-safety decision-support research; human-in-the-loop"
    legacy["architecture"].update({
        "cross_sensor_generalization": True,
        "rare_event_detection": True,
        "explainable_decision_support": True,
        "operational_aviation_risk": True,
        "deployment_readiness": True,
        "registration_quality_gate": True,
        "reliability_aware_fusion": True,
        "evidence_provenance": True,
        "adaptive_compute_policy": True,
    })
    legacy["rare_event_detection"] = rare
    legacy["cross_sensor_generalization"] = cross_sensor
    legacy["operational_aviation_risk"] = aviation
    legacy["hazard_prediction"] = forecast
    legacy["explainability"] = aviation["explainability"]
    legacy["deployment_readiness"] = deployment
    legacy["provenance"] = build_provenance_manifest(
        multisatellite_run=multisatellite_run, current_observation=(observations[-1] if observations else None),
        foundation_record=foundation_record, highres_evidence=highres_evidence,
    )
    legacy["aviation_safety_assessment"] = {
        "airfield_change_detection": aviation["domains"]["airfield_change"],
        "surface_condition_screening": aviation["domains"]["surface_condition"],
        "weather_environment_screening": aviation["domains"]["weather_environment"],
        "infrastructure_continuity_risk": aviation["domains"]["infrastructure_continuity"],
        "spatio_temporal_anomaly": aviation["domains"]["spatio_temporal_anomaly"],
        "rare_event_score": rare["score"],
        "operational_risk_screening": aviation["risk_score"],
        "status": aviation["status"],
        "interpretation_boundary": aviation["boundary"],
    }
    legacy["interpretation_hypothesis"] = {
        "hypothesis": "AOI-level aviation-safety and airfield-change relevance for human review",
        "aviation_risk_screening": aviation["risk_score"],
        "status": aviation["status"],
        "explanation": (
            "AEROSENTINEL combines multimodal satellite change evidence, weather and terrain context, "
            "open-world anomalies, temporal persistence, rare-event evidence and uncertainty. "
            "It supports human aviation-safety review and does not issue flight clearance or autonomous operational decisions."
        ),
        "top_risk_drivers": aviation["explainability"]["top_risk_drivers"],
    }
    legacy["explainability"]["counterfactual_source_contribution"] = (
        ((multisatellite_run or {}).get("fusion") or {}).get("counterfactual_source_contribution") or {}
    )
    legacy["predictive_early_warning"] = {
        **(legacy.get("predictive_early_warning") or {}),
        "score": aviation["risk_score"],
        "risk_score": aviation["risk_score"],
        "status": aviation["status"],
        "forecast_next_3_observations": [r["screening"] for r in forecast["forecast_next_3_observations"]],
        "forecast_with_uncertainty": forecast["forecast_next_3_observations"],
        "recommendation": aviation["recommendation"],
        "screening_semantics": "aviation-safety operational risk screening, not a certified probability",
    }
    legacy["adaptive_compute_policy"] = choose_processing_tier(legacy, deployment)
    legacy["limitations"] = [
        "Satellite acquisition is constrained by orbital revisit, cloud and provider availability; it is not continuous real-time sensing.",
        "Airfield-change and hazard scores are research screening indicators and require independent local validation.",
        "Public weather context does not replace certified aviation meteorological products or operational procedures.",
        "Cross-sensor readiness scores are sensitivity indicators; genuine generalization must be demonstrated on independent regions, seasons and sensors.",
        "Residual phase-correlation alignment only estimates translation; strong relief, parallax or nonrigid distortions still require higher-grade registration/orthorectification.",
        "Reliability-aware fusion and source ablation reduce negative-transfer risk but do not guarantee that combining sensors is better than the strongest single-sensor model.",
        "Rare-event detection identifies statistical novelty relative to the learned local baseline; it does not determine cause or severity.",
        "Forecasts are transparent short-horizon projections and are not calibrated hazard probabilities unless externally validated.",
        "Deployment cost metrics report measurable processing latency, worker concurrency and local data footprint; source transfer bytes and energy consumption are not fully instrumented.",
        "Human review remains mandatory for consequential aviation interpretation; the system does not provide flight clearance, dispatch authority or autonomous safety-critical decisions.",
    ]
    # Signature includes the new aviation-safety score so old cached AEROSENTINEL
    # results cannot masquerade as an AEROSENTINEL assessment.
    import hashlib, json
    sig = {
        "legacy": legacy.get("signature"),
        "aviation_risk": aviation["risk_score"],
        "rare_event": rare["score"],
        "cross_sensor": cross_sensor["generalization_readiness_score"],
        "version": VERSION,
    }
    legacy["signature"] = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:24]
    legacy["generated_at"] = datetime.now(timezone.utc).isoformat()
    return legacy


__all__ = [
    "FRAMEWORK", "TITLE", "VERSION", "apply_calibration", "bangladesh_season",
    "build_observation", "domain_evaluation_template_csv", "domain_reliability_evaluation",
    "fit_temperature", "latest_terramind_record", "resolution_research_table",
    "run_aerosentinel", "rare_event_screen", "cross_sensor_generalization",
    "aviation_hazard_assessment", "hazard_forecast", "deployment_readiness",
]

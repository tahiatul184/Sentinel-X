"""Automated aviation resilience screening for AEROSENTINEL v2.9.0.

This module intentionally limits itself to airfield/environment safety,
disaster response, infrastructure continuity and humanitarian-support context.
It does not perform targeting, adversary tracking, weapons employment or flight
release/clearance decisions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def _clip(value, lo=0.0, hi=1.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return float(np.clip(x, lo, hi))


def _n(value, scale, offset=0.0):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return float(np.clip((x - offset) / max(float(scale), 1e-9), 0, 1))


def _level(score, reliability):
    if reliability < 0.35:
        return "INSUFFICIENT DATA"
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "ELEVATED"
    return "LOW"


def _mean_temporal_change(fusion):
    rows = list((fusion.get("temporal_change") or {}).values())
    vals = []
    for row in rows:
        try:
            v = float(row.get("mean_absolute_screening_change"))
        except (TypeError, ValueError, AttributeError):
            continue
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


def _weather_age_score(weather):
    raw = weather.get("fetched_at") if isinstance(weather, Mapping) else None
    if not raw:
        return 0.0
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_h = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds()/3600)
    except ValueError:
        return 0.0
    return float(np.exp(-age_h/6.0))


def compute_resilience_assessment(multisatellite_run: Mapping | None,
                                  weather: Mapping | None,
                                  assets: Sequence[Mapping] = (),
                                  area_name: str = "Selected area",
                                  profile: str = "airfield_resilience",
                                  terrain: Mapping | None = None):
    run = multisatellite_run or {}
    fusion = run.get("fusion") or {}
    wsum = (weather or {}).get("summary") or {}
    tsum = (terrain or {}).get("summary") or {}

    wet = _clip(fusion.get("mean_wetness_screening"))
    if wet is None:
        wet = 0.5
    uncertainty = _clip(fusion.get("mean_fusion_uncertainty"))
    if uncertainty is None:
        uncertainty = 1.0
    coverage = _clip(fusion.get("mean_source_coverage"))
    if coverage is None:
        coverage = 0.0
    alignment = _clip(fusion.get("temporal_alignment_score"))
    if alignment is None:
        alignment = 0.0
    change = _clip(_mean_temporal_change(fusion)) or 0.0

    rain6 = float(wsum.get("rain_next_6h_mm") or 0.0)
    rain24 = float(wsum.get("rain_next_24h_mm") or 0.0)
    wind = max(float(wsum.get("max_wind_next_6h_kmh") or 0.0), float(wsum.get("current_wind_kmh") or 0.0))
    gust = max(float(wsum.get("max_gust_next_6h_kmh") or 0.0), float(wsum.get("current_gust_kmh") or 0.0))
    visibility = wsum.get("min_visibility_next_6h_m")
    if visibility is None:
        visibility = wsum.get("current_visibility_m")
    thunder_hours = int(wsum.get("thunderstorm_hours_next_24h") or 0)
    cape = float(wsum.get("max_cape_next_24h_jkg") or 0.0)
    max_temp = float(wsum.get("max_temperature_next_24h_c") or wsum.get("temperature_c") or 0.0)

    # These are transparent research scalings, not aviation operating limits.
    rain_pressure = max(_n(rain6, 40.0), _n(rain24, 120.0))
    wind_pressure = max(_n(wind, 45.0), _n(gust, 70.0))
    visibility_pressure = 0.0 if visibility is None else float(np.clip(1.0 - float(visibility)/10000.0, 0, 1))
    convective_pressure = max(_n(thunder_hours, 3.0), _n(cape, 1800.0))
    heat_pressure = _n(max_temp, 12.0, 32.0)
    slope_pressure = _n(tsum.get("p90_slope_deg"), 15.0)
    relief_pressure = _n(tsum.get("relief_p05_p95_m"), 120.0)
    terrain_pressure = float(np.clip(0.65*slope_pressure + 0.35*relief_pressure, 0, 1))

    weather_freshness = _weather_age_score(weather or {})
    satellite_present = bool(fusion)
    weather_present = bool(wsum)
    reliability = float(np.clip(
        0.30 * coverage + 0.22 * (1.0 - uncertainty) + 0.18 * alignment +
        0.18 * weather_freshness + 0.06 * float(satellite_present) + 0.06 * float(weather_present),
        0, 1
    ))

    surface_screening = float(np.clip(0.48*wet + 0.24*rain_pressure + 0.14*change + 0.14*uncertainty, 0, 1))
    weather_screening = float(np.clip(0.31*wind_pressure + 0.24*visibility_pressure + 0.22*convective_pressure +
                                      0.15*rain_pressure + 0.08*heat_pressure, 0, 1))
    airfield_environment = float(np.clip(0.56*surface_screening + 0.44*weather_screening, 0, 1))
    disaster_response = float(np.clip(0.31*wet + 0.28*rain_pressure + 0.15*wind_pressure + 0.10*change + 0.08*terrain_pressure + 0.08*uncertainty, 0, 1))
    infrastructure_risk = float(np.clip(0.27*wet + 0.21*rain_pressure + 0.16*wind_pressure + 0.11*change +
                                        0.10*terrain_pressure + 0.08*heat_pressure + 0.07*uncertainty, 0, 1))
    infrastructure_continuity = float(1.0 - infrastructure_risk)
    monsoon_flood = float(np.clip(0.45*wet + 0.35*rain_pressure + 0.20*change, 0, 1))
    high_wind_storm = float(np.clip(0.40*wind_pressure + 0.30*convective_pressure + 0.20*rain_pressure + 0.10*visibility_pressure, 0, 1))
    humanitarian_access = float(np.clip(0.45*disaster_response + 0.25*surface_screening + 0.20*weather_screening + 0.10*uncertainty, 0, 1))

    alerts = []
    def add(category, score, message, action):
        if score >= 0.40:
            alerts.append({"category": category, "level": "HIGH" if score >= 0.70 else "ELEVATED",
                           "score": round(float(score), 3), "message": message, "suggested_check": action})
    add("Surface water / drainage", surface_screening,
        "Satellite and rainfall context indicate elevated surface-water or drainage concern.",
        "Inspect low-lying pavement, drainage, culverts and standing-water reports before relying on the area.")
    add("Weather environment", weather_screening,
        "Wind, visibility, convection or rainfall context is elevated in the public forecast.",
        "Review official aviation weather and local procedures; verify actual field conditions.")
    add("Disaster response", disaster_response,
        "Flood/rain/wind context may increase humanitarian-response and access challenges.",
        "Check access routes, medical/relief staging areas, communications and local emergency coordination.")
    if reliability < 0.45:
        alerts.insert(0, {"category": "Data confidence", "level": "REVIEW", "score": round(reliability, 3),
                          "message": "The automated screen has limited source coverage, stale context or high uncertainty.",
                          "suggested_check": "Obtain fresh local observations and official weather before interpreting the screen."})
    priority = {"airfield_resilience": "Surface water / drainage", "disaster_response": "Disaster response", "infrastructure_resilience": "Data confidence"}.get(profile)
    if priority:
        alerts.sort(key=lambda a: 0 if a.get("category") == priority else 1)

    if not alerts:
        alerts.append({"category": "Automated screen", "level": "LOW", "score": round(airfield_environment, 3),
                       "message": "No elevated research-screening condition was identified from the currently available inputs.",
                       "suggested_check": "Continue routine monitoring and verify local/official sources."})

    asset_rows = []
    hazard = max(surface_screening, weather_screening, disaster_response)
    for rec in assets:
        payload = rec.get("payload", rec) if isinstance(rec, Mapping) else {}
        try:
            criticality = float(payload.get("criticality", 3))
        except (TypeError, ValueError):
            criticality = 3.0
        priority = float(np.clip(0.72*hazard + 0.28*((criticality-1.0)/4.0), 0, 1))
        asset_rows.append({
            "name": payload.get("name") or "Unnamed asset",
            "type": payload.get("type") or "Other",
            "site": payload.get("site") or "",
            "criticality": criticality,
            "inspection_priority_screening": priority,
        })
    asset_rows.sort(key=lambda r: r["inspection_priority_screening"], reverse=True)

    assessment = {
        "version": "AEROSENTINEL v2.9.0 automated aviation resilience screen",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "area_name": area_name,
        "mode": "defensive safety / disaster-response / infrastructure-resilience research",
        "profile": profile,
        "scores": {
            "airfield_surface_screening": surface_screening,
            "weather_environment_screening": weather_screening,
            "airfield_environment_screening": airfield_environment,
            "disaster_response_screening": disaster_response,
            "infrastructure_continuity_screening": infrastructure_continuity,
            "reliability": reliability,
            "monsoon_flood_screening": monsoon_flood,
            "high_wind_storm_screening": high_wind_storm,
            "humanitarian_access_screening": humanitarian_access,
        },
        "levels": {
            "airfield_environment": _level(airfield_environment, reliability),
            "disaster_response": _level(disaster_response, reliability),
            "infrastructure_continuity": "INSUFFICIENT DATA" if reliability < 0.35 else ("LOW" if infrastructure_continuity < 0.35 else "ELEVATED" if infrastructure_continuity < 0.65 else "HIGH"),
        },
        "drivers": {
            "satellite_wetness_screening": wet,
            "temporal_change_screening": change,
            "fusion_uncertainty": uncertainty,
            "source_coverage": coverage,
            "temporal_alignment": alignment,
            "rain_pressure": rain_pressure,
            "wind_pressure": wind_pressure,
            "visibility_pressure": visibility_pressure,
            "convective_pressure": convective_pressure,
            "heat_pressure": heat_pressure,
            "terrain_pressure": terrain_pressure,
        },
        "weather_summary": wsum,
        "terrain_summary": tsum,
        "satellite_summary": {
            "finished_at": run.get("finished_at"),
            "available_satellites": fusion.get("available_satellites") or [],
            "acquisition_skew_hours": fusion.get("acquisition_skew_hours"),
            "fusion_preview": fusion.get("preview_path"),
        },
        "response_context": {
            "monsoon_flood": _level(monsoon_flood, reliability),
            "high_wind_storm": _level(high_wind_storm, reliability),
            "humanitarian_access_challenge": _level(humanitarian_access, reliability),
        },
        "alerts": alerts,
        "asset_screening": asset_rows,
        "limitations": [
            "Scores are transparent screening indicators, not calibrated probabilities or certified aviation limits.",
            "Public forecast data is not certified aviation weather and does not replace METAR/TAF or official local products.",
            "Satellite wetness/low-backscatter signals require local verification, especially over pavement, shadow and built-up areas.",
            "Copernicus DEM is a DSM and does not replace surveyed airfield elevations, obstacle data or engineering drainage models.",
            "No targeting, adversary tracking, weapons employment or autonomous operational decision is performed.",
        ],
    }
    signature_body = {
        "sat": assessment["satellite_summary"],
        "weather_time": (weather or {}).get("fetched_at"),
        "scores": assessment["scores"],
    }
    assessment["signature"] = hashlib.sha256(json.dumps(signature_body, sort_keys=True, default=str).encode()).hexdigest()[:20]
    return assessment


def write_briefing(assessment: Mapping, output_dir: str | Path, report_generated_at: str | None = None, geo_x: Mapping | None = None, aircraft_awareness: Mapping | None = None):
    """Write a human-readable + machine-readable local report snapshot.

    ``assessment['generated_at']`` remains the time the analytical screen was
    computed. ``report_generated_at`` is the export/snapshot time, so a 5-minute
    report cadence never pretends that unchanged satellite evidence is new.
    """
    folder = Path(output_dir); folder.mkdir(parents=True, exist_ok=True)
    report = dict(assessment)
    if geo_x:
        report["aerosentinel"] = dict(geo_x)
        report["trust_geo_x"] = dict(geo_x)  # legacy compatibility for older readers/tests
    if aircraft_awareness:
        report["aircraft_awareness"] = dict(aircraft_awareness)
    assessment_time = str(assessment.get("generated_at") or "")
    export_time = report_generated_at or datetime.now(timezone.utc).isoformat()
    report["assessment_generated_at"] = assessment_time or None
    report["report_generated_at"] = export_time
    stamp = str(export_time).replace(":", "").replace("-", "").replace("+", "")[:15] or "latest"
    json_path = folder / f"briefing_{stamp}.json"
    html_path = folder / f"briefing_{stamp}.html"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    scores = report.get("scores") or {}
    levels = report.get("levels") or {}
    alerts = report.get("alerts") or []
    alert_html = "".join(
        f"<li><b>{html.escape(str(a.get('level')))} · {html.escape(str(a.get('category')))}</b>: "
        f"{html.escape(str(a.get('message')))}<br><small>{html.escape(str(a.get('suggested_check')))}</small></li>"
        for a in alerts
    )
    gx = report.get("aerosentinel") or report.get("trust_geo_x") or {}
    gxwarn = gx.get("predictive_early_warning") or {}
    gxunc = gx.get("uncertainty_engine") or {}
    gxinterp = gx.get("interpretation_hypothesis") or {}
    gxobs = gx.get("observed_evidence") or {}
    awareness = report.get("aircraft_awareness") or {}
    awareness_html = ""
    if awareness:
        awareness_html = ("<h2>Satellite aircraft awareness</h2><p><b>State:</b> "
            + html.escape(str(awareness.get("state", "UNKNOWN")))
            + " · Candidate observations: " + str(sum(x.get("presence") == "CANDIDATE" for x in awareness.get("observations", [])))
            + " · Possible movements: " + str(len(awareness.get("anomalies", [])))
            + "</p><p><small>Detector candidates require human review. Satellite revisits do not provide continuous tracking.</small></p>")
    gx_html = ""
    if gx:
        gx_html = f"""
<h2>AEROSENTINEL aviation-safety intelligence</h2>
<div class='cards'><div class='card'><b>Status</b><h2>{html.escape(str(gxwarn.get('status','—')))}</h2><small>human-in-the-loop aviation-safety screen</small></div>
<div class='card'><b>Operational risk screen</b><h2>{float(gxwarn.get('score',0))*100:.0f}/100</h2><small>not a certified aviation probability</small></div>
<div class='card'><b>Confidence</b><h2>{float(gxunc.get('confidence',0)):.0%}</h2><small>reliability × (1−uncertainty)</small></div>
<div class='card'><b>Uncertainty</b><h2>{float(gxunc.get('uncertainty',0)):.0%}</h2><small>OOD/data/history/direct-path limitations</small></div></div>
<h3>Observed evidence</h3><pre>{html.escape(json.dumps(gxobs, indent=2, default=str))}</pre>
<h3>Interpretation / hypothesis</h3><p>{html.escape(str(gxinterp.get('explanation','')))}</p>
<p><b>Recommendation:</b> {html.escape(str(gxwarn.get('recommendation','HUMAN REVIEW')))}</p>
<h3>Rare-event and cross-sensor checks</h3>
<pre>{html.escape(json.dumps({'rare_event_detection': gx.get('rare_event_detection') or {}, 'cross_sensor_generalization': gx.get('cross_sensor_generalization') or {}}, indent=2, default=str))}</pre>
<h3>Operational aviation-risk drivers</h3>
<pre>{html.escape(json.dumps((gx.get('operational_aviation_risk') or {}).get('explainability') or {}, indent=2, default=str))}</pre>
<h3>Deployment readiness</h3>
<pre>{html.escape(json.dumps(gx.get('deployment_readiness') or {}, indent=2, default=str))}</pre>
"""
    body = f"""<!doctype html><html><head><meta charset='utf-8'><title>AEROSENTINEL v2.9.0 Briefing</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#17242b}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{border:1px solid #ccd6db;border-radius:12px;padding:16px;min-width:210px}}small{{color:#566b75}}li{{margin:10px 0}}</style></head><body>
<h1>AEROSENTINEL v2.9.0 Aviation Safety / Automated Resilience Briefing</h1><p><b>Area:</b> {html.escape(str(report.get('area_name')))}<br><b>Report exported:</b> {html.escape(str(export_time))}<br><b>Assessment computed:</b> {html.escape(str(assessment_time or 'not available'))}</p>
<div class='cards'><div class='card'><b>Airfield environment</b><h2>{html.escape(str(levels.get('airfield_environment')))}</h2><small>screen {float(scores.get('airfield_environment_screening',0)):.2f}</small></div>
<div class='card'><b>Disaster response</b><h2>{html.escape(str(levels.get('disaster_response')))}</h2><small>screen {float(scores.get('disaster_response_screening',0)):.2f}</small></div>
<div class='card'><b>Infrastructure continuity</b><h2>{html.escape(str(levels.get('infrastructure_continuity')))}</h2><small>screen {float(scores.get('infrastructure_continuity_screening',0)):.2f}</small></div>
<div class='card'><b>Reliability</b><h2>{float(scores.get('reliability',0)):.0%}</h2><small>research confidence</small></div></div>
{gx_html}
{awareness_html}
<h2>Automated alerts and checks</h2><ul>{alert_html}</ul>
<h2>Important limitations</h2><ul>{''.join('<li>'+html.escape(str(x))+'</li>' for x in report.get('limitations',[]))}</ul>
<p><small>AEROSENTINEL is an aviation-safety research decision-support aid. Human verification and official aviation/weather procedures remain required. A new report file does not imply new satellite acquisition.</small></p></body></html>"""
    html_path.write_text(body, encoding="utf-8")
    latest_json = folder / "latest.json"; latest_html = folder / "latest.html"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_html.write_text(html_path.read_text(encoding="utf-8"), encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path), "latest_json": str(latest_json), "latest_html": str(latest_html),
            "report_generated_at": export_time}

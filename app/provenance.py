"""Evidence provenance records for AEROSENTINEL."""
from __future__ import annotations

from typing import Mapping


def build_provenance_manifest(multisatellite_run: Mapping | None = None,
                              current_observation: Mapping | None = None,
                              foundation_record: Mapping | None = None,
                              highres_evidence: Mapping | None = None) -> dict:
    records = []
    run = multisatellite_run or {}
    for scene in run.get("scene_results") or []:
        records.append({
            "source_type": "satellite_scene", "source": scene.get("satellite"),
            "scene_id": scene.get("scene_id"), "timestamp": scene.get("acquired_at"),
            "processing_level": scene.get("collection"), "observed_or_derived": "observed+processed",
            "native_gsd_m": scene.get("native_gsd_m"), "quality": scene.get("processing_quality"),
            "stack_path": scene.get("stack_path"),
        })
    obs = current_observation or {}
    if obs:
        records.append({
            "source_type": "context_observation", "source": "AEROSENTINEL_context",
            "timestamp": obs.get("timestamp"), "signature": obs.get("signature"),
            "observed_or_derived": "derived_from_observed_inputs",
            "quality": obs.get("quality"),
        })
    if foundation_record:
        records.append({
            "source_type": "foundation_embedding", "source": foundation_record.get("model") or foundation_record.get("backend") or "foundation_model",
            "timestamp": foundation_record.get("acquired_at") or foundation_record.get("timestamp"),
            "observed_or_derived": "derived_embedding", "modalities": foundation_record.get("modalities"),
            "input_sha256": foundation_record.get("input_sha256"),
        })
    if highres_evidence:
        records.append({
            "source_type": "high_resolution_evidence", "source": highres_evidence.get("source") or "authorised_user_evidence",
            "timestamp": highres_evidence.get("acquired_at"), "observed_or_derived": "observed_or_user_supplied",
            "candidate_count": highres_evidence.get("candidate_count"),
        })
    return {
        "records": records,
        "record_count": len(records),
        "policy": "Observed measurements, derived products, model embeddings and any generated/synthetic products must remain distinguishable. Generated data must never be treated as independent sensor confirmation.",
    }

"""Compute-adaptive edge/ground routing policy for AEROSENTINEL."""
from __future__ import annotations

import math
from typing import Mapping
import numpy as np


def _clip(v):
    try: v=float(v)
    except (TypeError,ValueError): return 0.0
    if not math.isfinite(v): return 0.0
    return float(np.clip(v,0,1))


def choose_processing_tier(result: Mapping, deployment: Mapping | None = None) -> dict:
    """Recommend the next compute tier without pretending the policy runs onboard."""
    uncertainty = _clip((result.get("uncertainty_engine") or {}).get("uncertainty"))
    reliability = _clip((result.get("uncertainty_engine") or {}).get("reliability"))
    risk = _clip((result.get("operational_aviation_risk") or {}).get("risk_score")
                 or (result.get("predictive_early_warning") or {}).get("score"))
    abstain = bool((result.get("uncertainty_engine") or {}).get("abstain"))
    quality = result.get("data_quality_and_fusion") or {}
    reg = _clip((quality.get("registration_quality") or {}).get("confidence", 1.0))

    if abstain or reliability < 0.35 or reg < 0.20:
        tier = "HUMAN_REVIEW_OR_REACQUIRE"
        reason = "Evidence quality/reliability is insufficient for automatic escalation."
    elif risk < 0.25 and uncertainty < 0.30:
        tier = "FAST_SCREEN_ARCHIVE"
        reason = "Low screening risk and low uncertainty; retain compressed analytical products."
    elif risk < 0.55 and uncertainty < 0.50:
        tier = "ROI_DETAILED_ANALYSIS"
        reason = "Intermediate evidence warrants focused ROI processing rather than full-scene expensive inference."
    else:
        tier = "GROUND_FOUNDATION_MODEL_PLUS_REVIEW"
        reason = "High risk or uncertainty warrants the higher-cost ground-side model and analyst review."
    return {
        "recommended_tier": tier, "reason": reason,
        "risk_screening": risk, "uncertainty": uncertainty, "reliability": reliability,
        "registration_confidence": reg,
        "operational_note": "Policy is a ground-side scheduling recommendation in this software build; it is not claimed to execute autonomously onboard a spacecraft.",
    }

"""Uncertainty-aware thermal processing for AEROSENTINEL.

This module preserves the provider-calibrated Landsat Collection-2 Level-2
surface-temperature measurement and adds diagnostics that help users judge
whether a pixel is suitable for quantitative interpretation. It intentionally
does not recompute Landsat surface temperature from raw radiance because the
USGS Level-2 product already includes atmospheric and emissivity corrections.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    """Return median and a MAD-derived robust sigma.

    The 1.4826 factor makes MAD comparable to standard deviation for an
    approximately Gaussian distribution. A small floor prevents unstable
    anomaly scores in nearly uniform synthetic/test scenes.
    """
    x = np.asarray(values, dtype="float64")
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan, math.nan
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = max(1.4826 * mad, 0.25)
    return med, sigma


def inverse_variance_weighted_mean(values: np.ndarray, uncertainty: np.ndarray,
                                   valid: np.ndarray | None = None) -> float | None:
    """Uncertainty-aware mean using 1/sigma^2 weights.

    ST_QA is per-pixel temperature uncertainty in Kelvin. This statistic is a
    scene summary only; it is not reported as a formal confidence interval
    because neighboring resampled thermal pixels are spatially correlated.
    """
    v = np.asarray(values, dtype="float64")
    u = np.asarray(uncertainty, dtype="float64")
    mask = np.isfinite(v) & np.isfinite(u) & (u >= 0)
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    if not np.any(mask):
        return None
    sigma = np.maximum(u[mask], 0.05)
    w = 1.0 / np.square(sigma)
    return float(np.sum(v[mask] * w) / np.sum(w))


def thermal_diagnostics(surface_temp_k: np.ndarray, valid: np.ndarray,
                        *, uncertainty_k: np.ndarray | None = None,
                        emissivity: np.ndarray | None = None,
                        emissivity_std: np.ndarray | None = None,
                        cloud_distance_km: np.ndarray | None = None,
                        atmospheric_transmittance: np.ndarray | None = None) -> dict:
    """Build uncertainty-aware Landsat thermal diagnostics.

    Returns measurement-preserving arrays plus conservative screening layers:
    * temperature anomaly from the scene median;
    * robust standardized anomaly (MAD based);
    * uncertainty-normalized anomaly when ST_QA exists;
    * inverse-variance weight for optional weighted summaries;
    * explicit cloud-proximity and elevated-uncertainty flags.

    The flags are workflow heuristics and are never presented as USGS quality
    classes or as a certified active-fire product.
    """
    t = np.asarray(surface_temp_k, dtype="float32")
    base_valid = np.asarray(valid, dtype=bool) & np.isfinite(t)
    c = (t - 273.15).astype("float32")

    median_c, robust_sigma_c = robust_location_scale(c[base_valid])
    delta_c = np.full_like(c, np.nan, dtype="float32")
    robust_z = np.full_like(c, np.nan, dtype="float32")
    if np.isfinite(median_c):
        delta_c[base_valid] = c[base_valid] - median_c
        robust_z[base_valid] = delta_c[base_valid] / float(robust_sigma_c)

    uncertainty_z = np.full_like(c, np.nan, dtype="float32")
    inverse_variance = np.full_like(c, np.nan, dtype="float32")
    elevated_uncertainty = np.zeros(c.shape, dtype=bool)
    uncertainty_summary = None
    weighted_mean_c = None
    if uncertainty_k is not None:
        u = np.asarray(uncertainty_k, dtype="float32")
        uq = base_valid & np.isfinite(u) & (u >= 0)
        if np.any(uq):
            # Compare the local temperature departure with both scene
            # variability and the product-provided per-pixel uncertainty.
            denom = np.sqrt(float(robust_sigma_c) ** 2 + np.square(u[uq]))
            uncertainty_z[uq] = delta_c[uq] / np.maximum(denom, 0.05)
            inverse_variance[uq] = 1.0 / np.square(np.maximum(u[uq], 0.05))
            weighted_mean_c = inverse_variance_weighted_mean(c, u, base_valid)
            vals = u[uq]
            uncertainty_summary = {
                "mean_k": float(np.mean(vals)),
                "median_k": float(np.median(vals)),
                "p90_k": float(np.percentile(vals, 90)),
                "max_k": float(np.max(vals)),
            }
            # AEROSENTINEL heuristic only: >2 K is surfaced as an elevated
            # uncertainty warning, not rejected or relabeled as invalid.
            elevated_uncertainty = uq & (u > 2.0)

    near_cloud = np.zeros(c.shape, dtype=bool)
    cloud_distance_summary = None
    if cloud_distance_km is not None:
        d = np.asarray(cloud_distance_km, dtype="float32")
        dq = base_valid & np.isfinite(d) & (d >= 0)
        if np.any(dq):
            vals = d[dq]
            cloud_distance_summary = {
                "median_km": float(np.median(vals)),
                "p10_km": float(np.percentile(vals, 10)),
                "min_km": float(np.min(vals)),
            }
            # Conservative UI warning threshold only. The official ST_QA
            # remains the quantitative uncertainty source.
            near_cloud = dq & (d < 1.0)

    def summary_optional(arr: np.ndarray | None, valid_range: tuple[float, float] | None = None):
        if arr is None:
            return None
        a = np.asarray(arr, dtype="float32")
        q = base_valid & np.isfinite(a)
        if valid_range is not None:
            q &= (a >= valid_range[0]) & (a <= valid_range[1])
        if not np.any(q):
            return None
        vals = a[q]
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
        }

    # Require a substantial positive departure and robust standardized signal.
    # This is an anomaly-screening flag, explicitly not an active-fire class.
    candidate = base_valid & np.isfinite(robust_z) & (delta_c >= 3.0) & (robust_z >= 3.0)
    if uncertainty_k is not None:
        uz = np.asarray(uncertainty_z)
        # If ST_QA is available, anomaly promotion requires an actual finite
        # uncertainty-qualified score for that pixel. Missing ST_QA is not
        # silently treated as high confidence.
        candidate &= np.isfinite(uz) & (uz >= 2.0)

    return {
        "surface_temp_c": c,
        "temperature_anomaly_c": delta_c,
        "robust_anomaly_z": robust_z,
        "uncertainty_normalized_anomaly": uncertainty_z,
        "inverse_variance_weight": inverse_variance,
        "elevated_uncertainty_flag": elevated_uncertainty,
        "near_cloud_flag": near_cloud,
        "thermal_anomaly_candidate": candidate,
        "stats": {
            "scene_median_surface_c": None if not np.isfinite(median_c) else float(median_c),
            "scene_robust_sigma_c": None if not np.isfinite(robust_sigma_c) else float(robust_sigma_c),
            "uncertainty_weighted_mean_surface_c": weighted_mean_c,
            "st_qa_uncertainty": uncertainty_summary,
            "cloud_distance": cloud_distance_summary,
            "emissivity": summary_optional(emissivity, (0.0, 1.1)),
            "emissivity_std": summary_optional(emissivity_std, (0.0, 1.0)),
            "atmospheric_transmittance": summary_optional(atmospheric_transmittance, (0.0, 1.2)),
            "elevated_uncertainty_percent": float(100.0 * np.mean(elevated_uncertainty[base_valid])) if np.any(base_valid) else None,
            "near_cloud_percent": float(100.0 * np.mean(near_cloud[base_valid])) if np.any(base_valid) else None,
            "thermal_anomaly_candidate_percent": float(100.0 * np.mean(candidate[base_valid])) if np.any(base_valid) else None,
        },
    }

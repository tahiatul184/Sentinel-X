"""Lightweight corruption stress profile for EO visualization/processing quality.

This is REOBench-inspired engineering stress testing, not a substitute for a
labelled task benchmark.  It measures degradation in AEROSENTINEL's image
processing-readiness metric under controlled corruptions.
"""
from __future__ import annotations

import numpy as np
from quality_control import image_quality_metrics


def _u8(x):
    return np.clip(x,0,255).astype("uint8")


def image_corruption_profile(rgb_uint8, valid=None, cloud_cover_percent=None, seed=2026):
    import cv2
    rgb = _u8(np.asarray(rgb_uint8))
    baseline = image_quality_metrics(rgb, valid, cloud_cover_percent)
    rng = np.random.default_rng(int(seed))
    corruptions = {}
    corruptions["gaussian_blur"] = cv2.GaussianBlur(rgb, (5,5), 1.2)
    noise = rng.normal(0, 12.0, rgb.shape)
    corruptions["gaussian_noise"] = _u8(rgb.astype("float32") + noise)
    low = cv2.resize(rgb, (max(2,rgb.shape[1]//2), max(2,rgb.shape[0]//2)), interpolation=cv2.INTER_AREA)
    corruptions["resample_loss"] = cv2.resize(low, (rgb.shape[1],rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    corruptions["contrast_loss"] = _u8((rgb.astype("float32") - 127.5) * 0.60 + 127.5)
    ok, enc = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 35])
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        corruptions["jpeg_q35"] = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    base_score = float(baseline.get("processing_readiness_score") or 0.0)
    rows = []
    for name, img in corruptions.items():
        q = image_quality_metrics(img, valid, cloud_cover_percent)
        score = float(q.get("processing_readiness_score") or 0.0)
        rows.append({"corruption":name,"processing_readiness_score":score,
                     "absolute_drop":max(0.0,base_score-score)})
    drops=[r["absolute_drop"] for r in rows]
    return {
        "baseline_processing_readiness": base_score,
        "corruptions": rows,
        "mean_readiness_drop": float(np.mean(drops)) if drops else 0.0,
        "worst_readiness_drop": float(np.max(drops)) if drops else 0.0,
        "note": "Engineering corruption stress on image-processing readiness only; not task mIoU/mAP robustness.",
    }

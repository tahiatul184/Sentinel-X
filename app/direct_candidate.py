"""Resolution-gated small-object saliency screening for user-supplied high-resolution imagery.

This is intentionally a generic candidate detector. It does not identify a UAV,
does not geolocate candidates for targeting, and returns image-space evidence
for human research review only.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from trust_geo_x import resolution_feasibility
from imagery_archive import archive_highres_result, count_highres_inputs, ensure_storage_layout


def _box_mean(arr: np.ndarray, radius=2):
    radius = max(1, int(radius))
    padded = np.pad(arr, radius, mode="reflect")
    # Integral image with a leading zero row/column makes window sums vectorized.
    integral = np.pad(padded, ((1,0),(1,0)), mode="constant").cumsum(0).cumsum(1)
    k = 2*radius + 1
    h, w = arr.shape
    sums = integral[k:k+h, k:k+w] - integral[:h, k:k+w] - integral[k:k+h, :w] + integral[:h, :w]
    return sums / float(k*k)


def _estimate_gsd(src, latitude=None):
    transform = src.transform
    if src.crs and getattr(src.crs, "is_projected", False):
        try:
            meters_per_unit = float(src.crs.linear_units_factor[1])
        except (AttributeError, TypeError, IndexError):
            return None
        return float((math.hypot(transform.a, transform.d) +
                      math.hypot(transform.b, transform.e)) / 2.0 * meters_per_unit)
    if src.crs and getattr(src.crs, "is_geographic", False):
        lat = latitude
        if lat is None:
            try:
                lat = (src.bounds.bottom + src.bounds.top) / 2.0
            except Exception:
                lat = 0.0
        mx = abs(transform.a) * 111320.0 * max(0.1, math.cos(math.radians(float(lat))))
        my = abs(transform.e) * 110540.0
        return float((mx + my) / 2.0)
    return None


def analyze_highres(path: str | Path, *, gsd_m: float | None = None,
                    nominal_object_m=2.0, max_candidates=25) -> dict:
    import rasterio
    from rasterio.enums import Resampling

    path = Path(path)
    with rasterio.open(path) as src:
        gsd = float(gsd_m) if gsd_m else _estimate_gsd(src)
        feasibility = resolution_feasibility(gsd, nominal_object_m=nominal_object_m)
        max_dim = 2048
        scale = min(1.0, max_dim / max(src.width, src.height))
        oh = max(1, int(round(src.height * scale)))
        ow = max(1, int(round(src.width * scale)))
        count = min(3, src.count)
        data = src.read(list(range(1, count+1)), out_shape=(count, oh, ow), masked=True, resampling=Resampling.bilinear).astype(np.float32)
        mask = np.ma.getmaskarray(data)
        valid = ~np.any(mask, axis=0) if mask.ndim == 3 else ~mask
        values = np.ma.filled(data, np.nan)
        gray = np.nanmean(values, axis=0)
    finite = valid & np.isfinite(gray)
    if not finite.any():
        raise ValueError("High-resolution image contains no valid pixels for screening.")
    vals = gray[finite]
    lo, hi = np.percentile(vals, [2, 98])
    if hi <= lo:
        norm = np.zeros_like(gray, dtype=np.float32)
    else:
        norm = np.clip((gray-lo)/(hi-lo), 0, 1).astype(np.float32)
    fill = float(np.nanmedian(norm[finite]))
    norm = np.where(finite, norm, fill)
    local = _box_mean(norm, radius=2)
    contrast = np.abs(norm-local)
    contrast[~finite] = 0.0

    candidates = []
    max_saliency = 0.0
    segmentation = np.zeros_like(contrast, dtype=bool)
    threshold = None
    if feasibility["feasible"]:
        threshold = float(np.quantile(contrast[finite], 0.9985))
        segmentation = (contrast >= threshold) & finite
        ys, xs = np.where(segmentation)
        scored = sorted(((float(contrast[y,x]), int(y), int(x)) for y,x in zip(ys,xs)), reverse=True)
        min_sep = max(4, int(round((nominal_object_m/max(gsd or 1e-6, 1e-6)) * scale)))
        for score, y, x in scored:
            if any((x-c["x_px"])**2 + (y-c["y_px"])**2 < min_sep**2 for c in candidates):
                continue
            candidates.append({"x_px": x, "y_px": y, "saliency": float(np.clip(score/max(threshold,1e-6),0,2)/2)})
            if len(candidates) >= int(max_candidates):
                break
        max_saliency = max((c["saliency"] for c in candidates), default=0.0)

    mask_path = None
    try:
        from PIL import Image
        mask_path = path.with_suffix(path.suffix + ".candidate_mask.png")
        Image.fromarray((segmentation.astype(np.uint8) * 255)).save(mask_path)
    except Exception:
        mask_path = None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "version": "AEROSENTINEL direct candidate research screening",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "file": str(path),
        "sha256": digest,
        "gsd_m": gsd,
        "feasibility": feasibility,
        "candidate_count": len(candidates),
        "max_candidate_saliency": float(max_saliency),
        "image_space_candidates": candidates,
        "segmentation_mask_path": str(mask_path) if mask_path else None,
        "candidate_mask_percent": float(segmentation.mean() * 100.0),
        "saliency_threshold": threshold,
        "identity": "UNKNOWN / NOT CLASSIFIED",
        "method": "resolution-gated local-contrast object-candidate detection + image-space saliency segmentation; no task-specific UAV classifier",
        "limitations": [
            "Candidates are image-space saliency points and may be vehicles, structures, shadows, glint, noise or other small objects.",
            "No geographic target coordinates are produced by this module.",
            "No hostile/benign classification is attempted.",
            "Direct screening is disabled automatically when source GSD is insufficient.",
        ],
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_inbox(root: str | Path, cfg: Mapping, previous: Mapping | None = None):
    root = Path(root)
    layout = ensure_storage_layout(root, cfg)
    inbox = Path(layout["highres_inbox"])
    previous = previous or {}
    seen = set(previous.get("seen_sha256") or [])
    seen_files = dict(previous.get("seen_files") or {})
    latest = previous.get("latest") or None
    history = list(previous.get("history") or [])[-99:]
    errors = []
    changed = False
    processed_this_scan = 0
    for path in sorted(list(inbox.glob("*.tif")) + list(inbox.glob("*.tiff")), key=lambda p:p.stat().st_mtime):
        try:
            stat = path.stat(); stat_key = f"{stat.st_size}:{stat.st_mtime_ns}"
            if seen_files.get(str(path)) == stat_key:
                continue
            digest = _sha256_file(path)
            if digest in seen:
                seen_files[str(path)] = stat_key
                continue
            gsd = None
            sidecar = path.with_suffix(path.suffix + ".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8")); gsd = meta.get("gsd_m")
                except Exception as exc:
                    errors.append({"file": str(path), "stage": "SIDECAR", "error": f"{type(exc).__name__}: {exc}"})
            result = analyze_highres(path, gsd_m=gsd)
            archive_record = archive_highres_result(root, cfg, path, result=result)
            latest = result
            seen.add(digest); seen_files[str(path)] = stat_key
            history.append({
                "file": str(path), "sha256": digest, "status": "ANALYZED",
                "analyzed_at": result.get("analyzed_at"),
                "candidate_count": result.get("candidate_count"),
                "gsd_m": result.get("gsd_m"),
                "archive_result": archive_record.get("result_path"),
            })
            processed_this_scan += 1; changed = True
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            errors.append({"file": str(path), "stage": "ANALYZE", "error": message})
            try:
                archive_highres_result(root, cfg, path, error=message)
                stat = path.stat(); seen_files[str(path)] = f"{stat.st_size}:{stat.st_mtime_ns}"
            except Exception:
                pass
            history.append({"file": str(path), "status": "FAILED", "error": message})
            changed = True
    state = {
        "latest": latest, "seen_sha256": sorted(seen)[-200:], "seen_files": seen_files,
        "inbox": str(inbox), "input_file_count": count_highres_inputs(root, cfg),
        "processed_this_scan": processed_this_scan, "history": history[-100:],
        "errors": errors[-20:], "last_scan_at": datetime.now(timezone.utc).isoformat(),
        "processed_output": layout["highres_output"], "highres_log": layout["highres_log"],
        "auto_download": False,
        "note": "High-resolution inbox is manual/licensed input; Sentinel/Landsat products are archived separately under data/imagery.",
    }
    return state, changed

"""Optional TerraMind 1.0 multimodal feature extraction for AEROSENTINEL.

This runner uses the official TerraTorch TerraMind backbone with three
pre-trained raw EO modalities: Sentinel-2 L2A, Sentinel-1 GRD and DEM.
Weather/rainfall and historical context are fused by explicit context adapters
in the overall hybrid foundation stage in ``multimodal_pipeline.py``; they are
not claimed as TerraMind raw pre-training modalities.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import json
import os
import time

import numpy as np

from collection_store import atomic_json, sha256
from trust_core import timestamp

MODEL = "terramind_v1_tiny"
MODEL_GUIDE = "https://github.com/torchgeo/terratorch/blob/main/docs/guide/terramind.md"
MODALITIES = ["S2L2A", "S1GRD", "DEM"]
S2_BANDS = [
    "COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
    "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2",
]
S1_BANDS = ["VV", "VH"]
S2_MEAN = np.array([1390.458,1503.317,1718.197,1853.910,2199.100,2779.975,2987.011,3083.234,3132.220,3162.988,2424.884,1857.648], dtype=np.float32)
S2_STD = np.array([2106.761,2141.107,2038.973,2134.138,2085.321,1889.926,1820.257,1871.918,1753.829,1797.379,1434.261,1334.311], dtype=np.float32)
S1_MEAN = np.array([-12.599, -20.293], dtype=np.float32)
S1_STD = np.array([5.195, 5.890], dtype=np.float32)
DEM_MEAN = np.array([670.665], dtype=np.float32)
DEM_STD = np.array([951.272], dtype=np.float32)


def ready():
    return all(importlib.util.find_spec(m) is not None for m in ["torch", "terratorch", "rasterio"])


def _center_s2(path, units):
    import rasterio
    from rasterio.windows import Window, transform as window_transform
    with rasterio.open(path) as src:
        if src.count != 12 or src.width < 224 or src.height < 224:
            raise ValueError("Sentinel-2 input must be a 12-band L2A stack of at least 224 x 224 pixels.")
        if not src.crs:
            raise ValueError("Sentinel-2 input needs a valid CRS.")
        window = Window((src.width - 224)//2, (src.height - 224)//2, 224, 224)
        data = src.read(window=window, masked=True).astype(np.float32)
        mask = np.ma.getmaskarray(data)
        if mask.any():
            raise ValueError("Sentinel-2 center patch contains nodata; supply a cloud/quality-reviewed valid patch.")
        data = np.asarray(data)
        if units == "reflectance":
            data = data * 10000.0
        elif units != "scaled_10000":
            raise ValueError("Sentinel-2 units must be reflectance or scaled_10000.")
        if data.min() < -2000 or data.max() > 20000:
            raise ValueError("Sentinel-2 values do not match declared reflectance scaling.")
        transform = window_transform(window, src.transform)
        crs = src.crs
        bounds = rasterio.windows.bounds(window, src.transform)
        return data, crs, transform, bounds


def _reproject_to_grid(path, count, target_crs, target_transform, shape, *, value_range, label):
    import rasterio
    from rasterio.warp import reproject, Resampling
    with rasterio.open(path) as src:
        if src.count != count:
            raise ValueError(f"{label} input must contain exactly {count} band(s).")
        if not src.crs:
            raise ValueError(f"{label} input needs a valid CRS.")
        dst = np.full((count, shape[0], shape[1]), np.nan, dtype=np.float32)
        for band in range(1, count + 1):
            reproject(
                source=rasterio.band(src, band), destination=dst[band-1],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=target_transform, dst_crs=target_crs,
                src_nodata=src.nodata, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    valid = np.isfinite(dst)
    if valid.mean() < 0.95:
        raise ValueError(f"{label} does not cover at least 95% of the Sentinel-2 center patch.")
    lo, hi = value_range
    vals = dst[valid]
    if vals.min() < lo or vals.max() > hi:
        raise ValueError(f"{label} values fall outside the expected research range [{lo}, {hi}].")
    # Fill a tiny residual edge gap only after the 95% coverage rule.
    if not valid.all():
        for b in range(count):
            finite = np.isfinite(dst[b])
            fill = float(np.median(dst[b, finite])) if finite.any() else 0.0
            dst[b, ~finite] = fill
    return dst


def _normalise(s2, s1, dem):
    return {
        "S2L2A": (s2 - S2_MEAN[:, None, None]) / S2_STD[:, None, None],
        "S1GRD": (s1 - S1_MEAN[:, None, None]) / S1_STD[:, None, None],
        "DEM": (dem - DEM_MEAN[:, None, None]) / DEM_STD[:, None, None],
    }


def extract(s2_path, s1_path, dem_path, output, units, acquired):
    import torch
    from terratorch.registry import BACKBONE_REGISTRY
    from importlib.metadata import version

    acquired = timestamp(acquired)
    s2, crs, target_transform, bounds = _center_s2(s2_path, units)
    s1 = _reproject_to_grid(
        s1_path, 2, crs, target_transform, (224, 224),
        value_range=(-70.0, 30.0), label="Sentinel-1 calibrated dB",
    )
    dem = _reproject_to_grid(
        dem_path, 1, crs, target_transform, (224, 224),
        value_range=(-600.0, 9000.0), label="DEM elevation metres",
    )
    arrays = _normalise(s2, s1, dem)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    model = BACKBONE_REGISTRY.build(
        MODEL, pretrained=True, modalities=MODALITIES, merge_method="mean"
    )
    model.eval()
    inputs = {name: torch.from_numpy(value)[None].float() for name, value in arrays.items()}
    with torch.inference_mode():
        features = model(inputs)
        value = features[-1] if isinstance(features, (list, tuple)) else features
        if value.ndim != 3:
            raise RuntimeError(f"Unexpected TerraMind feature shape: {tuple(value.shape)}")
        vector = value.mean(dim=1)[0].detach().cpu().numpy()
    if not np.isfinite(vector).all():
        raise RuntimeError("TerraMind returned non-finite features.")

    input_hashes = {
        "sentinel_2": sha256(Path(s2_path)),
        "sentinel_1": sha256(Path(s1_path)),
        "dem": sha256(Path(dem_path)),
    }
    result = {
        "model": MODEL,
        "model_guide": MODEL_GUIDE,
        "origin": "Model-derived",
        "acquired_at": acquired.isoformat(),
        "modalities": MODALITIES,
        "sentinel_2_bands": S2_BANDS,
        "sentinel_1_bands": S1_BANDS,
        "input_sha256": input_hashes,
        "crs": str(crs),
        "s2_center_bounds": [float(v) for v in bounds],
        "patch": [224, 224],
        "embedding": vector.astype(float).tolist(),
        "dimension": int(vector.size),
        "runtime": {"torch": version("torch"), "terratorch": version("terratorch")},
        "interpretation": "TerraMind multimodal S1+S2+DEM pooled feature vector; no task-specific classifier is attached.",
        "context_fusion": "Weather/rainfall and historical-image features enter explicit context adapters in the overall hybrid foundation stage in multimodal_pipeline.py; they are not claimed as TerraMind pre-trained raw modalities here.",
        "limitations": "Inputs must be co-located and correctly scaled. S1 must already be calibrated to dB. The embedding is not a flood/damage probability and needs task validation.",
    }
    atomic_json(Path(output), result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    status = Path(request["status"])
    try:
        atomic_json(status, {"state": "RUNNING", "pid": os.getpid(), "time": time.time(), "message": "Loading TerraMind multimodal backbone; first run may download weights."})
        extract(
            request["s2_input"], request["s1_input"], request["dem_input"],
            request["output"], request["s2_units"], request["acquired"],
        )
        atomic_json(status, {"state": "COMPLETE", "time": time.time(), "output": request["output"]})
    except Exception as exc:
        atomic_json(status, {"state": "FAILED", "time": time.time(), "error": f"{type(exc).__name__}: {exc}"})
        raise

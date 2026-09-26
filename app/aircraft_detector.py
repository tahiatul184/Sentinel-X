"""Tiled optical satellite aircraft inference with explicit geospatial provenance.

The default checkpoint is downloaded on first use. It was evaluated by its
publisher on high-resolution optical imagery; local domain validation remains
necessary. SAR/thermal scenes are not passed through this optical detector.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from datetime import datetime

import numpy as np

from aircraft_awareness import analyze, parse_detections, FIELDS

MODEL_REPO = 'iturslab/Efficient-YOLO-RS-Airplane-Detection'
MODEL_FILE = 'transfer-learning/experiment-12/best.pt'
MODEL_REVISION = '38fc6ae'
MODEL_ID = f'{MODEL_REPO}@{MODEL_REVISION}/{MODEL_FILE}'


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint=None):
    """Load a local checkpoint or the pinned research checkpoint on demand."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError('Aircraft AI packages are missing; rerun start_dashboard.py to install dashboard requirements.') from exc
    if checkpoint is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError('huggingface_hub is required for the research checkpoint.') from exc
        checkpoint = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REVISION)
    path = Path(checkpoint)
    if not path.is_file():
        raise ValueError('Aircraft checkpoint does not exist.')
    return YOLO(str(path)), _sha256(path), str(path)


def _rgb_tile(src, window):
    from rasterio.enums import Resampling
    if src.count < 3:
        raise ValueError('Optical aircraft detector requires a three-band RGB GeoTIFF.')
    cube = src.read([1, 2, 3], window=window, masked=True, resampling=Resampling.nearest)
    mask = np.ma.getmaskarray(cube)
    valid = ~np.any(mask, axis=0)
    values = np.ma.filled(cube.astype(np.float32), np.nan)
    if valid.mean() < 0.5:
        return None, float(valid.mean())
    rgb = np.empty((values.shape[1], values.shape[2], 3), dtype=np.uint8)
    for band in range(3):
        channel = values[band]
        good = channel[valid & np.isfinite(channel)]
        if not good.size:
            return None, float(valid.mean())
        lo, hi = np.percentile(good, [2, 98])
        rgb[:, :, band] = np.clip((channel - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb, float(valid.mean())


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    aa = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / max(aa+bb-inter, 1e-9)


def infer_geotiff(path, *, site_id, scene_id, acquired_at, model=None,
                  checkpoint=None, confidence=0.35, tile_size=960,
                  overlap=160, max_tiles=256, gsd_limit_m=2.0,
                  registration_error_m=None, cloud_fraction=0.0):
    """Run high-resolution RGB detection and return the awareness contract.

    `model` is injectable for offline tests. No inference is attempted on
    unreferenced, coarse-resolution, or non-RGB imagery.
    """
    if not str(site_id).strip() or not str(scene_id).strip():
        raise ValueError('site_id and scene_id are required.')
    try:
        when = datetime.fromisoformat(str(acquired_at).replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('Acquisition time must be ISO-8601 with timezone.') from exc
    if when.tzinfo is None:
        raise ValueError('Acquisition time requires a timezone.')
    if registration_error_m is not None and (not math.isfinite(float(registration_error_m)) or float(registration_error_m) < 0):
        raise ValueError('Registration error must be a nonnegative number of meters.')
    if not 0 < confidence < 1 or not 0 <= cloud_fraction <= 1:
        raise ValueError('Confidence and cloud fraction must be valid fractions.')
    if tile_size < 256 or overlap < 0 or overlap >= tile_size:
        raise ValueError('Invalid tile size or overlap.')
    source = Path(path)
    if not source.is_file():
        raise ValueError('GeoTIFF does not exist.')
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import transform as warp_transform
    from direct_candidate import _estimate_gsd
    with rasterio.open(source) as src:
        if not src.crs or not src.transform or src.count < 3:
            raise ValueError('A georeferenced RGB GeoTIFF with CRS and transform is required.')
        gsd = _estimate_gsd(src)
        if gsd is None or not math.isfinite(gsd) or gsd > gsd_limit_m:
            raise ValueError(f'Native resolution {gsd} m is insufficient; require at most {gsd_limit_m} m/pixel.')
        steps = tile_size - overlap
        xs = list(range(0, src.width, steps))
        ys = list(range(0, src.height, steps))
        if len(xs) * len(ys) > max_tiles:
            raise ValueError(f'Scene exceeds {max_tiles} inference tiles; crop the study area.')
        if model is None:
            model, digest, model_path = load_model(checkpoint)
            model_id = MODEL_ID if checkpoint is None else f'local:{Path(checkpoint).name}'
        else:
            digest, model_path, model_id = None, None, 'injected-model'
        found = []
        valid_tiles = 0
        for y in ys:
            for x in xs:
                width, height = min(tile_size, src.width-x), min(tile_size, src.height-y)
                if width < 128 or height < 128:
                    continue
                rgb, valid = _rgb_tile(src, Window(x, y, width, height))
                if rgb is None:
                    continue
                valid_tiles += 1
                prediction = model.predict(rgb, imgsz=tile_size, conf=confidence, verbose=False)[0]
                names = prediction.names
                for box in prediction.boxes:
                    cls = int(box.cls.item())
                    label = str(names[cls] if isinstance(names, dict) else names[cls]).lower()
                    if label not in {'aircraft', 'airplane', 'aeroplane', 'plane'}:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bounds = (x+x1, y+y1, x+x2, y+y2)
                    found.append((float(box.conf.item()), bounds, valid))
        # Deduplicate overlapping windows in source pixel coordinates.
        retained = []
        for item in sorted(found, key=lambda v:v[0], reverse=True):
            if not any(_iou(item[1], other[1]) > 0.5 for other in retained):
                retained.append(item)
        rows = []
        for score, (x1, y1, x2, y2), quality in retained:
            gx, gy = src.transform * ((x1+x2)/2, (y1+y2)/2)
            lon, lat = warp_transform(src.crs, 'EPSG:4326', [gx], [gy])
            rows.append(dict(site_id=str(site_id), scene_id=str(scene_id), acquired_at=str(acquired_at),
                modality='optical', latitude=lat[0], longitude=lon[0], confidence=score,
                gsd_m=gsd, object_length_m=max(x2-x1, y2-y1)*gsd,
                registration_error_m=registration_error_m, model_id=model_id,
                quality=quality, cloud_fraction=cloud_fraction))
    # Run all outputs through the same validated input contract, including time.
    import csv
    import io
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    checked = parse_detections(buf.getvalue()) if rows else []
    return dict(rows=checked, source=str(source), source_sha256=_sha256(source),
        model_id=model_id, model_sha256=digest, model_path=model_path,
        product_gsd_m=gsd, native_gsd_m=None, resolution_status="NATIVE RESOLUTION UNVERIFIED", tiles_analyzed=valid_tiles, detections=len(rows),
        state='CANDIDATES' if rows else 'NO DETECTION / ABSENCE UNDETERMINED',
        awareness=analyze(checked),
        limitations='Optical research detector only. No detections do not prove absence; site-specific precision and recall require independent evaluation.')

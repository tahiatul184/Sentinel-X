"""Persistent, user-visible imagery and high-resolution processing logs.

AEROSENTINEL's concurrent satellite worker writes analytical outputs under
``data/multisatellite/runs``. This module mirrors those outputs into the configured
``archive_root`` and maintains JSON/JSONL audit logs so users can see that imagery
was actually collected and processed.

The high-resolution inbox remains an *input* folder for user-supplied/licensed
fine-resolution GeoTIFFs. A README and persistent processed-output log make that
contract explicit instead of leaving an unexplained empty directory.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterable, Mapping


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _safe(value: object, limit: int = 96) -> str:
    text = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value or 'unknown')).strip('_.-') or 'unknown'
    return text[:limit]


def _local_path(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def _atomic_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str), encoding='utf-8')
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, allow_nan=False, default=str) + '\n')


def _link_or_copy(src: Path, dst: Path) -> str | None:
    """Prefer a hard-link to avoid duplicate large raster bytes; copy as fallback."""
    src = Path(src)
    if not src.is_file():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return str(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


def ensure_storage_layout(root: str | Path, cfg: Mapping) -> dict:
    root = Path(root)
    archive = _local_path(root, cfg.get('archive_root', 'data/imagery'))
    highres_inbox = _local_path(root, cfg.get('geo_x_highres_inbox', 'highres_inbox'))
    highres_output = root / 'data/highres'
    archive.mkdir(parents=True, exist_ok=True)
    highres_inbox.mkdir(parents=True, exist_ok=True)
    highres_output.mkdir(parents=True, exist_ok=True)

    archive_readme = archive / 'README_IMAGE_ARCHIVE.txt'
    if not archive_readme.exists():
        archive_readme.write_text(
            'AEROSENTINEL SATELLITE IMAGE ARCHIVE\n'
            '===================================\n\n'
            'Successful Sentinel-1, Sentinel-2, Landsat and fused analytical outputs\n'
            'are mirrored here automatically. image_log.jsonl appears after the first\n'
            'successful archive entry. If no imagery appears, first confirm that a\n'
            'monitoring AOI/location has been saved and check System Health/service logs.\n',
            encoding='utf-8')

    readme = highres_inbox / 'README_ADD_HIGH_RES_IMAGES.txt'
    if not readme.exists():
        readme.write_text(
            'AEROSENTINEL HIGH-RESOLUTION INPUT FOLDER\n'
            '========================================\n\n'
            'This folder is an INPUT inbox. AEROSENTINEL does not silently download\n'
            'commercial/fine-resolution imagery into it. Add legally/authoritatively\n'
            'available GeoTIFF files here, or upload them from the AEROSENTINEL Early\n'
            'Warning page. The background worker will detect new/changed .tif/.tiff\n'
            'files and write processing results to app\\data\\highres\\.\n\n'
            'Freely available Sentinel-1/Sentinel-2/Landsat analytical imagery is\n'
            'stored in app\\data\\imagery\\ and logged in image_log.jsonl.\n',
            encoding='utf-8')

    status = {
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'satellite_archive': str(archive),
        'satellite_image_log': str(archive / 'image_log.jsonl'),
        'highres_inbox': str(highres_inbox),
        'highres_output': str(highres_output),
        'highres_log': str(highres_output / 'highres_log.jsonl'),
        'highres_auto_download': False,
        'highres_contract': 'manual/licensed GeoTIFF input; not an automatic commercial-imagery downloader',
    }
    _atomic_json(root / 'data/collection/storage_status.json', status)
    _atomic_json(archive / 'archive_status.json', {**status, 'auto_location_ready': bool(cfg.get('auto_location_ready', False))})
    return status


def _run_key(run: Mapping) -> str:
    parts = [str(run.get("finished_at") or ""), str(run.get("started_at") or "")]
    parts.extend(str(r.get("satellite"))+":"+str(r.get("scene_id")) for r in (run.get("scene_results") or []))
    import hashlib
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def _load_archive_index(archive: Path) -> set[str]:
    path = archive / "archive_index.json"
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("run_keys") or [])
    except Exception:
        return set()


def _save_archive_index(archive: Path, keys: Iterable[str]) -> None:
    _atomic_json(archive / "archive_index.json", {"run_keys": sorted(set(keys)), "updated_at": datetime.now(timezone.utc).isoformat()})


def archive_multisatellite_run(root: str | Path, cfg: Mapping, run: Mapping) -> list[dict]:
    """Mirror analytical scene/fusion products into a predictable image archive."""
    root = Path(root)
    layout = ensure_storage_layout(root, cfg)
    archive = Path(layout['satellite_archive'])
    run_key = _run_key(run)
    archived_keys = _load_archive_index(archive)
    if run_key in archived_keys:
        return []
    run_stamp = _safe(str(run.get('finished_at') or _utc_stamp()).replace(':', '').replace('+', '_'))
    entries: list[dict] = []

    for row in run.get('scene_results', []) or []:
        satellite = _safe(row.get('satellite'))
        scene_id = _safe(row.get('scene_id'))
        dest = archive / satellite / run_stamp / scene_id
        stack_src = Path(str(row.get('stack_path') or ''))
        preview_src = Path(str(row.get('preview_path') or ''))
        stack_dst = _link_or_copy(stack_src, dest / 'analysis_stack.tif')
        preview_dst = _link_or_copy(preview_src, dest / 'preview.png')
        meta = {
            'kind': 'satellite_scene', 'archived_at': datetime.now(timezone.utc).isoformat(),
            'satellite': row.get('satellite'), 'scene_id': row.get('scene_id'),
            'collection': row.get('collection'), 'platform': row.get('platform'),
            'acquired_at': row.get('acquired_at'), 'native_gsd_m': row.get('native_gsd_m'),
            'cloud_cover_percent': row.get('cloud_cover_percent'), 'valid_percent': row.get('valid_percent'),
            'archive_stack': stack_dst, 'archive_preview': preview_dst,
            'source_stack': str(stack_src), 'source_preview': str(preview_src),
        }
        _atomic_json(dest / 'metadata.json', meta)
        append_jsonl(archive / 'image_log.jsonl', meta)
        _atomic_json(archive / f'latest_{satellite}.json', meta)
        entries.append(meta)

    fusion = run.get('fusion') or {}
    fusion_src = Path(str(fusion.get('fusion_path') or ''))
    preview_src = Path(str(fusion.get('preview_path') or ''))
    if fusion_src.is_file() or preview_src.is_file():
        dest = archive / 'fusion' / run_stamp
        fusion_dst = _link_or_copy(fusion_src, dest / 'multisatellite_fusion.tif')
        preview_dst = _link_or_copy(preview_src, dest / 'multisatellite_fusion_preview.png')
        meta = {
            'kind': 'multisatellite_fusion', 'archived_at': datetime.now(timezone.utc).isoformat(),
            'acquired_at': run.get('finished_at'),
            'available_satellites': fusion.get('available_satellites') or [],
            'acquisition_skew_hours': fusion.get('acquisition_skew_hours'),
            'archive_stack': fusion_dst, 'archive_preview': preview_dst,
            'source_stack': str(fusion_src), 'source_preview': str(preview_src),
        }
        _atomic_json(dest / 'metadata.json', meta)
        append_jsonl(archive / 'image_log.jsonl', meta)
        _atomic_json(archive / 'latest_fusion.json', meta)
        entries.append(meta)

    status = ensure_storage_layout(root, cfg)
    status.update(
        last_satellite_archive_at=datetime.now(timezone.utc).isoformat(),
        last_satellite_archive_entries=len(entries),
        image_log_exists=(archive / 'image_log.jsonl').exists(),
    )
    archived_keys.add(run_key)
    _save_archive_index(archive, archived_keys)
    _atomic_json(root / 'data/collection/storage_status.json', status)
    return entries


def backfill_multisatellite_archive(root: str | Path, cfg: Mapping, max_runs: int = 30) -> dict:
    """Archive outputs from older run folders after an upgrade."""
    root = Path(root)
    ensure_storage_layout(root, cfg)
    run_files = sorted((root / 'data/multisatellite/runs').glob('*/run.json'), key=lambda p: p.stat().st_mtime, reverse=True) if (root / 'data/multisatellite/runs').exists() else []
    archived = 0; errors = []
    for path in reversed(run_files[:max_runs]):
        try:
            run = json.loads(path.read_text(encoding='utf-8'))
            # Older run.json files may contain absolute paths from a previous
            # installation directory. Re-resolve the canonical run-local paths
            # when those originals no longer exist so copied data can be upgraded.
            for row in run.get('scene_results', []) or []:
                stack = Path(str(row.get('stack_path') or ''))
                preview = Path(str(row.get('preview_path') or ''))
                local = path.parent / _safe(row.get('satellite')) / _safe(row.get('scene_id'))
                if not stack.is_file() and (local / 'analysis_stack.tif').is_file():
                    row['stack_path'] = str(local / 'analysis_stack.tif')
                if not preview.is_file() and (local / 'preview.png').is_file():
                    row['preview_path'] = str(local / 'preview.png')
            fusion = run.get('fusion') or {}
            if not Path(str(fusion.get('fusion_path') or '')).is_file() and (path.parent / 'multisatellite_fusion.tif').is_file():
                fusion['fusion_path'] = str(path.parent / 'multisatellite_fusion.tif')
            if not Path(str(fusion.get('preview_path') or '')).is_file() and (path.parent / 'multisatellite_fusion_preview.png').is_file():
                fusion['preview_path'] = str(path.parent / 'multisatellite_fusion_preview.png')
            run['fusion'] = fusion
            archived += len(archive_multisatellite_run(root, cfg, run))
        except Exception as exc:
            errors.append(f'{path}: {type(exc).__name__}: {exc}')
    return {'runs_checked': min(len(run_files), max_runs), 'entries_archived': archived, 'errors': errors[-10:]}


def archive_inspected_scene(root: str | Path, cfg: Mapping, scene: Mapping, quality: Mapping) -> dict | None:
    """Archive manually imported/raw-inbox imagery after the imagery worker inspects it."""
    meta = scene.get('input_metadata') or {}
    if str(meta.get('source') or '').startswith('concurrent_multisatellite'):
        return None
    root = Path(root); layout = ensure_storage_layout(root, cfg); archive = Path(layout['satellite_archive'])
    dest = archive / 'library' / _safe(scene.get('id'))
    src = Path(str(scene.get('path') or ''))
    raster = _link_or_copy(src, dest / src.name) if src.is_file() else None
    preview_src = Path(str(quality.get('preview') or ''))
    preview = _link_or_copy(preview_src, dest / 'preview.png') if preview_src.is_file() else None
    record = {
        'kind': 'imagery_library_scene', 'archived_at': datetime.now(timezone.utc).isoformat(),
        'scene_id': scene.get('id'), 'filename': scene.get('filename'),
        'source': meta.get('source') or meta.get('acquisition_source') or 'local_inbox',
        'acquired_at': quality.get('acquired_at') or meta.get('acquired_at'),
        'archive_raster': raster, 'archive_preview': preview, 'crs': quality.get('crs'),
        'pixel_size': quality.get('pixel_size'), 'valid_percent_sampled': quality.get('valid_percent_sampled'),
    }
    marker = dest / 'metadata.json'
    if marker.exists():
        return record
    _atomic_json(marker, record); append_jsonl(archive / 'image_log.jsonl', record)
    return record


def archive_highres_result(root: str | Path, cfg: Mapping, source: str | Path,
                           result: Mapping | None = None, error: str | None = None) -> dict:
    root = Path(root)
    layout = ensure_storage_layout(root, cfg)
    source = Path(source)
    stamp = _utc_stamp()
    dest = Path(layout['highres_output']) / 'processed' / f'{stamp}_{_safe(source.stem)}'
    dest.mkdir(parents=True, exist_ok=True)

    archived_source = None
    if source.is_file():
        # Keep a durable evidence link where possible; if the filesystem refuses a
        # hardlink, retain the original path instead of blindly duplicating a huge file.
        try:
            archived_source = _link_or_copy(source, dest / source.name)
        except OSError:
            archived_source = str(source)

    mask_src = Path(str((result or {}).get('segmentation_mask_path') or ''))
    mask_dst = _link_or_copy(mask_src, dest / 'candidate_mask.png') if mask_src.is_file() else None
    record = {
        'kind': 'highres_processing', 'logged_at': datetime.now(timezone.utc).isoformat(),
        'source_path': str(source), 'archived_source': archived_source,
        'result_path': str(dest / 'result.json'), 'candidate_mask': mask_dst,
        'status': 'FAILED' if error else 'ANALYZED', 'error': error,
        'result': dict(result or {}),
    }
    _atomic_json(dest / 'result.json', record)
    append_jsonl(Path(layout['highres_log']), record)
    return record


def count_highres_inputs(root: str | Path, cfg: Mapping) -> int:
    root = Path(root)
    inbox = _local_path(root, cfg.get('geo_x_highres_inbox', 'highres_inbox'))
    if not inbox.exists():
        return 0
    return sum(1 for p in inbox.iterdir() if p.is_file() and p.suffix.lower() in {'.tif', '.tiff'})

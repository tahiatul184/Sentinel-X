"""Background concurrent multi-satellite collection and analysis service."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import threading
import time
import uuid

from collection_store import ROOT, Store, load_config, sha256
from collection_worker import worker_lock
from satellite_downloader import area_bbox, config_signature
from multisatellite_pipeline import run_concurrent_multisatellite, write_run_json
from imagery_archive import archive_multisatellite_run, ensure_storage_layout


class MultiSatelliteWorker:
    def __init__(self, root=ROOT, clock=time.time):
        self.root = Path(root)
        self.store = Store(self.root)
        self.clock = clock
        self.stopping = threading.Event()
        self.status_lock = threading.Lock()
        self.source_state = {}
        try:
            ensure_storage_layout(self.root, load_config(self.root))
        except Exception:
            pass

    def status(self, state, **fields):
        with self.status_lock:
            current = self.store.setting('multisatellite_download') or {}
            current.update(state=state, time=self.clock(), pid=os.getpid(), **fields)
            self.store.setting('multisatellite_download', current)
            return current

    def log(self, message):
        with self.store.db() as db:
            db.execute('INSERT INTO events(scene_id,time,stage,message) VALUES(?,?,?,?)',
                       (None, self.clock(), 'MULTISATELLITE', str(message)[:4000]))

    def event(self, payload):
        kind = payload.get('type')
        sat = payload.get('satellite')
        with self.status_lock:
            if sat:
                state = dict(self.source_state.get(sat, {}))
                if kind == 'search_finished':
                    state.update(stage='SEARCHED', matches=payload.get('count', 0), error=None)
                elif kind == 'scene_started':
                    state.update(stage='PROCESSING', scene=payload.get('scene'),
                                 acquired_at=payload.get('acquired_at'), error=None)
                elif kind == 'asset_read':
                    state.update(stage='STREAMING', scene=payload.get('scene'), asset=payload.get('asset'))
                elif kind == 'scene_finished':
                    state.update(stage='ANALYZED', scene=payload.get('scene'),
                                 elapsed_seconds=payload.get('elapsed_seconds'), asset=None)
                elif kind == 'source_error':
                    state.update(stage='ERROR', scene=payload.get('scene'), error=payload.get('error'))
                self.source_state[sat] = state
            current = self.store.setting('multisatellite_download') or {}
            current.update(time=self.clock(), sources=dict(self.source_state))
            if kind in {'scene_started', 'asset_read'}:
                current['state'] = 'CONCURRENT_PROCESSING'
            elif kind == 'run_finished':
                current.update(peak_workers=payload.get('peak_workers'), processed_scenes=payload.get('scene_count'))
            self.store.setting('multisatellite_download', current)

    def _register_outputs(self, run):
        registered = []
        for row in run.get('scene_results', []):
            path = Path(row['stack_path'])
            if not path.exists():
                continue
            metadata = {
                'source': 'concurrent_multisatellite', 'context_only': False,
                'satellite_source': row['satellite'], 'provider': 'Microsoft Planetary Computer STAC',
                'provider_scene_id': row['scene_id'], 'provider_collection': row.get('collection'),
                'platform': row.get('platform'), 'acquired_at': row.get('acquired_at'),
                'cloud_cover_percent': row.get('cloud_cover_percent'),
                'analysis_ready_stack': True,
                'enhanced_preview_path': row.get('enhanced_preview_path'),
                'processing_quality': row.get('processing_quality'),
            }
            ident, created = self.store.register(path, sha256(path), metadata)
            registered.append({'scene_id': ident, 'created': bool(created), 'path': str(path)})
        fusion = run.get('fusion', {})
        fpath = Path(fusion.get('fusion_path', ''))
        if fpath.is_file():
            metadata = {
                'source': 'concurrent_multisatellite_fusion', 'context_only': False,
                'provider': 'AEROSENTINEL multi-satellite fusion',
                'acquired_at': run.get('finished_at'), 'analysis_ready_stack': True,
                'satellite_source': 'multi-satellite',
            }
            ident, created = self.store.register(fpath, sha256(fpath), metadata)
            registered.append({'scene_id': ident, 'created': bool(created), 'path': str(fpath)})
        return registered

    def tick(self, cfg):
        if self.store.setting('paused') or not cfg['auto_collect_enabled']:
            self.status('PAUSED' if self.store.setting('paused') else 'DISABLED')
            return
        if not cfg.get('auto_location_ready', False):
            self.status('LOCATION_REQUIRED', next_check=None,
                        message='Choose or save a collection location before concurrent acquisition.')
            return

        signature = config_signature(cfg)
        previous = self.store.setting('multisatellite_download') or {}
        request = self.store.setting('multisatellite_request')
        force = bool(request and request != previous.get('handled_request'))
        if (not force and signature == previous.get('config_signature') and
                previous.get('state') not in {'STARTING', 'PAUSED', 'DISABLED', 'LOCATION_REQUIRED'} and
                self.clock() < float(previous.get('next_check') or 0)):
            return

        selected = list(cfg.get('multi_satellites') or ['sentinel-1', 'sentinel-2', 'landsat-8-9'])
        now = datetime.fromtimestamp(self.clock(), timezone.utc)
        start = now - timedelta(days=int(cfg['auto_lookback_days']))
        end = now
        stamp = now.strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        run_dir = self.root / 'data/multisatellite/runs' / stamp
        self.source_state = {key: {'stage': 'QUEUED'} for key in selected}
        self.status('SEARCHING', error=None, message='Searching selected satellite catalogs concurrently.',
                    config_signature=signature, handled_request=request,
                    last_check=self.clock(), next_check=None, sources=dict(self.source_state),
                    requested_satellites=selected, peak_workers=0, processed_scenes=0)
        try:
            run = run_concurrent_multisatellite(
                bbox=area_bbox(cfg), start_date=start.date(), end_date=end.date(),
                output_dir=run_dir, selected_satellites=selected,
                max_scenes_per_satellite=int(cfg.get('multi_scenes_per_satellite', 1)),
                max_cloud=float(cfg['auto_max_cloud']),
                workers=int(cfg.get('multi_concurrent_workers', 3)),
                grid_pixels=int(cfg.get('multi_grid_pixels', 512)),
                temporal_tolerance_hours=float(cfg.get('multi_temporal_tolerance_hours', 72)),
                event=self.event,
                cancelled=lambda: self.stopping.is_set() or bool(self.store.setting('paused')),
            )
            run_path = write_run_json(run, run_dir / 'run.json')
            registered = self._register_outputs(run)
            run['registered_outputs'] = registered
            try:
                archived = archive_multisatellite_run(self.root, cfg, run)
                run['imagery_archive'] = archived
            except Exception as exc:
                run['imagery_archive_error'] = f'{type(exc).__name__}: {exc}'
                self.log('Imagery archive/log update failed: ' + run['imagery_archive_error'])
            write_run_json(run, run_path)
            self.store.setting('multisatellite_last_result', run)
            failures = len(run.get('errors', {}))
            available = len(run.get('fusion', {}).get('available_satellites', []))
            if available >= 2:
                state = 'UP_TO_DATE' if failures == 0 else 'PARTIAL_SUCCESS'
                message = (f"Concurrent run complete: {available} satellite sources fused; "
                           f"peak {run['concurrency']['peak_scene_workers']} scene workers.")
            else:
                state = 'PARTIAL_SUCCESS'
                message = 'Run completed, but fewer than two satellite sources were available for fusion.'
            delay = float(cfg['auto_interval_minutes']) * 60
            self.status(state, message=message, error=' | '.join(run.get('errors', {}).values())[:2500] or None,
                        next_check=self.clock()+delay, last_download=self.clock(),
                        last_run_path=str(run_path), sources=dict(self.source_state),
                        peak_workers=run['concurrency']['peak_scene_workers'],
                        processed_scenes=len(run['scene_results']),
                        available_satellites=run['fusion']['available_satellites'],
                        acquisition_skew_hours=run['fusion']['acquisition_skew_hours'],
                        fusion_preview=run['fusion']['preview_path'])
            self.log(message)
        except Exception as exc:
            failures = int(previous.get('consecutive_failures', 0)) + 1
            delay = min(1800, 60 * (2 ** min(failures, 5)))
            self.status('ERROR', error=f'{type(exc).__name__}: {exc}',
                        message='Concurrent multi-satellite run failed; successful partial scene files are retained.',
                        next_check=self.clock()+delay, consecutive_failures=failures,
                        sources=dict(self.source_state))
            self.log(f'Concurrent multi-satellite failure: {type(exc).__name__}: {exc}')

    def run(self):
        with worker_lock(self.root/'data/collection/multisatellite_worker.lock'):
            self.status('STARTING', next_check=0)
            try:
                while not self.stopping.is_set():
                    try:
                        self.tick(load_config(self.root))
                    except Exception as exc:
                        self.status('ERROR', error=f'{type(exc).__name__}: {exc}', next_check=self.clock()+60)
                    self.stopping.wait(3)
            finally:
                self.status('STOPPED')


if __name__ == '__main__':
    folder = ROOT/'data/collection/logs'; folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[RotatingFileHandler(
        folder/'multisatellite.log', maxBytes=3_000_000, backupCount=2, encoding='utf-8')])
    worker = MultiSatelliteWorker()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: worker.stopping.set())
    worker.run()

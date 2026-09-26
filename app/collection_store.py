"""Persistent local collection state shared by the worker and operator dashboard."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "highres_auto_enabled": True, "highres_auto_infer": True,
    "highres_radius_km": 0.5, "highres_lookback_days": 90, "highres_max_cloud": 0.2,
    "watch_dir": "satellite_inbox", "archive_root": "data/imagery",
    "site_id": "", "sensor": "", "product": "", "acquisition_source": "Local GeoTIFF inbox",
    "settle_seconds": 5.0, "poll_seconds": 3.0, "min_free_gb": 1.0,
    "auto_collect_enabled": False,
    "ops_automation_enabled": True, "ops_weather_enabled": True, "ops_terrain_enabled": True,
    "ops_weather_interval_minutes": 5, "ops_weather_forecast_days": 3,
    "ops_auto_report_enabled": True, "ops_report_interval_minutes": 5, "ops_report_history_days": 7,
    "ops_profile": "airfield_resilience",
    "geo_x_enabled": True, "geo_x_abstain_threshold": 0.45,
    "geo_x_highres_enabled": True, "geo_x_highres_inbox": "highres_inbox",
    "auto_area_name": "Choose your location",
    "auto_latitude": 0.0, "auto_longitude": 0.0, "auto_radius_km": 5.0,
    "auto_interval_minutes": 30, "auto_lookback_days": 30,
    "auto_max_cloud": 70.0, "auto_max_scenes": 1,
    "auto_download_connections": 4,
    "multi_satellites": ["sentinel-1", "sentinel-2", "landsat-8-9"],
    "multi_concurrent_workers": 3, "multi_scenes_per_satellite": 2,
    "multi_grid_pixels": 512, "multi_temporal_tolerance_hours": 72.0,
    "auto_location_ready": False, "auto_location_source": "unset",
    "auto_location_accuracy_m": None, "auto_location_captured_at": None,
}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('w', encoding='utf-8') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def local_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_config(root: Path = ROOT):
    path = root / 'collection_config.json'
    cfg = dict(DEFAULTS)
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        cfg.update(saved)
        if 'auto_location_ready' not in saved:
            cfg['auto_location_ready'] = bool(saved.get('auto_latitude') is not None and
                saved.get('auto_longitude') is not None and
                'default' not in str(saved.get('auto_area_name', 'default')).lower())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    for key in ('highres_auto_enabled', 'highres_auto_infer'):
        if not isinstance(cfg.get(key, True), bool):
            raise ValueError(f'{key} must be true or false')
    for key, lo, hi in [('highres_radius_km',0.1,1), ('highres_lookback_days',1,365), ('highres_max_cloud',0,1)]:
        value = float(cfg.get(key, DEFAULTS[key]))
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be between {lo} and {hi}')
        if key == 'highres_lookback_days' and value != int(value):
            raise ValueError('High-resolution lookback must be a whole number of days')
    bounds = {'settle_seconds': (1, 300), 'poll_seconds': (1, 300),
              'min_free_gb': (0.01, 10000)}
    for key, (lo, hi) in bounds.items():
        value = float(cfg[key])
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be between {lo} and {hi}')
    for key in ['watch_dir', 'archive_root']:
        if not str(cfg[key]).strip():
            raise ValueError(f'{key} is required')
    for key, lo, hi in [('auto_latitude', -79, 83), ('auto_longitude', -179.8, 179.8),
                        ('auto_radius_km', 0.5, 20), ('auto_interval_minutes', 1, 1440),
                        ('auto_lookback_days', 1, 365), ('auto_max_cloud', 0, 100),
                        ('auto_max_scenes', 1, 5)]:
        value = float(cfg.get(key, DEFAULTS[key]))
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be between {lo} and {hi}')
        if key in ('auto_interval_minutes', 'auto_lookback_days', 'auto_max_scenes') and value != int(value):
            raise ValueError(f'{key} must be a whole number')
    if cfg.get('auto_download_connections', 4) not in (1, 2, 4):
        raise ValueError('Download connections must be 1, 2 or 4.')
    allowed_satellites = {'sentinel-1', 'sentinel-2', 'landsat-8-9'}
    selected = cfg.get('multi_satellites', list(allowed_satellites))
    if not isinstance(selected, list) or len(selected) < 2 or len(selected) > 3 or any(v not in allowed_satellites for v in selected):
        raise ValueError('Select two or three supported concurrent satellite sources.')
    for key, lo, hi in [('multi_concurrent_workers', 2, 8), ('multi_scenes_per_satellite', 1, 3),
                        ('multi_grid_pixels', 128, 1024), ('multi_temporal_tolerance_hours', 1, 720)]:
        value = float(cfg.get(key, DEFAULTS[key]))
        if not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f'{key} must be between {lo} and {hi}')
        if key in ('multi_concurrent_workers', 'multi_scenes_per_satellite', 'multi_grid_pixels') and value != int(value):
            raise ValueError(f'{key} must be a whole number')
    if not isinstance(cfg.get('auto_location_ready', False), bool):
        raise ValueError('Location readiness must be true or false')
    if not isinstance(cfg.get('auto_collect_enabled', False), bool):
        raise ValueError('Automatic collection must be enabled or disabled')
    if not isinstance(cfg.get('ops_automation_enabled', True), bool):
        raise ValueError('Operations automation must be enabled or disabled')
    if not isinstance(cfg.get('ops_weather_enabled', True), bool):
        raise ValueError('Weather automation must be enabled or disabled')
    if not isinstance(cfg.get('ops_terrain_enabled', True), bool):
        raise ValueError('Terrain automation must be enabled or disabled')
    if not isinstance(cfg.get('ops_auto_report_enabled', True), bool):
        raise ValueError('Automatic report export must be enabled or disabled')
    for key, lo, hi in [('ops_weather_interval_minutes', 5, 1440), ('ops_weather_forecast_days', 1, 7),
                        ('ops_report_interval_minutes', 1, 1440), ('ops_report_history_days', 1, 90)]:
        value=float(cfg.get(key, DEFAULTS[key]))
        if not math.isfinite(value) or not lo <= value <= hi or value != int(value):
            raise ValueError(f'{key} must be a whole number between {lo} and {hi}')
    if cfg.get('ops_profile','airfield_resilience') not in {'airfield_resilience','disaster_response','infrastructure_resilience','balanced'}:
        raise ValueError('Unknown automation profile')
    if not isinstance(cfg.get('geo_x_enabled', True), bool):
        raise ValueError('AEROSENTINEL must be enabled or disabled')
    if not isinstance(cfg.get('geo_x_highres_enabled', True), bool):
        raise ValueError('High-resolution research inbox must be enabled or disabled')
    if not str(cfg.get('geo_x_highres_inbox','highres_inbox')).strip():
        raise ValueError('High-resolution inbox path is required')



def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()




class Store:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.path = self.root / 'data/collection/collection.sqlite3'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS scenes (
                    id TEXT PRIMARY KEY, path TEXT NOT NULL, filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL, received REAL NOT NULL, updated REAL NOT NULL,
                    state TEXT NOT NULL, payload TEXT NOT NULL DEFAULT '{}');
                CREATE INDEX IF NOT EXISTS scenes_updated ON scenes(updated DESC);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, scene_id TEXT, time REAL,
                    stage TEXT, message TEXT);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
            ''')

    @contextlib.contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA busy_timeout=15000')
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, path, digest, metadata):
        # Upload filename and transport provenance must not defeat deduplication.
        identity_metadata = {'acquired_at':metadata.get('acquired_at') or None}
        ident = hashlib.sha256((digest + json.dumps(identity_metadata, sort_keys=True)).encode()).hexdigest()[:24]
        now = time.time()
        with self.db() as db:
            created = db.execute('INSERT OR IGNORE INTO scenes VALUES (?,?,?,?,?,?,?,?)',
                (ident, str(path), path.name, digest, now, now, 'QUEUED', json.dumps({'input_metadata': metadata}))).rowcount
            if created:
                db.execute('INSERT INTO events(scene_id,time,stage,message) VALUES(?,?,?,?)',
                           (ident, now, 'RECEIVED', 'Stable GeoTIFF registered from local inbox'))
        return ident, bool(created)

    def get(self, ident):
        with self.db() as db:
            row = db.execute('SELECT * FROM scenes WHERE id=?', (ident,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result.update(json.loads(result.pop('payload')))
        return result

    def update(self, ident, state, message='', **fields):
        with self.db() as db:
            row = db.execute('SELECT payload FROM scenes WHERE id=?', (ident,)).fetchone()
            if row is None:
                raise KeyError(ident)
            payload = json.loads(row['payload'])
            payload.update(fields)
            now = time.time()
            db.execute('UPDATE scenes SET state=?,updated=?,payload=? WHERE id=?',
                       (state, now, json.dumps(payload, allow_nan=False), ident))
            if message:
                db.execute('INSERT INTO events(scene_id,time,stage,message) VALUES(?,?,?,?)', (ident, now, state, message))

    def scenes(self, limit=500):
        with self.db() as db:
            rows = db.execute('SELECT * FROM scenes ORDER BY received DESC LIMIT ?', (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item.update(json.loads(item.pop('payload')))
            result.append(item)
        return result

    def counts(self):
        with self.db() as db:
            return dict(db.execute('SELECT state,count(*) FROM scenes GROUP BY state').fetchall())

    def pending(self):
        with self.db() as db:
            return [r[0] for r in db.execute("SELECT id FROM scenes WHERE state='QUEUED' ORDER BY received")]

    def events(self, ident=None, limit=200):
        with self.db() as db:
            if ident:
                rows = db.execute('SELECT time,stage,message FROM events WHERE scene_id=? ORDER BY id DESC LIMIT ?', (ident, limit)).fetchall()
            else:
                rows = db.execute('SELECT time,stage,message,scene_id FROM events ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        return [dict(r) for r in rows]

    def setting(self, key, value=None):
        with self.db() as db:
            if value is not None:
                db.execute('INSERT OR REPLACE INTO settings VALUES (?,?)', (key, json.dumps(value)))
            row = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
            return json.loads(row[0]) if row else None

    def retry(self, ident):
        scene = self.get(ident)
        if not scene or scene['state'] not in {'FAILED', 'QUALITY_BLOCKED', 'SUBMIT_FAILED'}:
            raise ValueError('Only failed or quality-blocked scenes can be retried')
        self.update(ident, 'QUEUED', 'Operator requested retry', error=None)

    def recover(self):
        for scene in self.scenes(limit=100000):
            if scene['state'] in {'INSPECTING', 'DETECTING', 'ARCHIVING', 'SUBMITTING'}:
                if scene['state'] == 'SUBMITTING':
                    self.update(scene['id'], 'SUBMIT_FAILED', 'Interrupted API delivery; inspect API audit before retrying',
                                error='Delivery acknowledgement may have been lost. Retry can duplicate the last API event.')
                else:
                    self.update(scene['id'], 'QUEUED', 'Resuming interrupted processing after restart')

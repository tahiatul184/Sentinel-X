"""Public Sentinel-2 area imagery: scheduled STAC search and native COG crops.

This downloads already acquired imagery for site context and human review.
All collected products are explicitly marked context_only.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import re
import shutil
import signal
import threading
from contextlib import contextmanager
import time
from urllib.parse import urlparse

import requests

from collection_store import ROOT, Store, atomic_json, load_config, local_path, sha256
from resumable_transfer import TransferPaused
from parallel_transfer import transfer_tiff, discard_transfer, has_transfer

SEARCH_URL = 'https://earth-search.aws.element84.com/v1/search'
COLLECTIONS = ['sentinel-2-c1-l2a', 'sentinel-2-l2a']
ASSET_HOSTS = {'sentinel-cogs.s3.us-west-2.amazonaws.com',
               'e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com',
               'sentinel-cogs.s3.amazonaws.com',
               'e84-earth-search-sentinel-data.s3.amazonaws.com'}


def area_bbox(cfg):
    lat, lon = float(cfg['auto_latitude']), float(cfg['auto_longitude'])
    dy = float(cfg['auto_radius_km']) / 111.32
    dx = dy / math.cos(math.radians(lat))
    box = [lon-dx, lat-dy, lon+dx, lat+dy]
    if box[0] < -180 or box[2] > 180 or box[1] < -80 or box[3] > 84:
        raise ValueError('Choose a smaller area away from the date line and polar limits.')
    return box


def config_signature(cfg):
    values = {k: v for k, v in cfg.items() if k.startswith('auto_') or k.startswith('multi_')}
    values['watch_dir'] = cfg['watch_dir']
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def acquired_time(item):
    raw = item['properties']['datetime']
    value = datetime.fromisoformat(raw.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('Provider acquisition time has no timezone.')
    return value.astimezone(timezone.utc)


class SearchRejected(RuntimeError):
    """A provider HTTP rejection, distinct from a missing internet connection."""


def check_search_response(response):
    if response.status_code >= 400:
        try:
            detail = response.json()
            if isinstance(detail, dict):
                detail = detail.get('description') or detail.get('message') or detail.get('detail') or detail
            detail = json.dumps(detail, ensure_ascii=False) if not isinstance(detail, str) else detail
        except (ValueError, TypeError):
            detail = response.text
        detail = ' '.join(str(detail).split())[:1600]
        raise SearchRejected(f'Satellite provider HTTP {response.status_code}: {detail or "No error detail supplied"}')


def search_scenes(cfg, session, now=None):
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=int(cfg['auto_lookback_days']))
    bbox = area_bbox(cfg)
    # Use the STAC POST representation: arrays/objects remain typed JSON,
    # avoiding provider-specific query-string parsing of collections and sortby.
    body = {
        'collections': list(COLLECTIONS),
        'bbox': [round(v, 7) for v in bbox],
        'datetime': start.isoformat(timespec='seconds').replace('+00:00', 'Z') + '/' + now.isoformat(timespec='seconds').replace('+00:00', 'Z'),
        'query': {'eo:cloud_cover': {'lte': float(cfg['auto_max_cloud'])}},
        'sortby': [{'field': 'datetime', 'direction': 'desc'}],
        'limit': 100,
    }
    response = session.post(SEARCH_URL, json=body, timeout=(10, 30))
    if response.status_code == 400:
        # Some deployments reject optional query/sort extensions. Retry a core
        # STAC search and filter/sort locally, following bounded pagination.
        core = {k: v for k, v in body.items() if k not in {'query', 'sortby'}}
        response = session.post(SEARCH_URL, json=core, timeout=(10, 30))
        body = core
    check_search_response(response)
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get('features'), list):
        raise ValueError('The provider returned an invalid STAC response.')
    features = list(payload['features'])
    # With local cloud filtering, later pages can contain the usable scenes.
    # Only same-provider pagination is accepted; no arbitrary response URLs.
    seen_pages = set()
    for _ in range(9 if 'query' not in body else 0):
        link = next((v for v in payload.get('links', []) if v.get('rel') == 'next'), None)
        if not link:
            break
        url = link.get('href', '')
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname != 'earth-search.aws.element84.com' or parsed.path != '/v1/search':
            raise ValueError('Provider pagination points to an unsupported search endpoint.')
        method = link.get('method', 'GET').upper()
        page_body = dict(body, **link.get('body', {})) if link.get('merge') else link.get('body', body)
        identity = json.dumps([method, url, page_body], sort_keys=True)
        if identity in seen_pages:
            raise ValueError('Provider pagination repeated a page; narrow the search window.')
        seen_pages.add(identity)
        if method == 'POST':
            response = session.post(url, json=page_body, timeout=(10, 30))
        elif method == 'GET':
            response = session.get(url, timeout=(10, 30))
        else:
            raise ValueError('Provider returned an unsupported pagination method.')
        check_search_response(response)
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get('features'), list):
            raise ValueError('The provider returned an invalid STAC page.')
        features.extend(payload['features'])
    if 'query' not in body and any(v.get('rel') == 'next' for v in payload.get('links', [])):
        raise ValueError('The compatibility search exceeded 1000 records. Reduce lookback days or area size.')
    scenes = []
    seen = set()
    for item in features:
        try:
            props = item['properties']
            acquired = acquired_time(item)
            cloud = float(props['eo:cloud_cover'])
            bounds = item['bbox']
            if len(bounds) == 6:
                bounds = [bounds[0], bounds[1], bounds[3], bounds[4]]
            if (item.get('collection') not in COLLECTIONS or
                    not start <= acquired <= now or not math.isfinite(cloud) or
                    not 0 <= cloud <= float(cfg['auto_max_cloud']) or
                    len(bounds) != 4 or bounds[0] >= bbox[2] or bounds[2] <= bbox[0] or
                    bounds[1] >= bbox[3] or bounds[3] <= bbox[1]):
                continue
            # C1 and legacy collections may publish the same acquisition.
            key = (props.get('s2:product_uri') or props.get('sentinel:product_id') or item['id'])
            if key in seen:
                continue
            seen.add(key)
            scenes.append(item)
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(scenes, key=acquired_time, reverse=True)


def visual_url(item):
    assets = item.get('assets', {})
    asset = assets.get('visual') or assets.get('tci')
    if not asset:
        asset = next((a for a in assets.values() if
                      'visual' in a.get('roles', []) and '.tif' in a.get('href', '').lower()), None)
    if not asset:
        raise ValueError('This provider scene has no downloadable true-color GeoTIFF asset.')
    url = asset['href']
    parsed = urlparse(url)
    if (parsed.scheme != 'https' or parsed.hostname not in ASSET_HOSTS or
            parsed.username or parsed.password or parsed.port not in (None, 443)):
        raise ValueError('Provider asset is not on a supported public Sentinel-2 host.')
    return url


class RemoteRasterReadError(RuntimeError):
    """Only opening or reading a remote raster can trigger the transfer fallback."""


def exception_detail(exc):
    parts, seen = [], set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append(f'{type(exc).__name__}: {exc}')
        exc = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return ' <- '.join(parts)[:2400]


@contextmanager
def open_source(opener, source, remote):
    try:
        dataset = opener(source)
    except Exception as exc:
        if remote:
            raise RemoteRasterReadError(exception_detail(exc)) from exc
        raise
    with dataset as src:
        yield src


def download_crop(item, bbox, output, cfg, *, raster_open=None, session=None,
                  progress=None, cancelled=None):
    progress = progress or (lambda **fields: None)
    cancelled = cancelled or (lambda: False)
    output = Path(output)
    url = visual_url(item)
    # The cache stays inside this installation's inbox, outside its top-level
    # GeoTIFF scan. The URL key is independent of the selected crop coordinates.
    cache = output.parent/'satellite_transfer_cache'
    source = cache/(hashlib.sha256(url.encode()).hexdigest() + '.tif.download')
    remote_error = None
    if not has_transfer(source):
        try:
            return crop_source(item, bbox, output, cfg, url, remote=True, raster_open=raster_open)
        except RemoteRasterReadError as exc:
            remote_error = exc
            logging.warning('Remote raster read failed; using resumable HTTPS transfer: %s', exception_detail(exc))
            output.unlink(missing_ok=True)
    owned = session is None
    session = session or requests.Session()
    try:
        transfer_tiff(url, source, cfg, session, progress, cancelled)
        result = crop_source(item, bbox, output, cfg, source, remote=False, raster_open=raster_open)
        result['transfer_method'] = 'resumable_https_then_local_crop'
        # Keep complete source until the area crop succeeds, then release space.
        discard_transfer(source)
        return result
    except TransferPaused:
        raise
    except Exception as fallback_error:
        detail = 'Image download/crop failed: ' + exception_detail(fallback_error)[:1600]
        if remote_error:
            detail += ' | Earlier remote-read error: ' + exception_detail(remote_error)[:600]
        raise RuntimeError(detail) from fallback_error
    finally:
        if owned:
            session.close()


def crop_source(item, bbox, output, cfg, source, *, remote, raster_open=None):
    """Read only intersecting COG blocks; preserve original pixel grid and CRS."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window, from_bounds, transform as window_transform
    from rasterio.warp import transform_bounds
    from rasterio.enums import ColorInterp
    raster_open = raster_open or rasterio.open
    url = visual_url(item)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                      GDAL_HTTP_CONNECTTIMEOUT='10', GDAL_HTTP_TIMEOUT='30',
                      GDAL_HTTP_MAX_RETRY='1', GDAL_HTTP_RETRY_DELAY='2',
                      AWS_NO_SIGN_REQUEST='YES', GDAL_TIFF_INTERNAL_MASK=True):
        with open_source(raster_open, source, remote) as src:
            if not src.crs or src.count < 3:
                raise ValueError('Provider raster has no CRS or has fewer than three color bands.')
            projected = transform_bounds('EPSG:4326', src.crs, *bbox, densify_pts=21)
            requested = from_bounds(*projected, transform=src.transform)
            left, top = math.floor(requested.col_off), math.floor(requested.row_off)
            right = math.ceil(requested.col_off + requested.width)
            bottom = math.ceil(requested.row_off + requested.height)
            window = Window(left, top, right-left, bottom-top).intersection(Window(0, 0, src.width, src.height))
            if window.width * window.height > 25_000_000:
                raise ValueError('Requested crop is too large. Reduce the collection radius.')
            reserve = float(cfg['min_free_gb']) * 1024**3 + window.width * window.height * 32
            if shutil.disk_usage(output.parent).free < reserve:
                raise RuntimeError('Not enough free storage for satellite download.')
            try:
                values = src.read([1, 2, 3], window=window, masked=True)
            except Exception as exc:
                if remote:
                    raise RemoteRasterReadError(exception_detail(exc)) from exc
                raise
            valid = ~np.any(np.ma.getmaskarray(values), axis=0)
            if not valid.any():
                raise ValueError('This satellite tile has no valid pixels inside the selected area.')
            profile = dict(driver='GTiff', width=int(window.width), height=int(window.height),
                           count=3, dtype=values.dtype, crs=src.crs,
                           transform=window_transform(window, src.transform),
                           compress='deflate', tiled=True, blockxsize=256, blockysize=256)
            coverage = min(100.0, float(window.width * window.height / (requested.width * requested.height) * 100))
            with rasterio.open(output, 'w', **profile) as dest:
                dest.write(values.filled(0))
                dest.write_mask(valid.astype('uint8') * 255)
                dest.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                dest.update_tags(ACQUISITION_TIME=acquired_time(item).isoformat(),
                                 CLOUD_COVER=str(item['properties']['eo:cloud_cover']),
                                 COLLECTION_MODE='context_only', PROVIDER='Earth Search / Copernicus Sentinel-2',
                                 PROVIDER_SCENE_ID=item['id'], PROVIDER_COLLECTION=item['collection'])
    return {'area_coverage_percent': round(coverage, 1), 'asset_url': url}


class Downloader:
    def __init__(self, root=ROOT, *, session=None, cropper=download_crop, clock=time.time):
        self.root = Path(root)
        self.store = Store(self.root)
        self.session = session or requests.Session()
        self.session.headers.update({'User-Agent': 'Satellite-Collection-Desk/1.8.5', 'Accept': 'application/json'})
        self.cropper, self.clock = cropper, clock
        self.stopping = threading.Event()

    def status(self, state, **fields):
        current = self.store.setting('satellite_download') or {}
        current.update(state=state, time=self.clock(), pid=os.getpid(), **fields)
        self.store.setting('satellite_download', current)
        return current

    def log(self, message):
        with self.store.db() as db:
            db.execute('INSERT INTO events(scene_id,time,stage,message) VALUES(?,?,?,?)',
                       (None, self.clock(), 'SATELLITE_DOWNLOAD', message))

    def tick(self, cfg):
        if self.store.setting('paused') or not cfg['auto_collect_enabled']:
            self.status('PAUSED' if self.store.setting('paused') else 'DISABLED')
            return
        if not cfg.get('auto_location_ready', False):
            self.status('LOCATION_REQUIRED', next_check=None, current_scene=None, error=None,
                        message='Choose your current location or save coordinates in Collect images.')
            return
        signature = config_signature(cfg)
        previous = self.store.setting('satellite_download') or {}
        request = self.store.setting('satellite_download_request')
        force = request and request != previous.get('handled_request')
        if (not force and previous.get('state') not in {'PAUSED', 'DISABLED', 'STARTING', 'LOCATION_REQUIRED'} and
                signature == previous.get('config_signature') and
                self.clock() < previous.get('next_check', 0)):
            self.status(previous.get('state', 'WAITING'))
            return
        self.status('SEARCHING', error=None, current_scene=None, last_check=self.clock(),
                    config_signature=signature, handled_request=request,
                    area=cfg['auto_area_name'], next_check=None, matched=0, collected_last_check=0)
        errors, collected, matched = [], 0, 0
        try:
            scenes = search_scenes(cfg, self.session, datetime.fromtimestamp(self.clock(), timezone.utc))
            matched = len(scenes)
            box = area_bbox(cfg)
            inbox = local_path(self.root, cfg['watch_dir'])
            inbox.mkdir(parents=True, exist_ok=True)
            latest_window = scenes[:int(cfg['auto_max_scenes'])]
            # Only the newest N matching scenes are eligible. Scheduled polls
            # do not silently backfill older scenes after downloading these.
            for item in latest_window:
                if self.stopping.is_set() or self.store.setting('paused'):
                    break
                fresh = load_config(self.root) if (self.root/'collection_config.json').exists() else cfg
                if config_signature(fresh) != signature:
                    self.status('WAITING', next_check=0, message='Collection settings changed; restarting search.')
                    return
                key = hashlib.sha256(json.dumps([item['collection'], item['id'], box], sort_keys=True).encode()).hexdigest()[:24]
                record = self.store.setting('satellite_item:' + key) or {}
                if record.get('path') and Path(record['path']).is_file():
                    continue
                name = re.sub(r'[^A-Za-z0-9_.-]', '_', item['id'])[:90]
                target = inbox / f'S2_{name}_{key[:10]}.tif'
                sidecar = target.with_suffix('.tif.json')
                partial = target.with_suffix('.tif.part')
                self.status('DOWNLOADING', current_scene=item['id'], matched=matched,
                            downloaded_bytes=0, total_bytes=None, speed_bps=0, eta_seconds=None,
                            transfer_attempt=0, resumed_bytes=0, transfer_phase='remote', download_connections=0,
                            message='Downloading the selected area at native resolution.')
                try:
                    # Atomic publication + embedded provenance allow restart
                    # recovery even if a process exits before updating SQLite.
                    if target.exists() and sidecar.exists():
                        metadata = json.loads(sidecar.read_text(encoding='utf-8'))
                    else:
                        if self.cropper is download_crop:
                            extra = self.cropper(item, box, partial, cfg, session=self.session,
                                progress=lambda **fields: self.status('DOWNLOADING', **fields),
                                cancelled=lambda: self.stopping.is_set() or bool(self.store.setting('paused')))
                        else:
                            extra = self.cropper(item, box, partial, cfg)
                        metadata = dict(source='earth_search_sentinel2', context_only=True,
                            provider='Earth Search / Copernicus Sentinel-2', provider_scene_id=item['id'],
                            provider_collection=item['collection'], acquired_at=acquired_time(item).isoformat(),
                            cloud_cover_percent=item['properties']['eo:cloud_cover'],
                            area_name=cfg['auto_area_name'], requested_bbox=box,
                            downloaded_at=datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(),
                            platform=item['properties'].get('platform', 'Sentinel-2'), **extra)
                        atomic_json(sidecar, metadata)
                        os.replace(partial, target)
                    ident, _ = self.store.register(target, sha256(target), metadata)
                    self.store.setting('satellite_item:' + key, {'path': str(target), 'scene_id': ident})
                    collected += 1
                    self.status('DOWNLOADING', last_download=self.clock(), last_scene=item['id'])
                    self.log(f"Collected {item['id']} for {cfg['auto_area_name']}; acquired {metadata['acquired_at']}.")
                except TransferPaused:
                    self.status('PAUSED' if self.store.setting('paused') else 'STOPPED',
                                message='Download paused. Progress is saved for the next launch or resume.',
                                next_check=0, error=None)
                    return
                except Exception as exc:
                    errors.append(f"{item['id']}: {exception_detail(exc)}")
                    logging.exception("Satellite imagery download failed for %s", item["id"])
                    self.log(errors[-1])
                finally:
                    partial.unlink(missing_ok=True)
            if not matched:
                message = 'No scenes match this area, time range and cloud limit. Increase lookback days or cloud limit.'
                state = 'NO_MATCHES'
            elif errors:
                message, state = 'Download interrupted; retry is scheduled. Valid saved bytes will be reused.', 'ERROR'
            elif collected:
                message, state = f'{collected} satellite scene(s) downloaded; waiting for new acquisitions.', 'WAITING'
            else:
                message, state = 'The newest matching scenes are already collected. Waiting for new acquisitions.', 'UP_TO_DATE'
            failures = int(previous.get('consecutive_failures', 0)) + 1 if errors else 0
            delay = min(300, 15 * (2 ** min(failures-1, 5))) if errors else float(cfg['auto_interval_minutes']) * 60
            self.status(state, error=' | '.join(errors)[:2500] or None, message=message,
                        matched=matched, collected_last_check=collected, current_scene=None,
                        consecutive_failures=failures, next_check=self.clock()+delay,
                        last_search_success=self.clock())
        except Exception as exc:
            failures = int(previous.get('consecutive_failures', 0)) + 1
            delay = min(1800, 60 * (2 ** min(failures, 5)))
            self.status('ERROR', error=str(exc)[:2500], current_scene=None,
                        message=('The satellite provider rejected the search. See its response below; automatic retry is scheduled.'
                                 if isinstance(exc, SearchRejected) else
                                 'Cannot reach or read the satellite archive. See details below; automatic retry is scheduled.'),
                        consecutive_failures=failures, next_check=self.clock()+delay)
            self.log(f'Satellite search failed: {exc}')

    def run(self):
        from collection_worker import worker_lock
        with worker_lock(self.root/'data/collection/satellite_downloader.lock'):
            # A crash during HTTP/raster work should be retried on relaunch.
            self.status('STARTING', next_check=0)
            try:
                while not self.stopping.is_set():
                    try:
                        self.tick(load_config(self.root))
                    except Exception as exc:
                        self.status('ERROR', error=str(exc), message='Check collection settings.')
                    self.stopping.wait(3)
            finally:
                self.status('STOPPED')
                self.session.close()


if __name__ == '__main__':
    folder = ROOT/'data/collection/logs'
    folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[RotatingFileHandler(
        folder/'satellite_download.log', maxBytes=2_000_000, backupCount=2, encoding='utf-8')])
    downloader = Downloader()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: downloader.stopping.set())
    downloader.run()

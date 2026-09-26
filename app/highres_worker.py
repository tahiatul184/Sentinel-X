"""Scheduled high-resolution archive download and optical aircraft inference."""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import signal
import shutil
import threading
import time

from collection_store import ROOT, Store, atomic_json, load_config
from highres_collection import PlanetArchive, ProviderError, location_aoi, site_key, download_asset, crop_rgb


class HighresWorker:
    def __init__(self, root=ROOT, clock=time.time):
        self.root = Path(root)
        self.store = Store(self.root)
        self.clock = clock
        self.stopping = threading.Event()

    def status(self, state, **fields):
        # Replace rather than merge: an old location/scene cannot masquerade as current.
        self.store.setting('highres_collection_status', dict(state=state, updated_at=self.clock(), **fields))

    def tick(self, cfg):
        if self.store.setting('paused') or not cfg.get('highres_auto_enabled', True):
            self.status('PAUSED')
            return
        if not cfg.get('auto_location_ready'):
            self.status('LOCATION_REQUIRED')
            return
        key = os.environ.get('PL_API_KEY','').strip()
        if not key:
            self.status('CREDENTIAL_REQUIRED', message='Set PL_API_KEY with SkySat archive download access, then restart the app.')
            return
        site = site_key(cfg)
        request = self.store.setting('highres_request')
        fingerprint = hashlib.sha256(repr((site, cfg.get('highres_lookback_days',90), cfg.get('highres_max_cloud',0.2),
                                           cfg.get('highres_auto_infer',True), request)).encode()).hexdigest()
        previous = self.store.setting('highres_collection_status') or {}
        if previous.get('fingerprint') == fingerprint and previous.get('next_check',0) > self.clock():
            return
        common = dict(fingerprint=fingerprint, site_id=site,
                      latitude=cfg['auto_latitude'], longitude=cfg['auto_longitude'])
        self.status('SEARCHING', **common)
        client = PlanetArchive(key)
        try:
            items = client.search(cfg)
            rejected = self.store.setting('highres_rejected_'+site) or []
            items = [item for item in items if str(item.get('id')) not in rejected]
            if not items:
                self.status('NO_ACCESSIBLE_COVERAGE', **common, next_check=self.clock()+1800,
                            message='No usable entitled SkySat RGB scene met the location, date, cloud and raster checks in the latest 50 catalog matches.')
                return
            item = items[0]
            scene = str(item['id'])
            props = item.get('properties') or {}
            acquired = datetime.fromisoformat(str(props['acquired']).replace('Z','+00:00'))
            if acquired.tzinfo is None:
                raise ValueError('Provider acquisition time requires timezone.')
            common.update(scene_id=scene, acquired_at=acquired.isoformat(),
                          acquisition_age_hours=max(0,(self.clock()-acquired.timestamp())/3600))
            token = hashlib.sha256((site+scene).encode()).hexdigest()[:24]
            folder = self.root/'data/highres_auto'/token
            folder.mkdir(parents=True, exist_ok=True)
            crop = folder/'rgb.tif'
            metadata = folder/'source.json'
            if not crop.exists() or not metadata.exists():
                location = client.active_asset(item)
                if not location:
                    self.status('ACTIVATING', **common, next_check=self.clock()+30)
                    return
                self.status('DOWNLOADING', **common)
                if shutil.disk_usage(folder).free < 1.25 * 1024**3:
                    raise ProviderError('At least 1.25 GB free disk space is required for an archive download.')
                original = folder/'provider.tif'
                download_asset(location, original)
                try:
                    quality = crop_rgb(original, crop, cfg)
                    atomic_json(metadata, dict(provider='Planet', item_type='SkySatCollect', asset='ortho_visual',
                        scene_id=scene, acquired_at=acquired.isoformat(), site_id=site,
                        bbox=location_aoi(cfg)[0], scene_cloud_fraction=props.get('cloud_cover'), **quality))
                except ValueError:
                    self.store.setting('highres_rejected_'+site, (rejected+[scene])[-200:])
                    raise
                finally:
                    original.unlink(missing_ok=True)
            import json
            info = json.loads(metadata.read_text())
            common.update(path=str(crop), product_pixel_spacing_m=info['product_pixel_spacing_m'],
                          full_aoi_valid=info['full_aoi_valid'], location_pixel_valid=info['location_pixel_valid'])
            if cfg.get('highres_auto_infer', True):
                result_path = folder/'detector.json'
                if not result_path.exists():
                    self.status('DETECTING', **common)
                    from aircraft_detector import infer_geotiff
                    result = infer_geotiff(crop, site_id=site, scene_id=scene, acquired_at=acquired.isoformat(),
                                           cloud_fraction=float(props.get('cloud_cover') or 0))
                    atomic_json(result_path, result)
                else:
                    result = json.loads(result_path.read_text())
                # A relocated study area has a separate history and output.
                history = self.store.setting('highres_rows_'+site) or []
                history = [r for r in history if r['scene_id'] != scene] + result['rows']
                history = history[-10000:]
                self.store.setting('highres_rows_'+site, history)
                from aircraft_awareness import analyze
                assessment = analyze(history)
                assessment.update(source='automatic_skysat_collection', site_id=site,
                                  analyzed_at=datetime.now(timezone.utc).isoformat(), latest_scene=result['state'],
                                  source_scene_id=scene, source_acquired_at=acquired.isoformat(),
                                  model_id=result['model_id'], model_sha256=result['model_sha256'],
                                  input_sha256=result['source_sha256'], product_metadata=info)
                self.store.setting('aircraft_awareness_latest', assessment)
                common.update(detector_state=result['state'], detections=result['detections'])
            self.status('UP_TO_DATE', **common, next_check=self.clock()+1800)
        except ProviderError as exc:
            self.status('PROVIDER_ERROR', **common, message=str(exc), next_check=self.clock()+300)
        except Exception as exc:
            # Requests/GDAL exceptions may contain signed URLs. Never persist them.
            self.status('ERROR', **common, error_type=type(exc).__name__,
                        message='Collection or inference failed. Check access, disk space, RGB coverage and model dependencies; the worker will retry.',
                        next_check=self.clock()+300)
        finally:
            client.session.close()

    def run(self):
        from collection_worker import worker_lock
        with worker_lock(self.root/'data/collection/highres_worker.lock'):
            while not self.stopping.is_set():
                try:
                    self.tick(load_config(self.root))
                except Exception as exc:
                    self.status('CONFIG_ERROR', error_type=type(exc).__name__)
                self.stopping.wait(5)


if __name__ == '__main__':
    worker = HighresWorker()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: worker.stopping.set())
    worker.run()

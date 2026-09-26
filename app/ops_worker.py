"""Fully automated weather, resilience and AEROSENTINEL report worker."""
from __future__ import annotations

from datetime import datetime, timezone
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import signal
import threading
import time

from collection_store import ROOT, Store, load_config
from collection_worker import worker_lock
from trust_core import ResearchStore
from weather_service import fetch_weather
from terrain_service import fetch_terrain
from satellite_downloader import area_bbox
from resilience_ops import compute_resilience_assessment, write_briefing
from direct_candidate import scan_inbox
from aerosentinel import build_observation, run_aerosentinel, latest_terramind_record


class OpsWorker:
    def __init__(self, root=ROOT, clock=time.time, weather_fetcher=fetch_weather):
        self.root = Path(root)
        self.store = Store(self.root)
        self.research = ResearchStore(self.root)
        self.clock = clock
        self.weather_fetcher = weather_fetcher
        self.stopping = threading.Event()

    def status(self, state, **fields):
        current = self.store.setting('ops_status') or {}
        current.update(state=state, time=self.clock(), pid=os.getpid(), **fields)
        self.store.setting('ops_status', current)
        return current

    def log(self, message):
        with self.store.db() as db:
            db.execute('INSERT INTO events(scene_id,time,stage,message) VALUES(?,?,?,?)',
                       (None, self.clock(), 'OPS-AUTO', str(message)[:4000]))

    def _asset_records(self):
        return self.research.records('asset')

    def _run_geo_x(self, cfg, sat, weather, terrain, assessment):
        if not cfg.get('geo_x_enabled', True):
            return self.store.setting('geo_x_latest') or {}, False
        direct_state = self.store.setting('geo_x_highres_state') or {}
        highres_changed = False
        if cfg.get('geo_x_highres_enabled', True):
            try:
                direct_state, highres_changed = scan_inbox(self.root, cfg, direct_state)
                self.store.setting('geo_x_highres_state', direct_state)
            except Exception as exc:
                self.log('High-resolution research inbox scan failed: ' + f'{type(exc).__name__}: {exc}')
        obs = build_observation(sat, weather, terrain, assessment,
                                area_name=cfg.get('auto_area_name') or 'Selected area')
        last_obs_sig = self.store.setting('geo_x_last_observation_signature')
        new_observation = obs['signature'] != last_obs_sig
        if new_observation:
            self.research.add('aerosentinel_observation', 'AEROSENTINEL EO/context observation', 'Model-derived', obs)
            self.store.setting('geo_x_last_observation_signature', obs['signature'])
        # Merge new AEROSENTINEL history with legacy TRUST-GEO-X observations so
        # an upgrade does not reset the temporal/seasonal baseline.
        history_records = self.research.records('aerosentinel_observation')[:240] + self.research.records('geo_x_observation')[:240]
        by_signature = {}
        for record in history_records:
            payload = record.get('payload') or {}
            key = str(payload.get('signature') or record.get('id'))
            by_signature[key] = payload
        history = list(by_signature.values())
        history.sort(key=lambda x: str(x.get('timestamp') or ''))
        history = history[-240:]
        foundation = latest_terramind_record(self.root, obs.get('timestamp'))
        calibration = self.store.setting('geo_x_calibration')
        result = run_aerosentinel(
            history, multisatellite_run=sat, highres_evidence=(direct_state or {}).get('latest'),
            foundation_record=foundation, calibration=calibration,
            abstain_threshold=float(cfg.get('geo_x_abstain_threshold', 0.45)),
            area_name=cfg.get('auto_area_name') or 'Selected area', assessment=assessment,
            deployment_context={
                'collection_interval_minutes': cfg.get('auto_interval_minutes', 30),
            })
        previous = self.store.setting('geo_x_latest') or {}
        changed = new_observation or highres_changed or result.get('signature') != previous.get('signature')
        self.store.setting('geo_x_latest', result)
        self.store.setting('aerosentinel_latest', result)
        if changed and result.get('signature') != self.store.setting('geo_x_last_record_signature'):
            self.research.add('aerosentinel_assessment', 'AEROSENTINEL aviation-safety research assessment', 'Model-derived', result)
            self.store.setting('geo_x_last_record_signature', result['signature'])
        return result, changed

    def _fetch_weather_if_due(self, cfg, force=False):
        previous = self.store.setting('weather_latest') or {}
        fetched_epoch = float(previous.get('_fetched_epoch') or 0)
        interval = max(5, int(cfg.get('ops_weather_interval_minutes', 5))) * 60
        if previous and not force and self.clock() - fetched_epoch < interval:
            return previous, False
        self.status('WEATHER', message='Refreshing public weather context.')
        weather = self.weather_fetcher(float(cfg['auto_latitude']), float(cfg['auto_longitude']),
                                       forecast_days=int(cfg.get('ops_weather_forecast_days', 3)))
        weather['_fetched_epoch'] = self.clock()
        self.store.setting('weather_latest', weather)
        return weather, True

    def _fetch_terrain_if_needed(self, cfg):
        if not cfg.get('ops_terrain_enabled', True):
            return self.store.setting('terrain_latest') or {}, False
        signature = f"{float(cfg['auto_latitude']):.6f}|{float(cfg['auto_longitude']):.6f}|{float(cfg['auto_radius_km']):.3f}|{int(cfg.get('multi_grid_pixels',512))}"
        previous = self.store.setting('terrain_latest') or {}
        if previous.get('_area_signature') == signature and previous.get('summary'):
            return previous, False
        self.status('TERRAIN', message='Downloading static Copernicus DEM terrain context.')
        terrain = fetch_terrain(area_bbox(cfg), self.root/'data/terrain'/signature.replace('|','_'),
                                grid_pixels=int(cfg.get('multi_grid_pixels',512)))
        terrain['_area_signature'] = signature
        self.store.setting('terrain_latest', terrain)
        return terrain, True

    def _report_due(self, cfg):
        if not cfg.get('ops_auto_report_enabled', True):
            return False
        latest = self.store.setting('ops_latest_report') or {}
        last = float(latest.get('epoch') or 0)
        interval = max(1, int(cfg.get('ops_report_interval_minutes', 5))) * 60
        return self.clock() - last >= interval

    def _prune_reports(self, cfg):
        folder = self.root/'data/reports/auto'
        if not folder.exists():
            return
        cutoff = self.clock() - max(1, int(cfg.get('ops_report_history_days', 7))) * 86400
        for path in folder.glob('briefing_*.*'):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass

    def _export_report(self, assessment, cfg, geo_x=None):
        report_time = datetime.fromtimestamp(self.clock(), timezone.utc).isoformat()
        paths = write_briefing(assessment, self.root/'data/reports/auto', report_generated_at=report_time, geo_x=geo_x,
                               aircraft_awareness=self.store.setting('aircraft_awareness_latest') or {})
        report = dict(paths, epoch=self.clock(), interval_minutes=int(cfg.get('ops_report_interval_minutes', 5)))
        self.store.setting('ops_latest_report', report)
        self._prune_reports(cfg)
        self.log('Automatic 5-minute report snapshot saved: ' + str(paths.get('html')))
        return report

    def tick(self, cfg):
        if self.store.setting('paused') or not cfg.get('ops_automation_enabled', True):
            self.status('PAUSED' if self.store.setting('paused') else 'DISABLED')
            return

        # High-resolution imagery is a local/manual input and must be watched even
        # when the AOI/location has not yet been saved. v1.1 incorrectly tied this
        # watcher to location-dependent weather/satellite automation.
        if cfg.get('geo_x_highres_enabled', True):
            try:
                direct_state, highres_changed = scan_inbox(
                    self.root, cfg, self.store.setting('geo_x_highres_state') or {})
                self.store.setting('geo_x_highres_state', direct_state)
            except Exception as exc:
                direct_state, highres_changed = {}, False
                self.log('High-resolution independent inbox scan failed: ' + f'{type(exc).__name__}: {exc}')
        else:
            direct_state, highres_changed = {}, False

        if not cfg.get('auto_location_ready', False):
            self.status('LOCATION_REQUIRED',
                        message='Location is required for automatic satellite/weather collection. High-res local inbox scanning is still active.',
                        highres_input_count=int((direct_state or {}).get('input_file_count',0) or 0),
                        highres_processed_this_scan=int((direct_state or {}).get('processed_this_scan',0) or 0),
                        highres_last_scan=(direct_state or {}).get('last_scan_at'))
            return

        request = self.store.setting('ops_request')
        previous_status = self.store.setting('ops_status') or {}
        force = bool(request and request != previous_status.get('handled_request'))
        weather_error = None
        terrain_error = None
        try:
            if cfg.get('ops_weather_enabled', True):
                weather, weather_changed = self._fetch_weather_if_due(cfg, force=force)
            else:
                weather, weather_changed = (self.store.setting('weather_latest') or {}), False
        except Exception as exc:
            weather = self.store.setting('weather_latest') or {}
            weather_changed = False
            weather_error = f'{type(exc).__name__}: {exc}'
            self.log('Weather refresh failed: ' + weather_error)
        try:
            terrain, terrain_changed = self._fetch_terrain_if_needed(cfg)
        except Exception as exc:
            terrain = self.store.setting('terrain_latest') or {}
            terrain_changed = False
            terrain_error = f'{type(exc).__name__}: {exc}'
            self.log('Terrain refresh failed: ' + terrain_error)

        sat = self.store.setting('multisatellite_last_result') or {}
        sat_signature = str(sat.get('finished_at') or '')
        weather_signature = str(weather.get('fetched_at') or '')
        terrain_signature = str(terrain.get('_area_signature') or '')
        combined_signature = sat_signature + '|' + weather_signature + '|' + terrain_signature
        last_sig = previous_status.get('assessment_input_signature')
        assessment_due = force or weather_changed or terrain_changed or combined_signature != last_sig
        assessment = self.store.setting('ops_latest_assessment') or {}

        if assessment_due or not assessment:
            self.status('ANALYZING', handled_request=request,
                        message='Fusing satellite, weather, terrain and local asset context.',
                        weather_error=weather_error, terrain_error=terrain_error)
            assessment = compute_resilience_assessment(
                sat, weather, self._asset_records(), area_name=cfg.get('auto_area_name') or 'Selected area',
                profile=cfg.get('ops_profile','airfield_resilience'), terrain=terrain)
            self.store.setting('ops_latest_assessment', assessment)
            last_record_sig = self.store.setting('ops_last_record_signature')
            if assessment['signature'] != last_record_sig:
                self.research.add('auto_assessment', 'Automated resilience screening', 'Model-derived', assessment)
                self.store.setting('ops_last_record_signature', assessment['signature'])

        geo_x = self.store.setting('geo_x_latest') or {}
        geo_x_changed = False
        if cfg.get('geo_x_enabled', True):
            self.status('EARLY_WARNING', handled_request=request,
                        message='Running AEROSENTINEL multimodal, spatio-temporal, rare-event, uncertainty and aviation-risk layers.')
            geo_x, geo_x_changed = self._run_geo_x(cfg, sat, weather, terrain, assessment)

        report = self.store.setting('ops_latest_report') or {}
        if cfg.get('ops_auto_report_enabled', True) and (assessment_due or geo_x_changed or not report or self._report_due(cfg)):
            report = self._export_report(assessment, cfg, geo_x=geo_x)

        next_report = None
        if cfg.get('ops_auto_report_enabled', True):
            next_report = float(report.get('epoch') or self.clock()) + max(1, int(cfg.get('ops_report_interval_minutes', 5))) * 60

        self.status('UP_TO_DATE', handled_request=request, assessment_input_signature=combined_signature,
                    last_assessment=assessment.get('generated_at'), latest_signature=assessment.get('signature'),
                    message='Satellite + weather + terrain screening is current; reports auto-save on schedule.',
                    weather_error=weather_error, terrain_error=terrain_error,
                    last_report_epoch=report.get('epoch'), next_report_epoch=next_report,
                    latest_report_html=report.get('latest_html'), latest_report_json=report.get('latest_json'),
                    geo_x_status=(geo_x.get('predictive_early_warning') or {}).get('status'),
                    geo_x_score=(geo_x.get('predictive_early_warning') or {}).get('score'),
                    geo_x_reliability=(geo_x.get('uncertainty_engine') or {}).get('reliability'),
                    aerosentinel_status=(geo_x.get('operational_aviation_risk') or {}).get('status') or (geo_x.get('predictive_early_warning') or {}).get('status'),
                    aerosentinel_risk=(geo_x.get('operational_aviation_risk') or {}).get('risk_score') or (geo_x.get('predictive_early_warning') or {}).get('score'),
                    aerosentinel_reliability=(geo_x.get('uncertainty_engine') or {}).get('reliability'))

    def run(self):
        with worker_lock(self.root/'data/collection/ops_worker.lock'):
            self.status('STARTING')
            try:
                while not self.stopping.is_set():
                    try:
                        self.tick(load_config(self.root))
                    except Exception as exc:
                        self.status('ERROR', error=f'{type(exc).__name__}: {exc}',
                                    message='Automation worker error; it will retry automatically.')
                        self.log(f'Automation failure: {type(exc).__name__}: {exc}')
                    self.stopping.wait(5)
            finally:
                self.status('STOPPED')


if __name__ == '__main__':
    folder = ROOT/'data/collection/logs'; folder.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[RotatingFileHandler(
        folder/'ops.log', maxBytes=3_000_000, backupCount=2, encoding='utf-8')])
    worker = OpsWorker()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: worker.stopping.set())
    worker.run()

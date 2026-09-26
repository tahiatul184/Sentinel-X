import json
from pathlib import Path
import tempfile
import unittest

from collection_store import DEFAULTS, Store
from ops_worker import OpsWorker
from trust_core import ResearchStore


class OpsWorkerTests(unittest.TestCase):
    def test_tick_automates_weather_assessment_and_briefing(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            cfg=dict(DEFAULTS)
            cfg.update(auto_collect_enabled=True,ops_automation_enabled=True,ops_weather_enabled=True,ops_terrain_enabled=False,
                       auto_location_ready=True,auto_area_name='Test AOI',auto_latitude=23.8,auto_longitude=90.4,
                       multi_satellites=['sentinel-1','sentinel-2'])
            (root/'collection_config.json').write_text(json.dumps(cfg),encoding='utf-8')
            Store(root).setting('multisatellite_last_result',{
                'finished_at':'2026-09-12T06:00:00+00:00','fusion':{
                    'mean_wetness_screening':.4,'mean_fusion_uncertainty':.2,'mean_source_coverage':1,
                    'temporal_alignment_score':.8,'available_satellites':['sentinel-1','sentinel-2'],
                    'acquisition_skew_hours':4,'preview_path':'','temporal_change':{}}})
            def fake_weather(*args,**kwargs):
                return {'provider':'test','fetched_at':'2099-01-01T00:00:00+00:00','summary':{
                    'rain_next_6h_mm':5,'rain_next_24h_mm':20,'max_wind_next_6h_kmh':15,
                    'max_gust_next_6h_kmh':25,'min_visibility_next_6h_m':9000,'thunderstorm_hours_next_24h':0}}
            worker=OpsWorker(root=root,clock=lambda:2_000_000_000.0,weather_fetcher=fake_weather)
            worker.tick(cfg)
            assessment=Store(root).setting('ops_latest_assessment')
            self.assertTrue(assessment['signature'])
            report=Store(root).setting('ops_latest_report')
            self.assertTrue(Path(report['latest_html']).exists())
            aero=Store(root).setting('aerosentinel_latest')
            self.assertEqual(aero['framework'],'AEROSENTINEL')
            self.assertIn('operational_aviation_risk',aero)
            payload=json.loads(Path(report['latest_json']).read_text(encoding='utf-8'))
            self.assertIn('aerosentinel',payload)
            self.assertEqual(Store(root).setting('ops_status')['state'],'UP_TO_DATE')
            self.assertEqual(len(ResearchStore(root).records('auto_assessment')),1)
            self.assertEqual(len(ResearchStore(root).records('aerosentinel_assessment')),1)

    def test_report_snapshot_repeats_on_five_minute_cadence_without_fake_new_assessment(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); now=[2_000_000_000.0]
            cfg=dict(DEFAULTS)
            cfg.update(auto_collect_enabled=True,ops_automation_enabled=True,ops_weather_enabled=False,ops_terrain_enabled=False,
                       ops_auto_report_enabled=True,ops_report_interval_minutes=5,auto_location_ready=True,
                       auto_area_name='Test AOI',auto_latitude=23.8,auto_longitude=90.4,
                       multi_satellites=['sentinel-1','sentinel-2'])
            (root/'collection_config.json').write_text(json.dumps(cfg),encoding='utf-8')
            Store(root).setting('multisatellite_last_result',{'finished_at':'2026-09-12T06:00:00+00:00','fusion':{
                'mean_wetness_screening':.4,'mean_fusion_uncertainty':.2,'mean_source_coverage':1,
                'temporal_alignment_score':.8,'available_satellites':['sentinel-1','sentinel-2'],
                'acquisition_skew_hours':4,'preview_path':'','temporal_change':{}}})
            worker=OpsWorker(root=root,clock=lambda:now[0])
            worker.tick(cfg)
            first_report=Store(root).setting('ops_latest_report')
            first_assessment=Store(root).setting('ops_latest_assessment')['generated_at']
            now[0]+=299
            worker.tick(cfg)
            self.assertEqual(Store(root).setting('ops_latest_report')['epoch'],first_report['epoch'])
            now[0]+=1
            worker.tick(cfg)
            second_report=Store(root).setting('ops_latest_report')
            self.assertGreater(second_report['epoch'],first_report['epoch'])
            self.assertEqual(Store(root).setting('ops_latest_assessment')['generated_at'],first_assessment)
            self.assertTrue(Path(second_report['latest_html']).exists())

    def test_highres_inbox_scans_even_when_location_is_not_ready(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); cfg=dict(DEFAULTS)
            cfg.update(ops_automation_enabled=True,geo_x_highres_enabled=True,auto_location_ready=False)
            (root/'collection_config.json').write_text(json.dumps(cfg),encoding='utf-8')
            box=root/'highres_inbox';box.mkdir()
            image=box/'test.tif';arr=np.ones((32,32),dtype='float32')
            with rasterio.open(image,'w',driver='GTiff',width=32,height=32,count=1,dtype='float32',
                               crs='EPSG:32646',transform=from_origin(500000,2600000,.3,.3)) as dst: dst.write(arr,1)
            worker=OpsWorker(root=root,clock=lambda:2_000_000_000.0)
            worker.tick(cfg)
            state=Store(root).setting('geo_x_highres_state')
            self.assertEqual(state['input_file_count'],1)
            self.assertTrue(state['last_scan_at'])
            self.assertTrue((root/'data/highres/highres_log.jsonl').exists())
            self.assertEqual(Store(root).setting('ops_status')['state'],'LOCATION_REQUIRED')


if __name__=='__main__':unittest.main()

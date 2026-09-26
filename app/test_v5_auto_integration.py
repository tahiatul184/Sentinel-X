import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from collection_store import DEFAULTS, Store
from multisatellite_worker import MultiSatelliteWorker
from ops_worker import OpsWorker


def tif(path,base=0):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(path,'w',driver='GTiff',width=16,height=16,count=2,dtype='float32',
        crs='EPSG:4326',transform=from_bounds(90.3,23.7,90.4,23.8,16,16)) as dst:
        dst.write(np.full((16,16),base,dtype='float32'),1);dst.write(np.full((16,16),base+1,dtype='float32'),2)
    return path


class V5AutoIntegrationTests(unittest.TestCase):
    def test_one_config_drives_satellite_to_automated_briefing(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=dict(DEFAULTS)
            cfg.update(auto_collect_enabled=True,ops_automation_enabled=True,ops_weather_enabled=True,ops_terrain_enabled=True,
                auto_location_ready=True,auto_area_name='Authorized test area',auto_latitude=23.8,auto_longitude=90.4,
                auto_radius_km=2,multi_satellites=['sentinel-1','sentinel-2'],multi_concurrent_workers=2,
                multi_scenes_per_satellite=1,multi_grid_pixels=128)
            (root/'collection_config.json').write_text(json.dumps(cfg),encoding='utf-8')
            def fake_run(**kwargs):
                out=Path(kwargs['output_dir']);s1=tif(out/'s1.tif',1);s2=tif(out/'s2.tif',2);f=tif(out/'fusion.tif',3)
                preview=out/'fusion.png';preview.write_bytes(b'png')
                rows=[]
                for key,path in [('sentinel-1',s1),('sentinel-2',s2)]:
                    rows.append({'satellite':key,'label':key,'kind':'sar' if key=='sentinel-1' else 'optical','scene_id':key,
                        'collection':key,'platform':key,'acquired_at':'2026-09-12T06:00:00+00:00','cloud_cover_percent':5,
                        'valid_percent':100,'stack_path':str(path),'preview_path':'','analysis':{},'asset_keys':{},
                        'started_at':'2026-09-12T06:00:00+00:00','finished_at':'2026-09-12T06:01:00+00:00','elapsed_seconds':1})
                return {'version':'v5-test','started_at':'2026-09-12T06:00:00+00:00','finished_at':'2026-09-12T06:01:00+00:00',
                    'elapsed_seconds':1,'concurrency':{'peak_scene_workers':2},'aoi':{},'period':{},'requested_satellites':['sentinel-1','sentinel-2'],
                    'search_counts':{'sentinel-1':1,'sentinel-2':1},'scene_results':rows,'errors':{},'simultaneous_computation':True,
                    'simultaneous_acquisition_claimed':False,'fusion':{'fusion_path':str(f),'preview_path':str(preview),
                    'available_satellites':['sentinel-1','sentinel-2'],'acquisition_skew_hours':0,'mean_wetness_screening':.55,
                    'mean_fusion_uncertainty':.2,'mean_source_coverage':1,'temporal_alignment_score':1,'temporal_change':{}}}
            sat=MultiSatelliteWorker(root=root,clock=lambda:2_000_000_000.0)
            with patch('multisatellite_worker.run_concurrent_multisatellite',side_effect=fake_run):sat.tick(cfg)
            def fake_weather(*a,**k):return {'provider':'test','fetched_at':'2099-01-01T00:00:00+00:00','summary':{
                'rain_next_6h_mm':10,'rain_next_24h_mm':30,'max_wind_next_6h_kmh':18,'max_gust_next_6h_kmh':30,
                'min_visibility_next_6h_m':8000,'thunderstorm_hours_next_24h':0,'max_temperature_next_24h_c':34}}
            terrain={'provider':'test','fetched_at':'2099-01-01T00:00:00+00:00','summary':{
                'mean_slope_deg':2,'p90_slope_deg':4,'relief_p05_p95_m':15},'preview_path':'','raster_path':''}
            ops=OpsWorker(root=root,clock=lambda:2_000_000_100.0,weather_fetcher=fake_weather)
            with patch('ops_worker.fetch_terrain',return_value=terrain):ops.tick(cfg)
            result=Store(root).setting('ops_latest_assessment')
            self.assertEqual(result['area_name'],'Authorized test area')
            self.assertIn('monsoon_flood_screening',result['scores'])
            report=Store(root).setting('ops_latest_report')
            self.assertTrue(Path(report['latest_html']).exists())
            gx=Store(root).setting('aerosentinel_latest')
            self.assertTrue(gx['architecture']['open_world_anomaly_path'])
            self.assertTrue(gx['architecture']['rare_event_detection'])
            self.assertIn('operational_aviation_risk',gx)
            self.assertIn(gx['predictive_early_warning']['status'],{'ABSTAIN / HUMAN REVIEW','NORMAL','MONITOR','ELEVATED','HIGH REVIEW'})
            payload=json.loads(Path(report['latest_json']).read_text(encoding='utf-8'))
            self.assertIn('aerosentinel',payload)
            self.assertIn('trust_geo_x',payload)
            self.assertEqual(Store(root).setting('ops_status')['state'],'UP_TO_DATE')


if __name__=='__main__':unittest.main()

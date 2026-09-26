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


def make_tif(path, base=0):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(path,'w',driver='GTiff',width=16,height=16,count=3,dtype='float32',
                       crs='EPSG:4326',transform=from_bounds(90.3,23.7,90.4,23.8,16,16)) as dst:
        for i in range(1,4): dst.write(np.full((16,16),base+i,dtype='float32'),i)
    return path


class WorkerIntegrationTests(unittest.TestCase):
    def test_tick_runs_concurrent_pipeline_and_persists_result(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            cfg=dict(DEFAULTS)
            cfg.update(auto_collect_enabled=True,auto_location_ready=True,auto_area_name='Test',
                       auto_latitude=23.8,auto_longitude=90.4,auto_radius_km=2.0,
                       multi_satellites=['sentinel-1','sentinel-2'],multi_concurrent_workers=2,
                       multi_scenes_per_satellite=1,multi_grid_pixels=128,multi_temporal_tolerance_hours=72.0)
            (root/'collection_config.json').write_text(json.dumps(cfg),encoding='utf-8')

            def fake_run(**kwargs):
                out=Path(kwargs['output_dir'])
                s1=make_tif(out/'scenes/s1.tif', 0); s2=make_tif(out/'scenes/s2.tif', 10)
                fusion=make_tif(out/'fusion/multisatellite_fusion.tif', 20)
                preview=out/'fusion/preview.png'; preview.parent.mkdir(parents=True,exist_ok=True); preview.write_bytes(b'png')
                rows=[]
                for key,path in [('sentinel-1',s1),('sentinel-2',s2)]:
                    rows.append({'satellite':key,'label':key,'kind':'sar' if key=='sentinel-1' else 'optical',
                                 'scene_id':key+'-scene','collection':key,'platform':key,
                                 'acquired_at':'2026-09-12T06:00:00+00:00','cloud_cover_percent':5,
                                 'valid_percent':100.0,'stack_path':str(path),'preview_path':'','analysis':{},
                                 'asset_keys':{},'started_at':'2026-09-12T06:00:00+00:00',
                                 'finished_at':'2026-09-12T06:01:00+00:00','elapsed_seconds':1.0})
                return {'version':'test','started_at':'2026-09-12T06:00:00+00:00','finished_at':'2026-09-12T06:01:00+00:00',
                        'elapsed_seconds':1.0,'concurrency':{'peak_scene_workers':2},'aoi':{},'period':{},
                        'requested_satellites':['sentinel-1','sentinel-2'],'search_counts':{'sentinel-1':1,'sentinel-2':1},
                        'scene_results':rows,'fusion':{'fusion_path':str(fusion),'preview_path':str(preview),
                            'available_satellites':['sentinel-1','sentinel-2'],'acquisition_skew_hours':0.0},
                        'errors':{},'simultaneous_computation':True,'simultaneous_acquisition_claimed':False}

            worker=MultiSatelliteWorker(root=root,clock=lambda: 1_800_000_000.0)
            Store(root).setting('multisatellite_request','run-now')
            with patch('multisatellite_worker.run_concurrent_multisatellite',side_effect=fake_run):
                worker.tick(cfg)
            status=Store(root).setting('multisatellite_download')
            self.assertEqual(status['state'],'UP_TO_DATE')
            self.assertEqual(status['peak_workers'],2)
            self.assertEqual(len(Store(root).scenes()),3)
            self.assertTrue(Store(root).setting('multisatellite_last_result')['simultaneous_computation'])
            self.assertTrue((root/'data/imagery/image_log.jsonl').exists())
            self.assertTrue(any((root/'data/imagery').rglob('analysis_stack.tif')))


if __name__=='__main__':
    unittest.main()

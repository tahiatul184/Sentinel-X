import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from trust_core import ResearchStore,water_balance,simulate,evaluation,parse_sensors,validate_asset
from foundation_job import normalize,MEAN

class CoreTests(unittest.TestCase):
    def test_water_mass_conservation(self):
        rows=water_balance([20,40,0],area_ha=10,runoff=.8,infiltration_mm_h=2,drain_m3_s=.1,initial_mm=5)
        previous=500
        for row in rows:
            self.assertAlmostEqual(previous+row['inflow_m3']-row['infiltrated_m3']-row['drained_m3'],row['stored_m3'])
            self.assertGreaterEqual(row['stored_m3'],0);previous=row['stored_m3']
    def test_no_rain_no_water(self):
        rows=water_balance([0]*10,1,.5,10,10)
        self.assertEqual(max(r['surface_storage_mm'] for r in rows),0)
    def test_extra_drainage_reduces_storage(self):
        low=water_balance([50]*10,10,.9,0,.01)
        high=water_balance([50]*10,10,.9,0,.5)
        self.assertGreater(low[-1]['stored_m3'],high[-1]['stored_m3'])
    def test_bad_scenario_input(self):
        with self.assertRaises(ValueError):water_balance([float('nan')],1,.5,1,1)
        with self.assertRaises(ValueError):water_balance([30],0,.5,1,1)
    def test_scenario_reproducible_and_labelled(self):
        p=dict(hours=2,rain_mm_h=30,area_ha=10,runoff=.8,infiltration_mm_h=2,drain_m3_s=.1,initial_mm=0,uncertainty_percent=20)
        a=simulate(p);b=simulate(p)
        self.assertEqual(a,b);self.assertEqual(a['origin'],'Simulated')
        for r in a['series']:self.assertLessEqual(r['p10_mm'],r['p90_mm'])
    def test_known_classification_metrics(self):
        result=evaluation([{'label':y,'probability':p} for y,p in [(1,.9),(0,.8),(1,.2),(0,.1)]])
        self.assertEqual(result['tp'],1);self.assertEqual(result['fp'],1);self.assertEqual(result['fn'],1)
        self.assertEqual(result['f1'],.5);self.assertAlmostEqual(result['brier'],.325)
    def test_undefined_precision_not_misreported(self):
        r=evaluation([dict(label=0,probability=.1)])
        self.assertIsNone(r['precision']);self.assertIsNone(r['recall'])
    def test_bad_prediction_inputs(self):
        with self.assertRaises(ValueError):evaluation([dict(label=.2,probability=.9)])
        with self.assertRaises(ValueError):evaluation([dict(label=1,probability=2)])
    def test_sensor_timezones_and_duplicates(self):
        base='timestamp,site,rain_mm_h,water_level_m,source\n2026-09-12T06:00:00Z,A,20,1,Gauge\n'
        self.assertEqual(len(parse_sensors(base)),1)
        with self.assertRaises(ValueError):parse_sensors(base+'2026-09-12T06:00:00Z,A,20,1,Gauge\n')
        with self.assertRaises(ValueError):parse_sensors(base.replace('T06:00:00Z','T06:00:00'))
    def test_asset_site_and_coordinates_required(self):
        a=dict(name='Drain',site='SITE_A',latitude=23,longitude=90,criticality=3,threshold_mm=20)
        self.assertEqual(validate_asset(a)['site'],'SITE_A')
        with self.assertRaises(ValueError):validate_asset(dict(a,latitude=100))
        with self.assertRaises(ValueError):validate_asset(dict(a,site=''))
    def test_records_and_reviews_survive_restart(self):
        with tempfile.TemporaryDirectory() as d:
            s=ResearchStore(Path(d));ident=s.add('scenario','Test','Simulated',dict(value=2))
            s.review(ident,'Researcher','Reviewed','Checked assumptions')
            export=ResearchStore(Path(d)).export()
            self.assertEqual(export['records'][0]['origin'],'Simulated')
            self.assertEqual(len(export['reviews']),1)
            self.assertEqual(export['records'][0]['id'],ident)
    def test_prithvi_normalization_and_rgb_rejection(self):
        data=np.broadcast_to(MEAN[:,None,None],(6,224,224))
        self.assertTrue(np.allclose(normalize(data,'scaled_10000'),0))
        self.assertTrue(np.allclose(normalize(data/10000,'reflectance'),0,atol=1e-6))
        with self.assertRaises(ValueError):normalize(np.zeros((3,224,224)),'scaled_10000')
        with self.assertRaises(ValueError):normalize(np.full((6,224,224),np.nan),'reflectance')

class RasterTests(unittest.TestCase):
    def test_real_geotiff_inspection(self):
        import rasterio
        from rasterio.transform import from_origin
        from collection_worker import inspect_scene
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/'test.tif'
            with rasterio.open(path,'w',driver='GTiff',height=256,width=256,count=3,dtype='uint8',crs='EPSG:32646',transform=from_origin(400000,2700000,10,10)) as out:
                out.write(np.random.default_rng(1).integers(0,255,(3,256,256),dtype=np.uint8))
            q=inspect_scene(dict(id='test',path=str(path),input_metadata={'acquired_at':'2026-09-12T06:00:00Z'}),root)
            self.assertEqual(q['pixel_size'],[10,10]);self.assertTrue(Path(q['preview']).exists())
            self.assertEqual(q['valid_percent_sampled'],100)

if __name__=='__main__':unittest.main()

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from multisatellite_pipeline import (
    SATELLITES, build_analysis_grid, run_concurrent_multisatellite, select_assets, process_item,
    search_planetary_computer,
)


def fake_item(spec, suffix='1'):
    assets = {}
    if spec.kind == 'optical':
        for key in ['red', 'green', 'blue', 'nir08', 'swir22', 'qa_pixel']:
            assets[key] = {'href': f'https://example.invalid/{key}.tif'}
        if spec.key == 'landsat-8-9':
            assets['lwir11'] = {
                'href': 'https://example.invalid/lwir11.tif',
                'raster:bands': [{'scale': 0.00341802, 'offset': 149.0}],
            }
            assets['qa'] = {'href':'https://example.invalid/st_qa.tif','raster:bands':[{'scale':0.01}]}
            assets['emis'] = {'href':'https://example.invalid/emis.tif','raster:bands':[{'scale':0.0001}]}
            assets['cdist'] = {'href':'https://example.invalid/cdist.tif','raster:bands':[{'scale':0.01}]}
    else:
        assets = {'vv': {'href': 'https://example.invalid/vv.tif'}, 'vh': {'href': 'https://example.invalid/vh.tif'}}
    return {
        'id': f'{spec.key}-{suffix}', 'collection': spec.collections[0],
        'properties': {'datetime': f'2026-09-1{suffix}T06:00:00Z', 'platform': spec.key, 'eo:cloud_cover': 5},
        'assets': assets,
    }


class MultiSatelliteTests(unittest.TestCase):
    def test_selects_optical_and_sar_assets(self):
        optical = select_assets(SATELLITES['landsat-8-9'], fake_item(SATELLITES['landsat-8-9']))
        self.assertTrue({'red','green','blue','nir'}.issubset(optical))
        self.assertIn('thermal', optical)
        self.assertIn('thermal_qa', optical)
        self.assertIn('emissivity', optical)
        self.assertIn('cloud_distance', optical)
        self.assertEqual(optical['qa_pixel'][0], 'qa_pixel')
        sar = select_assets(SATELLITES['sentinel-1'], fake_item(SATELLITES['sentinel-1']))
        self.assertEqual(sar['polarization'], ('vv','vh'))

    def test_landsat_generic_qa_is_thermal_uncertainty_not_qa_pixel(self):
        item=fake_item(SATELLITES['landsat-8-9'])
        item['assets'].pop('qa_pixel')
        selected=select_assets(SATELLITES['landsat-8-9'],item)
        self.assertIn('thermal_qa',selected)
        self.assertNotIn('qa_pixel',selected)
        self.assertEqual(selected['thermal_qa'][0],'qa')

    def test_common_grid_is_local_metric_grid(self):
        grid = build_analysis_grid([90.35,23.75,90.45,23.85], 256)
        self.assertIn('326', str(grid['crs']))
        self.assertLessEqual(max(grid['width'], grid['height']), 257)
        self.assertGreater(grid['resolution_m'], 0)

    def test_scene_workers_overlap_and_fuse(self):
        active = 0
        peak = 0
        lock = threading.Lock()

        def searcher(spec, bbox, start, end, max_items, max_cloud):
            time.sleep(0.03)
            return [fake_item(spec)]

        def processor(spec, item, grid, output_dir, event=None):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                time.sleep(0.12)
                shape = (grid['height'], grid['width'])
                base = {'sentinel-1': .7, 'sentinel-2': .5, 'landsat-8-9': .4}[spec.key]
                water = np.full(shape, base, dtype='float32')
                valid = np.ones(shape, dtype=bool)
                arrays = {'water': water, 'valid': valid}
                if spec.kind == 'optical':
                    arrays['ndvi'] = np.full(shape, .3 if spec.key == 'sentinel-2' else .2, dtype='float32')
                if spec.key == 'landsat-8-9':
                    arrays['surface_temp_c'] = np.full(shape, 30.0, dtype='float32')
                    arrays['heat'] = np.full(shape, .6, dtype='float32')
                    arrays['thermal_valid'] = valid.copy()
                now = time.time()
                return {
                    'satellite': spec.key, 'label': spec.label, 'kind': spec.kind,
                    'scene_id': item['id'], 'collection': item['collection'], 'platform': spec.key,
                    'acquired_at': item['properties']['datetime'], 'cloud_cover_percent': 5,
                    'valid_percent': 100.0, 'stack_path': str(Path(output_dir)/'fake.tif'),
                    'preview_path': str(Path(output_dir)/'fake.png'), 'analysis': {},
                    'asset_keys': {}, 'started_at': item['properties']['datetime'],
                    'finished_at': item['properties']['datetime'], 'elapsed_seconds': .12,
                    '_arrays': arrays,
                }
            finally:
                with lock:
                    active -= 1

        with tempfile.TemporaryDirectory() as td:
            run = run_concurrent_multisatellite(
                bbox=[90.39,23.79,90.41,23.81], start_date='2026-09-01', end_date='2026-09-12',
                output_dir=td, selected_satellites=['sentinel-1','sentinel-2','landsat-8-9'],
                workers=3, grid_pixels=128, searcher=searcher, processor=processor,
            )
            self.assertGreaterEqual(peak, 2)
            self.assertGreaterEqual(run['concurrency']['peak_scene_workers'], 2)
            self.assertTrue(run['simultaneous_computation'])
            self.assertEqual(set(run['fusion']['available_satellites']), {'sentinel-1','sentinel-2','landsat-8-9'})
            self.assertTrue(Path(run['fusion']['fusion_path']).exists())
            self.assertTrue(Path(run['fusion']['preview_path']).exists())
            self.assertEqual(run['fusion']['thermal_source_count'], 1)
            self.assertAlmostEqual(run['fusion']['mean_land_surface_temperature_celsius'], 30.0, places=2)
            self.assertAlmostEqual(run['fusion']['mean_relative_heat_screening'], .6, places=2)
            self.assertFalse(run['simultaneous_acquisition_claimed'])



    def test_planetary_computer_rest_search_signs_blob_assets(self):
        class Response:
            def __init__(self, body): self.body=body
            def raise_for_status(self): return None
            def json(self): return self.body
        class Session:
            def __init__(self): self.headers={}
            def post(self,url,json=None,timeout=None):
                self.payload=json
                return Response({'features':[{'id':'S2-x','collection':'sentinel-2-l2a',
                    'properties':{'datetime':'2026-09-12T06:00:00Z'},
                    'assets':{'red':{'href':'https://example.blob.core.windows.net/c/red.tif'},
                              'green':{'href':'https://planetarycomputer.microsoft.com/api/data/v1/item/preview.png'}}}]})
            def get(self,url,timeout=None): return Response({'token':'sv=test&sig=abc'})
            def close(self): pass
        fake=Session()
        with patch('requests.Session',return_value=fake):
            items=search_planetary_computer(SATELLITES['sentinel-2'],[90.3,23.7,90.4,23.8],
                                             '2026-09-01','2026-09-12',1,80)
        self.assertIn('?sv=test&sig=abc',items[0]['assets']['red']['href'])
        self.assertNotIn('?sv=test',items[0]['assets']['green']['href'])
        self.assertEqual(fake.payload['collections'],['sentinel-2-l2a'])

    def test_real_raster_optical_processing_on_common_grid(self):
        import rasterio
        from rasterio.transform import from_bounds
        bbox=[90.39,23.79,90.41,23.81]
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            transform=from_bounds(*bbox,64,64)
            values={'red':0.2,'green':0.25,'blue':0.15,'nir08':0.4,'swir16':0.3,'swir22':0.1}
            assets={}
            for key,value in values.items():
                path=td/f'{key}.tif'
                with rasterio.open(path,'w',driver='GTiff',width=64,height=64,count=1,dtype='float32',
                                   crs='EPSG:4326',transform=transform,nodata=-9999) as dst:
                    dst.write(np.full((64,64),value,dtype='float32'),1)
                assets[key]={'href':str(path)}
            item={'id':'S2-test','collection':'sentinel-2-l2a','properties':{'datetime':'2026-09-12T06:00:00Z','platform':'sentinel-2a'},'assets':assets}
            grid=build_analysis_grid(bbox,128)
            with patch('multisatellite_pipeline._validated_remote_href',side_effect=lambda href: href):
                result=process_item(SATELLITES['sentinel-2'],item,grid,td/'out')
            self.assertAlmostEqual(result['analysis']['ndvi']['mean'],1/3,places=2)
            self.assertTrue(Path(result['stack_path']).exists())
            self.assertTrue(Path(result['preview_path']).exists())
            self.assertTrue(Path(result['enhanced_preview_path']).exists())
            self.assertIn('mndwi',result['analysis'])
            self.assertIn('ndmi',result['analysis'])
            self.assertIn('savi',result['analysis'])
            self.assertIn('nbr2',result['analysis'])
            self.assertGreater(result['processing_quality']['processing_readiness_score'],0)
            self.assertGreater(result['valid_percent'],95)

    def test_landsat_thermal_surface_temperature_processing(self):
        import rasterio
        from rasterio.transform import from_bounds
        bbox = [90.39, 23.79, 90.41, 23.81]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            transform = from_bounds(*bbox, 64, 64)
            assets = {}
            reflectance = {'red': .20, 'green': .25, 'blue': .15, 'nir08': .40, 'swir22': .10}
            for key, value in reflectance.items():
                path = td / f'{key}.tif'
                with rasterio.open(path, 'w', driver='GTiff', width=64, height=64, count=1,
                                   dtype='float32', crs='EPSG:4326', transform=transform, nodata=-9999) as dst:
                    dst.write(np.full((64, 64), value, dtype='float32'), 1)
                assets[key] = {'href': str(path)}

            # Raw Landsat Collection-2 ST_B10 DN. The STAC metadata below should
            # convert this to about 302.63 K / 29.48 C.
            thermal_path = td / 'lwir11.tif'
            with rasterio.open(thermal_path, 'w', driver='GTiff', width=64, height=64, count=1,
                               dtype='uint16', crs='EPSG:4326', transform=transform, nodata=0) as dst:
                dst.write(np.full((64, 64), 44947, dtype='uint16'), 1)
            assets['lwir11'] = {
                'href': str(thermal_path),
                'raster:bands': [{'scale': 0.00341802, 'offset': 149.0}],
            }

            def write_aux(name, value, dtype, scale):
                path=td/f'{name}.tif'
                with rasterio.open(path,'w',driver='GTiff',width=64,height=64,count=1,
                                   dtype=dtype,crs='EPSG:4326',transform=transform,nodata=-9999) as dst:
                    dst.write(np.full((64,64),value,dtype=dtype),1)
                assets[name]={'href':str(path),'raster:bands':[{'scale':scale}]}
            write_aux('qa',100,'int16',0.01)       # 1.00 K ST_QA uncertainty
            write_aux('emis',9600,'int16',0.0001) # emissivity 0.96
            write_aux('emsd',100,'int16',0.0001)  # emissivity std 0.01
            write_aux('cdist',250,'int16',0.01)   # 2.5 km from cloud
            write_aux('atran',8000,'int16',0.0001)# atmospheric transmittance 0.8
            qars_path=td/'qa_radsat.tif'
            with rasterio.open(qars_path,'w',driver='GTiff',width=64,height=64,count=1,
                               dtype='uint16',crs='EPSG:4326',transform=transform,nodata=None) as dst:
                dst.write(np.zeros((64,64),dtype='uint16'),1)
            assets['qa_radsat']={'href':str(qars_path)}

            qa_path = td / 'qa_pixel.tif'
            with rasterio.open(qa_path, 'w', driver='GTiff', width=64, height=64, count=1,
                               dtype='uint16', crs='EPSG:4326', transform=transform, nodata=None) as dst:
                dst.write(np.zeros((64, 64), dtype='uint16'), 1)
            assets['qa_pixel'] = {'href': str(qa_path)}

            item = {
                'id': 'LC09-thermal-test', 'collection': 'landsat-c2-l2',
                'properties': {'datetime': '2026-09-12T06:00:00Z', 'platform': 'landsat-9'},
                'assets': assets,
            }
            grid = build_analysis_grid(bbox, 128)
            with patch('multisatellite_pipeline._validated_remote_href', side_effect=lambda href: href):
                result = process_item(SATELLITES['landsat-8-9'], item, grid, td / 'out')
            temp = result['analysis']['surface_temperature_celsius']['mean']
            expected = 44947 * 0.00341802 + 149.0 - 273.15
            self.assertAlmostEqual(temp, expected, places=2)
            self.assertIn('relative_heat_screening', result['analysis'])
            self.assertAlmostEqual(result['analysis']['surface_temperature_uncertainty_kelvin']['mean'],1.0,places=2)
            self.assertAlmostEqual(result['analysis']['surface_emissivity']['mean'],.96,places=3)
            self.assertAlmostEqual(result['analysis']['distance_to_cloud_km']['mean'],2.5,places=2)
            self.assertIn('thermal_quality', result['analysis'])
            self.assertGreater(result['valid_percent'], 95)
            with rasterio.open(result['stack_path']) as src:
                self.assertIn('landsat_surface_temperature_celsius', src.descriptions)
                self.assertIn('landsat_st_qa_uncertainty_kelvin', src.descriptions)
                self.assertIn('landsat_surface_emissivity', src.descriptions)
                self.assertIn('thermal_uncertainty_normalized_anomaly', src.descriptions)
                self.assertIn('relative_heat_screening', src.descriptions)

    def test_two_scenes_create_temporal_change(self):
        def searcher(spec, bbox, start, end, max_items, max_cloud):
            return [fake_item(spec,'2'), fake_item(spec,'1')]

        def processor(spec, item, grid, output_dir, event=None):
            shape=(grid['height'],grid['width'])
            level=.8 if item['id'].endswith('-2') else .2
            arr=np.full(shape,level,dtype='float32')
            arrays={'water':arr,'valid':np.ones(shape,bool)}
            if spec.kind=='optical': arrays['ndvi']=np.full(shape,.1,dtype='float32')
            return {'satellite':spec.key,'label':spec.label,'kind':spec.kind,'scene_id':item['id'],
                    'collection':item['collection'],'platform':spec.key,'acquired_at':item['properties']['datetime'],
                    'cloud_cover_percent':5,'valid_percent':100.0,'stack_path':'x','preview_path':'x',
                    'analysis':{},'asset_keys':{},'started_at':item['properties']['datetime'],
                    'finished_at':item['properties']['datetime'],'elapsed_seconds':0.0,'_arrays':arrays}

        with tempfile.TemporaryDirectory() as td:
            run=run_concurrent_multisatellite(bbox=[90.39,23.79,90.41,23.81],start_date='2026-09-01',end_date='2026-09-12',
                output_dir=td,selected_satellites=['sentinel-1','sentinel-2'],max_scenes_per_satellite=2,
                workers=4,grid_pixels=128,searcher=searcher,processor=processor)
            self.assertIn('sentinel-1',run['fusion']['temporal_change'])
            self.assertGreater(run['fusion']['temporal_change']['sentinel-1']['mean_absolute_screening_change'],.5)


if __name__ == '__main__':
    unittest.main()

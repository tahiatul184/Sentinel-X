import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from highres_collection import location_aoi, search_body, site_key, PlanetArchive, ProviderError, download_asset
from highres_worker import HighresWorker

CFG = dict(auto_location_ready=True, auto_latitude=23.75, auto_longitude=90.4,
           highres_radius_km=0.5, highres_lookback_days=90, highres_max_cloud=0.2)


class HighresTests(unittest.TestCase):
    def test_location_bounds_and_isolated_site_keys(self):
        bbox, geom = location_aoi(CFG)
        self.assertTrue(bbox[0] < 90.4 < bbox[2])
        self.assertTrue(bbox[1] < 23.75 < bbox[3])
        self.assertEqual(geom['coordinates'][0][0],geom['coordinates'][0][-1])
        self.assertNotEqual(site_key(CFG),site_key(dict(CFG,auto_latitude=24)))
        with self.assertRaises(ValueError):location_aoi(dict(CFG,auto_location_ready=False))
        with self.assertRaises(ValueError):location_aoi(dict(CFG,auto_latitude=float('nan')))

    def test_search_requires_rgb_download_entitlement(self):
        body = search_body(CFG)
        self.assertEqual(body['item_types'],['SkySatCollect'])
        self.assertIn({'type':'AssetFilter','config':['ortho_visual']},body['filter']['config'])
        session=Mock()
        response=Mock(status_code=200)
        response.json.return_value={'features':[
            {'id':'not-entitled','_permissions':[]},
            {'id':'entitled','_permissions':['assets.ortho_visual:download']}]}
        session.request.return_value=response
        client=PlanetArchive('test-placeholder',session)
        self.assertEqual([x['id'] for x in client.search(CFG)],['entitled'])
        self.assertFalse(session.request.call_args.kwargs['allow_redirects'])

    def test_activation_does_not_order_or_task(self):
        session=Mock()
        response=Mock(status_code=200)
        response.json.return_value={'ortho_visual':{'status':'inactive','_links':{'activate':
            'https://api.planet.com/data/v1/activation-test'}}}
        session.request.return_value=response
        client=PlanetArchive('test-placeholder',session)
        self.assertIsNone(client.active_asset({'id':'scene'}))
        self.assertEqual(session.request.call_count,2)
        self.assertTrue(all(call.args[0]=='GET' for call in session.request.call_args_list))
        with self.assertRaises(ProviderError):client.request('GET','https://example.com/key-leak')
        with self.assertRaises(ProviderError):download_asset('http://localhost/private','unused.tif')

    def test_worker_waits_for_location_and_credentials(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{},clear=True):
            worker=HighresWorker(Path(d))
            worker.tick(dict(CFG,auto_location_ready=False))
            self.assertEqual(worker.store.setting('highres_collection_status')['state'],'LOCATION_REQUIRED')
            worker.tick(CFG)
            state=worker.store.setting('highres_collection_status')
            self.assertEqual(state['state'],'CREDENTIAL_REQUIRED')
            self.assertNotIn('api_key',state)

    def test_worker_no_accessible_coverage(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'PL_API_KEY':'test-placeholder'}), patch('highres_worker.PlanetArchive') as factory:
            factory.return_value.search.return_value=[]
            worker=HighresWorker(Path(d))
            worker.tick(CFG)
            self.assertEqual(worker.store.setting('highres_collection_status')['state'],'NO_ACCESSIBLE_COVERAGE')
            factory.return_value.active_asset.assert_not_called()
            worker.tick(CFG)
            factory.return_value.search.assert_called_once()

    def test_download_detect_and_reuse_cached_scene(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ,{'PL_API_KEY':'test-placeholder'}), \
             patch('highres_worker.PlanetArchive') as factory, \
             patch('highres_worker.download_asset') as download, \
             patch('highres_worker.crop_rgb') as crop, \
             patch('aircraft_detector.infer_geotiff') as infer:
            factory.return_value.search.return_value=[{'id':'scene-a', 'properties':{'acquired':'2026-09-20T00:00:00Z','cloud_cover':0.1}}]
            factory.return_value.active_asset.return_value='https://download.planet.com/test'
            download.side_effect=lambda _url,path:Path(path).write_bytes(b'scene')
            def crop_fake(_source,dest,_cfg):
                Path(dest).write_bytes(b'crop')
                return dict(product_pixel_spacing_m=0.5,full_aoi_valid=True,location_pixel_valid=True)
            crop.side_effect=crop_fake
            infer.return_value=dict(rows=[],state='NO DETECTION / ABSENCE UNDETERMINED',detections=0,
                                    model_id='test',model_sha256='test-digest',source_sha256='image-digest')
            worker=HighresWorker(Path(d))
            worker.tick(CFG)
            status=worker.store.setting('highres_collection_status')
            self.assertEqual(status['state'],'UP_TO_DATE')
            self.assertEqual(status['acquired_at'],'2026-09-20T00:00:00+00:00')
            self.assertTrue(Path(status['path']).exists())
            self.assertEqual(worker.store.setting('aircraft_awareness_latest')['state'],'NO OBSERVATIONS')
            worker.store.setting('highres_request','force-refresh')
            worker.tick(CFG)
            download.assert_called_once()
            infer.assert_called_once()


if __name__=='__main__':unittest.main()

"""HTTP serialization/error regressions. No external network or raster fixtures."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import requests
from collection_store import DEFAULTS
from satellite_downloader import SEARCH_URL, COLLECTIONS, Downloader, SearchRejected, search_scenes

NOW = datetime(2026, 9, 11, tzinfo=timezone.utc)
CFG = dict(DEFAULTS, auto_collect_enabled=True, auto_location_ready=True,
           auto_latitude=23.8103, auto_longitude=90.4125)

def scene(name='sample', cloud=10, acquired='2026-09-10T06:00:00Z'):
    return {'id':name, 'collection':COLLECTIONS[0], 'bbox':[90,23,91,24],
            'properties':{'datetime':acquired, 'eo:cloud_cover':cloud}}

class Adapter(requests.adapters.BaseAdapter):
    def __init__(self, responses):
        self.responses=list(responses); self.sent=[]
    def send(self, request, **kwargs):
        self.sent.append(request)
        status, body=self.responses.pop(0)
        response=requests.Response();response.status_code=status
        response._content=json.dumps(body).encode();response.request=request
        response.url=request.url;response.headers['Content-Type']='application/json'
        return response
    def close(self):pass

def session(responses):
    s=requests.Session();adapter=Adapter(responses);s.mount('https://',adapter)
    return s,adapter

class SearchFixTests(unittest.TestCase):
    def test_actual_requests_json_serialization(self):
        s,a=session([(200,{'features':[scene()]})])
        self.assertEqual(len(search_scenes(CFG,s,NOW)),1)
        request=a.sent[0];body=json.loads(request.body)
        self.assertEqual(request.method,'POST');self.assertEqual(request.url,SEARCH_URL)
        self.assertEqual(request.headers['Content-Type'],'application/json')
        self.assertEqual(body['collections'],COLLECTIONS)
        self.assertTrue(all(isinstance(v,float) for v in body['bbox']))
        self.assertEqual(body['sortby'],[{'field':'datetime','direction':'desc'}])
        self.assertIsInstance(body['query'],dict)

    def test_400_retries_core_and_filters_locally(self):
        s,a=session([(400,{'message':'unsupported sort'}),(200,{'features':[scene('cloudy',99),scene()]})])
        self.assertEqual([x['id'] for x in search_scenes(CFG,s,NOW)],['sample'])
        body=json.loads(a.sent[1].body)
        self.assertNotIn('sortby',body);self.assertNotIn('query',body)
        self.assertEqual(body['bbox'],json.loads(a.sent[0].body)['bbox'])

    def test_core_pagination_returns_sorted_usable_scenes(self):
        s,a=session([(400,{}),(200,{'features':[scene('cloudy',99)],'links':[
            {'rel':'next','href':SEARCH_URL,'method':'POST','merge':True,'body':{'token':'page2'}}]}),
            (200,{'features':[scene('old',acquired='2026-09-09T06:00:00Z'),scene('new')]})])
        self.assertEqual([x['id'] for x in search_scenes(CFG,s,NOW)],['new','old'])
        self.assertEqual(json.loads(a.sent[2].body)['token'],'page2')

    def test_provider_explanation_survives_rejection(self):
        s,a=session([(400,{'message':'bad query'}),(400,{'description':'collection unavailable'})])
        with self.assertRaisesRegex(SearchRejected,'HTTP 400: collection unavailable'):
            search_scenes(CFG,s,NOW)

    def test_503_does_not_trigger_bad_request_fallback(self):
        s,a=session([(503,{'message':'Service unavailable'})])
        with self.assertRaisesRegex(SearchRejected,'503'):search_scenes(CFG,s,NOW)
        self.assertEqual(len(a.sent),1)

    def test_untrusted_pagination_rejected(self):
        s,a=session([(400,{}),(200,{'features':[],'links':[{'rel':'next','href':'https://example.org/private'}]})])
        with self.assertRaisesRegex(ValueError,'unsupported search endpoint'):search_scenes(CFG,s,NOW)

    def test_dashboard_status_identifies_http_rejection(self):
        s,a=session([(400,{}),(400,{'message':'invalid field'})])
        with tempfile.TemporaryDirectory() as directory:
            downloader=Downloader(Path(directory),session=s,clock=lambda:NOW.timestamp())
            downloader.tick(CFG)
            status=downloader.store.setting('satellite_download')
            self.assertEqual(status['state'],'ERROR')
            self.assertIn('provider rejected',status['message'])
            self.assertIn('invalid field',status['error'])
            self.assertGreater(status['next_check'],NOW.timestamp())

if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from collection_store import DEFAULTS, Store
from deployment_health import local_readiness, live_provider_probe

class Response:
    status_code=200
    def raise_for_status(self): pass

class DeploymentHealthTests(unittest.TestCase):
    def test_local_readiness_reports_components(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=dict(DEFAULTS)
            cfg.update(auto_collect_enabled=True,ops_automation_enabled=True,auto_location_ready=True,
                       auto_latitude=23.8,auto_longitude=90.4,multi_satellites=['sentinel-1','sentinel-2'])
            (root/'collection_config.json').write_text(json.dumps(cfg),encoding='utf-8')
            store=Store(root);now=2_000_000_000
            store.setting('supervisor',{'services':{k:'RUNNING' for k in ['imagery','multisatellite','operations','highres','dashboard']}})
            store.setting('ops_latest_report',{'epoch':now-60})
            store.setting('weather_latest',{'_fetched_epoch':now-60})
            store.setting('multisatellite_download',{'state':'UP_TO_DATE'})
            store.setting('ops_status',{'state':'UP_TO_DATE'})
            store.setting('aerosentinel_latest',{'predictive_early_warning':{'status':'MONITOR'},'operational_aviation_risk':{'status':'MONITOR'},'deployment_readiness':{'deployment_readiness_score':0.8}})
            result=local_readiness(root,now=now)
            self.assertEqual(result['state'],'READY')
            self.assertEqual(result['passed'],result['total'])

    def test_provider_probe_uses_saved_location(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=dict(DEFAULTS)
            cfg.update(auto_location_ready=True,auto_latitude=23.8,auto_longitude=90.4,
                       multi_satellites=['sentinel-1','sentinel-2'])
            (root/'collection_config.json').write_text(json.dumps(cfg),encoding='utf-8')
            calls=[]
            def get(url,**kwargs): calls.append((url,kwargs));return Response()
            result=live_provider_probe(root,get=get)
            self.assertTrue(result['ok']);self.assertEqual(len(calls),2)

if __name__=='__main__':unittest.main()

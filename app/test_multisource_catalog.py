import unittest
from datetime import date
from multisource_catalog import bbox_around, discover_multisource


class FakeResponse:
    def __init__(self, payload): self.payload=payload
    def raise_for_status(self): return None
    def json(self): return self.payload


class FakeSession:
    def __init__(self): self.headers={}; self.post_calls=[]; self.get_calls=[]
    def post(self, url, json=None, timeout=None):
        self.post_calls.append((url,json))
        collection=json['collections'][0]
        item={
            'id':f'{collection}-1','collection':collection,
            'properties':{'datetime':'2026-09-10T06:00:00Z','eo:cloud_cover':12},
            'assets':{'data':{'href':'https://example.test/data.tif'}},
        }
        return FakeResponse({'features':[item]})
    def get(self, url, params=None, timeout=None):
        self.get_calls.append((url,params))
        return FakeResponse({'hourly':{
            'time':['2026-09-10T00:00','2026-09-10T01:00'],
            'precipitation':[1.0,2.0],'rain':[1.0,2.0],
        }})


class MultisourceCatalogTests(unittest.TestCase):
    def test_bbox(self):
        bbox=bbox_around(23.8,90.4,5)
        self.assertEqual(len(bbox),4)
        self.assertLess(bbox[0],90.4); self.assertGreater(bbox[2],90.4)

    def test_discovers_all_architecture_sources(self):
        session=FakeSession()
        result=discover_multisource(23.8,90.4,date(2026,9,1),date(2026,9,10),5,session=session)
        self.assertTrue(result['all_groups_discovered'])
        self.assertEqual(set(result['source_groups']),{
            'Sentinel-1 SAR','Sentinel-2 Optical','DEM','Weather / Rainfall','Historical imagery'
        })
        self.assertEqual(result['source_groups']['Weather / Rainfall']['summary']['precipitation_sum_mm'],3.0)


if __name__=='__main__':
    unittest.main()

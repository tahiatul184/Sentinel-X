import tempfile
from pathlib import Path
import unittest

from resilience_ops import compute_resilience_assessment, write_briefing


class ResilienceOpsTests(unittest.TestCase):
    def test_assessment_combines_satellite_weather_and_assets(self):
        sat={'finished_at':'2026-09-12T06:00:00+00:00','fusion':{
            'mean_wetness_screening':0.75,'mean_fusion_uncertainty':0.2,'mean_source_coverage':1.0,
            'temporal_alignment_score':0.9,'available_satellites':['sentinel-1','sentinel-2','landsat-8-9'],
            'acquisition_skew_hours':8,'preview_path':'x.png',
            'temporal_change':{'sentinel-1':{'mean_absolute_screening_change':0.4}}}}
        weather={'fetched_at':'2099-01-01T00:00:00+00:00','summary':{
            'rain_next_6h_mm':30,'rain_next_24h_mm':90,'max_wind_next_6h_kmh':25,
            'max_gust_next_6h_kmh':45,'min_visibility_next_6h_m':5000,
            'thunderstorm_hours_next_24h':2,'max_cape_next_24h_jkg':1200,'max_temperature_next_24h_c':35}}
        assets=[{'payload':{'name':'Drain 1','type':'Drainage','site':'A','criticality':5}}]
        result=compute_resilience_assessment(sat,weather,assets,'Test AOI','disaster_response',terrain={'summary':{'p90_slope_deg':4,'relief_p05_p95_m':20}})
        self.assertEqual(result['profile'],'disaster_response')
        self.assertGreater(result['scores']['airfield_environment_screening'],0.3)
        self.assertIn('monsoon_flood_screening',result['scores'])
        self.assertEqual(result['asset_screening'][0]['name'],'Drain 1')
        self.assertTrue(result['alerts'])
        self.assertIn('No targeting',result['limitations'][-1])

    def test_briefing_writes_html_and_json(self):
        result=compute_resilience_assessment({}, {'fetched_at':'2099-01-01T00:00:00+00:00','summary':{}}, [], 'Test')
        with tempfile.TemporaryDirectory() as td:
            paths=write_briefing(result,td)
            self.assertTrue(Path(paths['latest_html']).exists())
            self.assertTrue(Path(paths['latest_json']).exists())
            html=Path(paths['latest_html']).read_text()
            self.assertIn('Automated Resilience Briefing',html)


if __name__=='__main__':unittest.main()

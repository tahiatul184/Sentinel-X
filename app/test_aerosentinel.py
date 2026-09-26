import unittest
from aerosentinel import run_aerosentinel, rare_event_screen, cross_sensor_generalization


def obs(i, anomaly=False):
    base={
        'wetness_screening':.20,'temporal_change_screening':.10,'fusion_uncertainty':.12,
        'source_coverage':1.0,'temporal_alignment':.95,'optical_ndvi':.60,
        'rain_pressure':.15,'wind_pressure':.12,'visibility_pressure':.10,'convective_pressure':.05,
        'airfield_environment_screening':.20,'disaster_response_screening':.18,
        'infrastructure_continuity_screening':.80,
    }
    if anomaly:
        base.update(wetness_screening=.90,temporal_change_screening=.88,rain_pressure=.90,
                    wind_pressure=.75,visibility_pressure=.70,convective_pressure=.65,
                    airfield_environment_screening=.86,disaster_response_screening=.78,
                    infrastructure_continuity_screening=.35)
    return {'timestamp':f'2026-09-{i+1:02d}T00:00:00+00:00','features':base,
            'quality':{'source_count':3,'source_coverage':1,'temporal_alignment':.95,'mean_valid_percent':95,
                       'mean_optical_cloud_percent':10,'weather_available':True,'terrain_available':True},
            'observed_evidence':{'wetness_screening':base['wetness_screening'],
                                 'temporal_change_screening':base['temporal_change_screening']},
            'signature':f's{i}'}


def run_record():
    return {
        'elapsed_seconds':12.5,
        'requested_satellites':['sentinel-1','sentinel-2','landsat-8-9'],
        'scene_results':[
            {'satellite':'sentinel-1','native_gsd_m':10,'kind':'sar','elapsed_seconds':4.1},
            {'satellite':'sentinel-2','native_gsd_m':10,'kind':'optical','elapsed_seconds':5.2},
            {'satellite':'landsat-8-9','native_gsd_m':30,'kind':'optical','elapsed_seconds':6.4},
        ],
        'concurrency':{'configured_workers':3,'peak_scene_workers':3},
        'fusion':{
            'available_satellites':['sentinel-1','sentinel-2','landsat-8-9'],
            'requested_satellites':['sentinel-1','sentinel-2','landsat-8-9'],
            'mean_source_coverage':1.0,'temporal_alignment_score':.92,
            'mean_fusion_uncertainty':.16,'acquisition_skew_hours':8.0,
        }
    }


class AeroSentinelTests(unittest.TestCase):
    def test_full_aviation_safety_contract(self):
        rows=[obs(i) for i in range(12)] + [obs(12,True)]
        assessment={'scores':{
            'airfield_environment_screening':.82,
            'weather_environment_screening':.72,
            'airfield_surface_screening':.78,
            'infrastructure_continuity_screening':.40,
        }}
        result=run_aerosentinel(
            rows,multisatellite_run=run_record(),assessment=assessment,
            abstain_threshold=.3,area_name='Test Airfield',
            deployment_context={'collection_interval_minutes':30},
        )
        self.assertEqual(result['framework'],'AEROSENTINEL')
        self.assertTrue(result['version'].startswith('AEROSENTINEL v2.9.2'))
        self.assertIn('rare_event_detection',result)
        self.assertIn('cross_sensor_generalization',result)
        self.assertIn('operational_aviation_risk',result)
        self.assertIn('hazard_prediction',result)
        self.assertIn('explainability',result)
        self.assertIn('deployment_readiness',result)
        self.assertGreater(result['operational_aviation_risk']['risk_score'],.4)
        self.assertEqual(len(result['hazard_prediction']['forecast_next_3_observations']),3)
        self.assertEqual(result['cross_sensor_generalization']['available_sensors'],['sentinel-1','sentinel-2','landsat-8-9'])
        self.assertGreater(result['deployment_readiness']['assessment_compute_latency_ms'],0)
        self.assertIn(result['operational_aviation_risk']['status'],{'MONITOR','ELEVATED','HIGH REVIEW'})

    def test_cross_sensor_screen_flags_single_source(self):
        rows=[obs(i) for i in range(8)]
        result=run_aerosentinel(rows,multisatellite_run={'scene_results':[]},assessment={'scores':{}},abstain_threshold=.2)
        cross=cross_sensor_generalization({'requested_satellites':['sentinel-1','sentinel-2'],
                                           'fusion':{'available_satellites':['sentinel-1'],'requested_satellites':['sentinel-1','sentinel-2']}},result)
        self.assertEqual(cross['state'],'LIMITED / SINGLE-SOURCE')

    def test_rare_event_reports_baseline_limit(self):
        legacy={'anomaly_path':{'feature_deviation':{'x':7.0},'current_anomaly_score':.9,'ood_score':.9,'baseline_rows':2}}
        rare=rare_event_screen(legacy)
        self.assertEqual(rare['state'],'INSUFFICIENT BASELINE')
        self.assertEqual(rare['features_above_6sigma'],1)

if __name__=='__main__':
    unittest.main()

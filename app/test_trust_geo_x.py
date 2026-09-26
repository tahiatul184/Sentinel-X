from pathlib import Path
import tempfile
import unittest

from trust_geo_x import (
    build_observation, fit_temperature, resolution_feasibility,
    run_trust_geo_x, resolution_research_table, domain_reliability_evaluation,
)


def obs(i, anomaly=False):
    base={
        'wetness_screening':.20,'temporal_change_screening':.10,'fusion_uncertainty':.12,
        'source_coverage':1.0,'temporal_alignment':.95,'optical_ndvi':.60,
        'rain_pressure':.15,'wind_pressure':.12,'visibility_pressure':.10,'convective_pressure':.05,
        'airfield_environment_screening':.20,'disaster_response_screening':.18,
        'infrastructure_continuity_screening':.20,
    }
    if anomaly:
        base.update(wetness_screening=.85,temporal_change_screening=.82,rain_pressure=.90,
                    airfield_environment_screening=.80,disaster_response_screening=.78)
    return {'timestamp':f'2026-09-{i+1:02d}T00:00:00+00:00','features':base,
            'quality':{'source_count':3,'source_coverage':1,'temporal_alignment':.95,'mean_valid_percent':95,
                       'mean_optical_cloud_percent':10,'weather_available':True,'terrain_available':True},
            'observed_evidence':{},'signature':f's{i}'}


class TrustGeoXTests(unittest.TestCase):
    def test_resolution_gate_rejects_sentinel_scale(self):
        self.assertFalse(resolution_feasibility(10.0)['feasible'])
        self.assertTrue(resolution_feasibility(0.3)['feasible'])
        self.assertGreater(len(resolution_research_table()),10)

    def test_open_world_anomaly_temporal_prediction_and_abstention(self):
        rows=[obs(i) for i in range(12)] + [obs(12,True)]
        run={'scene_results':[{'native_gsd_m':10,'kind':'optical'}]}
        result=run_trust_geo_x(rows,multisatellite_run=run,abstain_threshold=.3,area_name='AOI-X')
        self.assertGreater(result['anomaly_path']['current_anomaly_score'],.5)
        self.assertIn(result['predictive_early_warning']['status'],{'ELEVATED','INVESTIGATE'})
        self.assertFalse(result['uncertainty_engine']['abstain'])
        self.assertEqual(result['direct_path']['status'],'INSUFFICIENT RESOLUTION')
        self.assertEqual(len(result['reliability_stress_suite']),5)
        self.assertTrue(result['architecture']['human_review'])

    def test_immature_baseline_abstains(self):
        result=run_trust_geo_x([obs(0)],multisatellite_run={'scene_results':[]})
        self.assertTrue(result['uncertainty_engine']['abstain'])
        self.assertEqual(result['predictive_early_warning']['status'],'ABSTAIN / HUMAN REVIEW')

    def test_weather_history_cannot_replace_satellite_coverage(self):
        rows=[obs(i) for i in range(12)]
        for row in rows:
            row['quality']['source_count']=0;row['quality']['source_coverage']=0;row['quality']['temporal_alignment']=0
        result=run_trust_geo_x(rows,multisatellite_run={'scene_results':[]})
        self.assertLessEqual(result['uncertainty_engine']['reliability'],.20)
        self.assertTrue(result['uncertainty_engine']['abstain'])

    def test_temperature_calibration(self):
        scores=[.05,.08,.1,.15,.2,.25,.35,.65,.75,.8,.85,.9,.92,.95]
        labels=[0,0,0,0,0,0,0,1,1,1,1,1,1,1]
        model=fit_temperature(scores,labels)
        self.assertEqual(model['method'],'temperature_scaling')
        self.assertGreater(model['validation_rows'],12)

    def test_domain_reliability_evaluation(self):
        rows=[]
        for i in range(12):
            rows.append({'region':'A' if i<6 else 'B','season':'monsoon' if i%2 else 'dry',
                         'condition':'cloud' if i%3==0 else 'normal','score':.8 if i%2 else .2,
                         'label':1 if i%2 else 0,'reliability':.8 if i!=5 else .3})
        result=domain_reliability_evaluation(rows,.45)
        self.assertEqual(result['overall']['rows'],12)
        self.assertGreaterEqual(len(result['groups']),6)

    def test_observation_contract(self):
        run={'finished_at':'2026-09-13T00:00:00+00:00','requested_satellites':['sentinel-1','sentinel-2'],
             'scene_results':[{'kind':'optical','cloud_cover_percent':10,'valid_percent':90,'native_gsd_m':10}],
             'fusion':{'available_satellites':['sentinel-1','sentinel-2'],'requested_satellites':['sentinel-1','sentinel-2'],
                       'mean_source_coverage':1,'temporal_alignment_score':.9,'mean_fusion_uncertainty':.2,
                       'mean_wetness_screening':.4,'mean_optical_ndvi':.2,'temporal_change':{}}}
        weather={'summary':{'rain_next_24h_mm':20,'max_gust_next_6h_kmh':30,'min_visibility_next_6h_m':8000,'thunderstorm_hours_next_24h':1}}
        assessment={'scores':{'airfield_environment_screening':.4,'disaster_response_screening':.3,'infrastructure_continuity_screening':.2}}
        o=build_observation(run,weather,{'summary':{'mean_slope_deg':2}},assessment,area_name='A')
        self.assertEqual(o['quality']['source_count'],2)
        self.assertEqual(o['quality']['best_native_gsd_m'],10)
        self.assertIn('signature',o)


if __name__=='__main__':unittest.main()

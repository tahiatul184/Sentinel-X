import unittest
import numpy as np

from registration_quality import estimate_translation
from change_detection import composite_change, registration_aware_composite_change
from reliability_fusion import weighted_fusion, scene_reliability
from robustness_profile import image_corruption_profile
from trust_geo_x import resolution_feasibility, run_trust_geo_x
from provenance import build_provenance_manifest
from compute_policy import choose_processing_tier


class ReliabilityUpgradeTests(unittest.TestCase):
    def test_phase_registration_detects_known_shift(self):
        ref=np.zeros((64,64),dtype='float32')
        ref[12:26,18:31]=1.0; ref[40:48,42:55]=0.55
        moving=np.roll(np.roll(ref,3,axis=0),-2,axis=1)
        reg=estimate_translation(ref,moving,np.ones_like(ref,dtype=bool),max_shift_px=6)
        self.assertTrue(reg['accepted'])
        self.assertAlmostEqual(reg['shift_x_px'],2.0,delta=.35)
        self.assertAlmostEqual(reg['shift_y_px'],-3.0,delta=.35)
        self.assertGreater(reg['confidence'],.5)

    def test_registration_aware_change_reduces_shift_false_positive(self):
        ref=np.zeros((64,64),dtype='float32')
        ref[10:24,10:28]=.9; ref[36:50,40:54]=.35
        prev=np.roll(np.roll(ref,2,axis=0),-2,axis=1)
        latest={'valid':np.ones_like(ref,dtype=bool),'water':ref,'ndvi':ref}
        previous={'valid':np.ones_like(ref,dtype=bool),'water':prev,'ndvi':prev}
        raw,rs=composite_change(latest,previous,signals=('water','ndvi'))
        aligned,summary=registration_aware_composite_change(latest,previous,signals=('water','ndvi'))
        self.assertTrue(summary['registration']['accepted'])
        self.assertLess(float(np.nanmean(aligned)),float(np.nanmean(raw)))

    def test_weighted_fusion_downweights_weak_source(self):
        good=np.full((8,8),.2,dtype='float32')
        weak=np.full((8,8),.9,dtype='float32')
        fused,dis,eff,q=weighted_fusion([good,weak],[.95,.05])
        self.assertLess(float(np.mean(fused)),.30)
        self.assertGreater(float(np.mean(q)),.45)
        self.assertGreater(float(np.mean(dis)),0)

    def test_scene_reliability_penalizes_cloud(self):
        clear={'kind':'optical','valid_percent':95,'cloud_cover_percent':5,
               'processing_quality':{'processing_readiness_score':.9},'acquired_at':'2026-09-14T00:00:00Z'}
        cloud={**clear,'cloud_cover_percent':90}
        a=scene_reliability(clear,reference_time='2026-09-14T00:00:00Z')['weight']
        b=scene_reliability(cloud,reference_time='2026-09-14T00:00:00Z')['weight']
        self.assertGreater(a,b)

    def test_corruption_profile_reports_degradation(self):
        x=np.tile(np.arange(64,dtype='uint8'),(64,1))*4
        rgb=np.stack([x,np.flipud(x),x],axis=-1)
        profile=image_corruption_profile(rgb,np.ones((64,64),dtype=bool))
        self.assertGreaterEqual(len(profile['corruptions']),4)
        self.assertGreaterEqual(profile['worst_readiness_drop'],0)

    def test_resolution_screen_uses_mtf_and_snr_when_supplied(self):
        base=resolution_feasibility(.3,nominal_object_m=2.0)
        degraded=resolution_feasibility(.3,nominal_object_m=2.0,mtf50=.05,snr_db=4)
        self.assertTrue(base['feasible'])
        self.assertFalse(degraded['feasible'])
        self.assertLess(degraded['effective_pixels_across'],base['effective_pixels_across'])

    def test_same_season_baseline_is_preferred(self):
        def obs(ts,val):
            features={
                'wetness_screening':val,'temporal_change_screening':val,'fusion_uncertainty':.1,
                'source_coverage':1.0,'temporal_alignment':1.0,'optical_ndvi':val,
                'rain_pressure':val,'wind_pressure':.1,'visibility_pressure':.1,'convective_pressure':.1,
                'airfield_environment_screening':val,'disaster_response_screening':val,
                'infrastructure_continuity_screening':val,
            }
            return {'timestamp':ts,'season':'monsoon' if '-07-' in ts else 'dry','features':features,
                    'quality':{'source_count':3,'source_coverage':1,'temporal_alignment':1,'mean_valid_percent':95,
                               'mean_optical_cloud_percent':5,'weather_available':True,'terrain_available':True},
                    'observed_evidence':{},'signature':ts}
        rows=[obs(f'202{i}-07-10T00:00:00+00:00',.2) for i in range(2,6)]
        rows += [obs(f'202{i}-01-10T00:00:00+00:00',.9) for i in range(2,6)]
        rows.append(obs('2026-07-10T00:00:00+00:00',.22))
        result=run_trust_geo_x(rows)
        self.assertEqual(result['anomaly_path']['baseline_strategy'],'SAME_SEASON')
        self.assertEqual(result['anomaly_path']['baseline_rows'],4)

    def test_provenance_and_compute_policy(self):
        manifest=build_provenance_manifest({'scene_results':[{'satellite':'sentinel-2','scene_id':'x','acquired_at':'2026-01-01T00:00:00Z'}]})
        self.assertEqual(manifest['record_count'],1)
        policy=choose_processing_tier({'uncertainty_engine':{'uncertainty':.8,'reliability':.2,'abstain':True},
                                       'predictive_early_warning':{'score':.7},
                                       'data_quality_and_fusion':{'registration_quality':{'confidence':.1}}})
        self.assertEqual(policy['recommended_tier'],'HUMAN_REVIEW_OR_REACQUIRE')


if __name__=='__main__':
    unittest.main()

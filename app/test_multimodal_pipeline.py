import unittest
from multimodal_pipeline import parse_multimodal_csv, run_multimodal_pipeline, asset_decision_rows, template_csv


class MultimodalPipelineTests(unittest.TestCase):
    def test_template_parses_and_pipeline_runs(self):
        rows = parse_multimodal_csv(template_csv())
        result = run_multimodal_pipeline(rows, forecast_horizon_steps=2)
        self.assertEqual(len(result['predictive_digital_twin']), 3)
        self.assertIn(result['latest']['audit'], {'PASS', 'REVIEW', 'ABSTAIN'})
        for key in ['current_state', 'future_prediction', 'risk_screening', 'reliability']:
            self.assertGreaterEqual(result['latest'][key], 0)
            self.assertLessEqual(result['latest'][key], 1)

    def test_missing_sources_increase_uncertainty(self):
        rich = parse_multimodal_csv(template_csv())
        sparse = [dict(timestamp=r['timestamp'], s1_vv_db=r['s1_vv_db']) for r in rich]
        a = run_multimodal_pipeline(rich)
        b = run_multimodal_pipeline(sparse)
        self.assertGreater(b['reliability_audit']['latest']['uncertainty_score'],
                           a['reliability_audit']['latest']['uncertainty_score'])
        self.assertEqual(b['latest']['audit'], 'ABSTAIN')


    def test_blank_modalities_are_supported(self):
        text = ('timestamp,s1_vv_db,s2_ndvi,rain_mm_h\n'
                '2026-09-12T06:00:00Z,-11,,3\n'
                '2026-09-13T06:00:00Z,-12,0.4,\n')
        rows = parse_multimodal_csv(text)
        result = run_multimodal_pipeline(rows)
        self.assertEqual(len(result['predictive_digital_twin']), 2)

    def test_duplicate_and_naive_timestamps_rejected(self):
        text = 'timestamp,s2_ndvi\n2026-09-12T06:00:00,0.2\n2026-09-13T06:00:00Z,0.3\n'
        with self.assertRaises(ValueError):
            parse_multimodal_csv(text)
        dup = 'timestamp,s2_ndvi\n2026-09-12T06:00:00Z,0.2\n2026-09-12T06:00:00Z,0.3\n'
        with self.assertRaises(ValueError):
            parse_multimodal_csv(dup)

    def test_asset_decision_rows(self):
        latest = run_multimodal_pipeline(parse_multimodal_csv(template_csv()))['latest']
        assets = [{'title':'Drain A','payload':{'site':'S','type':'Drainage','latitude':23,'longitude':90,'criticality':5}}]
        rows = asset_decision_rows(assets, latest)
        self.assertEqual(rows[0]['asset'], 'Drain A')
        self.assertGreaterEqual(rows[0]['screening_priority'], rows[0]['risk_screening'])


# Additional v0.3 architecture tests are defined in a separate class so the
# original regression tests remain readable.
class MultimodalArchitectureV03Tests(unittest.TestCase):
    def test_terramind_records_activate_strict_architecture(self):
        rows = parse_multimodal_csv(template_csv())
        foundation = []
        for i, row in enumerate(rows):
            foundation.append({
                'acquired_at': row['timestamp'],
                'model': 'terramind_v1_tiny',
                'modalities': ['S2L2A','S1GRD','DEM'],
                'embedding': [float(i + j/10) for j in range(16)],
            })
        result = run_multimodal_pipeline(rows, foundation_records=foundation)
        self.assertTrue(result['architecture_compliance']['strict_architecture_run'])
        self.assertEqual(result['architecture_compliance']['multimodal_foundation_model'], 'ACTIVE_TERRAMIND_HYBRID')
        self.assertEqual(result['foundation_records_used'], 3)
        self.assertEqual(set(result['multimodal_foundation_stage']['all_architecture_input_groups']), {
            'Sentinel-1 SAR','Sentinel-2 Optical','DEM','Weather / Rainfall','Historical imagery'
        })
        self.assertEqual(result['multimodal_foundation_stage']['context_adapter_inputs'], ['Weather / Rainfall','Historical imagery'])

    def test_strict_architecture_rejects_wrong_foundation_backend(self):
        rows = parse_multimodal_csv(template_csv())
        foundation = [{
            'acquired_at': row['timestamp'],
            'model': 'generic_encoder',
            'modalities': ['S2L2A','S1GRD','DEM'],
            'embedding': [float(i + j/10) for j in range(16)],
        } for i, row in enumerate(rows)]
        result = run_multimodal_pipeline(rows, foundation_records=foundation)
        self.assertFalse(result['architecture_compliance']['strict_architecture_run'])
        self.assertFalse(result['architecture_compliance']['strict_terramind_coverage'])

    def test_strict_architecture_requires_complete_five_source_coverage(self):
        rows = parse_multimodal_csv(template_csv())
        rows[1]['historical_ndvi'] = float('nan')
        rows[1]['historical_ndwi'] = float('nan')
        rows[1]['historical_nbr'] = float('nan')
        rows[1]['historical_change_score'] = float('nan')
        foundation = [{
            'acquired_at': row['timestamp'],
            'model': 'terramind_v1_tiny',
            'modalities': ['S2L2A','S1GRD','DEM'],
            'embedding': [float(i + j/10) for j in range(16)],
        } for i, row in enumerate(rows)]
        result = run_multimodal_pipeline(rows, foundation_records=foundation)
        self.assertFalse(result['architecture_compliance']['strict_architecture_run'])
        self.assertFalse(result['architecture_compliance']['strict_source_coverage'])

    def test_architecture_exposes_all_required_stages(self):
        result = run_multimodal_pipeline(parse_multimodal_csv(template_csv()))
        required = {
            'multi_source_earth_observation','sentinel_1_sar','sentinel_2_optical','dem',
            'weather_rainfall','historical_imagery','multimodal_foundation_model',
            'temporal_representation','change_detection','anomaly_detection',
            'land_surface_understanding','uncertainty_estimation','reliability_audit',
            'current_state','future_prediction','risk','predictive_digital_twin',
            'decision_support_map','airfield_risk','disaster_risk','infrastructure_resilience',
        }
        self.assertTrue(required.issubset(result['architecture_compliance']))

    def test_decision_surface_and_domain_mapping(self):
        from multimodal_pipeline import decision_surface_points
        latest = run_multimodal_pipeline(parse_multimodal_csv(template_csv()))['latest']
        assets = [
            {'title':'Runway','payload':{'site':'S','type':'Runway','latitude':23.0,'longitude':90.0,'criticality':5}},
            {'title':'Bridge','payload':{'site':'S','type':'Bridge','latitude':23.01,'longitude':90.02,'criticality':4}},
        ]
        rows = asset_decision_rows(assets, latest)
        self.assertEqual(rows[0]['domain'], 'airfield')
        self.assertEqual(rows[1]['domain'], 'infrastructure')
        surface = decision_surface_points(rows, grid_size=7)
        self.assertEqual(len(surface), 49)
        self.assertTrue(all(0 <= p['priority'] <= 1 for p in surface))


if __name__ == '__main__':
    unittest.main()

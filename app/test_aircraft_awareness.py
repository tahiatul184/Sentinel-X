import unittest

from aircraft_awareness import TEMPLATE, analyze, parse_detections


def row(site='A', scene='S1', time='2026-01-01T00:00:00Z', modality='optical',
        lat='23.1', gsd='0.5', confidence='0.9'):
    return f'{site},{scene},{time},{modality},{lat},90.1,{confidence},{gsd},20,2,validated-v1,0.9,0.1\n'


class AircraftAwarenessTests(unittest.TestCase):
    def test_rejects_non_satellite_and_naive_time(self):
        with self.assertRaisesRegex(ValueError, 'unsupported satellite modality'):
            parse_detections(TEMPLATE + row(modality='rf'))
        with self.assertRaisesRegex(ValueError, 'timezone'):
            parse_detections(TEMPLATE + row(time='2026-01-01T00:00:00'))

    def test_resolution_abstention_and_thermal_only(self):
        result = analyze(parse_detections(TEMPLATE + row(gsd='10') +
                                          row(scene='S2', modality='thermal')))
        self.assertEqual(result['state'], 'INSUFFICIENT EVIDENCE')
        self.assertFalse(result['movement_candidates'])

    def test_sparse_temporal_change_is_only_possible_movement(self):
        result = analyze(parse_detections(TEMPLATE + row() +
            row(scene='S2', time='2026-01-02T00:00:00Z', modality='sar', lat='23.1005')))
        self.assertEqual(len(result['observations']), 2)
        self.assertEqual(result['movement_candidates'][0]['state'], 'POSSIBLE MOVEMENT')
        self.assertEqual(len(result['anomalies']), 1)
        self.assertIn('cannot establish identity', result['movement_candidates'][0]['note'])

    def test_site_boundary_and_registration_uncertainty(self):
        result = analyze(parse_detections(TEMPLATE + row() +
            row(site='B', scene='S2', time='2026-01-02T00:00:00Z', lat='23.1005')))
        self.assertFalse(result['movement_candidates'])


if __name__ == '__main__':
    unittest.main()

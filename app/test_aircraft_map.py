import copy
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aircraft_awareness import analyze
from aircraft_map import candidate_rows, filter_rows, map_view, synthetic_assessment
from collection_store import Store
import collection_store


class AircraftMapTests(unittest.TestCase):
    def test_evidence_gate_and_confidence_survive_mapping(self):
        source = dict(site_id='A', scene_id='S', acquired_at='2026-01-01T00:00:00Z',
            modality='optical', latitude=1, longitude=2, confidence=.9, gsd_m=.5,
            object_length_m=20, registration_error_m=2, model_id='test', quality=1, cloud_fraction=.1)
        assessment = analyze([source, dict(source, longitude=3, confidence=.1)])
        rows, rejected = candidate_rows(assessment)
        self.assertEqual(rejected, 0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['model_confidence'], .9)
        self.assertEqual(rows[0]['screening_score'], .81)
        self.assertEqual(rows[0]['longitude'], 2)

    def test_ids_are_stable_and_duplicates_collapse(self):
        a = synthetic_assessment()
        a['observations'].append(copy.deepcopy(a['observations'][0]))
        rows, _ = candidate_rows(a)
        other, _ = candidate_rows(dict(a, observations=list(reversed(a['observations']))))
        self.assertEqual(len(rows), 3)
        self.assertEqual([r['candidate_id'] for r in rows], [r['candidate_id'] for r in other])

    def test_invalid_location_and_timezone_are_excluded(self):
        a = synthetic_assessment()
        a['observations'][0]['latitude'] = float('nan')
        a['observations'][1]['longitude'] = 181
        a['observations'][2]['acquired_at'] = '2026-01-01T12:00:00'
        rows, rejected = candidate_rows(a)
        self.assertEqual(rows, [])
        self.assertEqual(rejected, 3)

    def test_filters_do_not_fall_back_to_older_stronger_candidate(self):
        a = synthetic_assessment()
        newer = dict(a['observations'][0], acquired_at='2026-01-02T12:00:00Z', screening_score=.55)
        a['observations'].append(newer)
        rows, _ = candidate_rows(a)
        self.assertEqual(filter_rows(rows, ['DEMO-SITE'], .8, True), [])
        self.assertEqual(len(filter_rows(rows, ['DEMO-SITE'], .8, False)), 1)
        self.assertEqual(filter_rows(rows, [], 0, False), [])

    def test_utc_normalization_and_dateline_view(self):
        a = synthetic_assessment()
        a['observations'][0].update(acquired_at='2026-01-01T18:00:00+06:00')
        rows, _ = candidate_rows(a)
        self.assertTrue(all(datetime.fromisoformat(r['acquired_at']).hour == 12 for r in rows))
        view = map_view([dict(latitude=0, longitude=179.9), dict(latitude=0, longitude=-179.9)])
        self.assertAlmostEqual(abs(view['longitude']), 180)
        self.assertGreater(view['zoom'], 8)


class AircraftMapUITests(unittest.TestCase):
    def test_saved_detections_demo_filters_and_refresh(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'app'; root.mkdir()
            saved = Store(root)
            real = synthetic_assessment()
            real.update(source='optical_geotiff_detector', data_origin='Test fixture')
            saved.setting('aircraft_awareness_latest', real)
            with patch.object(collection_store, 'ROOT', root):
                at = AppTest.from_file(str(Path(__file__).parent / 'dashboard.py'), default_timeout=25).run()
                self.assertFalse(at.exception)
                self.assertFalse(at.error)
                self.assertEqual(at.sidebar.radio[0].value, 'Aircraft Map')
                self.assertEqual(at.metric[0].value, '3')
                self.assertTrue(at.get('deck_gl_json_chart'))
                import streamlit as st
                chart = st.pydeck_chart
                target = candidate_rows(real)[0][1]['candidate_id']
                def selected_chart(*args, **kwargs):
                    chart(*args, **kwargs)
                    return {'selection': {'objects': {'aircraft-observations': [
                        {'candidate_id': target, 'site_id': 'UNTRUSTED BROWSER FIELD'}]}}}
                with patch.object(st, 'pydeck_chart', side_effect=selected_chart):
                    at.run()
                self.assertFalse(at.exception)
                self.assertEqual(at.selectbox(key='aircraft_map_selected').value, target)
                self.assertFalse(any('UNTRUSTED BROWSER FIELD' in m.value for m in at.markdown))
                at.slider(key='aircraft_map_score').set_value(.85).run()
                self.assertFalse(at.exception)
                self.assertEqual(at.metric[0].value, '1')
                at.multiselect(key='aircraft_map_sites').set_value([]).run()
                self.assertEqual(at.metric[0].value, '0')
                at.toggle(key='aircraft_map_demo').set_value(True).run()
                self.assertTrue(any('SYNTHETIC PREVIEW' in w.value for w in at.warning))
                self.assertEqual(saved.setting('aircraft_awareness_latest'), real)
                # Change the stored result as a background worker would.
                at.toggle(key='aircraft_map_demo').set_value(False).run()
                saved.setting('aircraft_awareness_latest', dict(state='NO OBSERVATIONS', observations=[]))
                at.run()
                self.assertFalse(at.exception)
                self.assertFalse(at.error)
                self.assertEqual(at.metric[0].value, '0')
                self.assertFalse(at.get('deck_gl_json_chart'))


if __name__ == '__main__':
    unittest.main()

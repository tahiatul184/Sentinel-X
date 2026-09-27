import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests
from aircraft_map import candidate_rows, synthetic_assessment
from collection_store import Store
import collection_store
from flight_comparison import (API_URL, FlightDataError, compare, fetch_historical,
                               normalize_records, parse_upload, query_bounds)


class FlightComparisonTests(unittest.TestCase):
    def setUp(self):
        self.candidate = candidate_rows(synthetic_assessment())[0][0]
        self.raw = dict(fr24_id='fixture-flight', timestamp=self.candidate['acquired_at'],
                        lat=self.candidate['latitude'], lon=self.candidate['longitude'],
                        callsign='TEST001', reg='TEST-ONLY', source='ADSB', squawk='0123')

    def test_broadcast_match_is_possible_not_confirmed_identity(self):
        result = compare(self.candidate, normalize_records([self.raw]))
        self.assertEqual(result['status'], 'POSSIBLE BROADCAST MATCH')
        self.assertIn('unconfirmed', result['transponder_assessment'])
        self.assertEqual(result['matches'][0]['squawk'], '0123')
        self.assertEqual(result['matches'][0]['distance_m'], 0)

    def test_absence_old_records_and_far_records_never_mean_off(self):
        cases = [([], 'NO RECORDS RETURNED'),
                 ([dict(self.raw, timestamp='2000-01-01T00:00:00Z')], 'NO TIME-ALIGNED RECORDS'),
                 ([dict(self.raw, lat=45)], 'NO MATCHING FLIGHT RECORD')]
        for rows, expected in cases:
            with self.subTest(expected=expected):
                result = compare(self.candidate, normalize_records(rows))
                self.assertEqual(result['status'], expected)
                self.assertEqual(result['transponder_assessment'], 'UNDETERMINED')
                self.assertFalse(result['matches'])

    def test_estimated_position_is_not_broadcast_evidence(self):
        result = compare(self.candidate, normalize_records([dict(self.raw, source='ESTIMATION')]))
        self.assertEqual(result['status'], 'POSSIBLE ESTIMATED MATCH')
        self.assertEqual(result['transponder_assessment'], 'UNDETERMINED')

    def test_multiple_flights_ambiguous_same_flight_positions_deduplicated(self):
        rows = normalize_records([self.raw, self.raw, dict(self.raw, fr24_id='another-flight')])
        result = compare(self.candidate, rows)
        self.assertEqual(result['status'], 'AMBIGUOUS MATCH')
        self.assertEqual(result['match_count'], 2)
        self.assertEqual(result['transponder_assessment'], 'UNDETERMINED')

    def test_invalid_snapshot_fields_and_missing_csv_header(self):
        for replacement in [dict(lat=float('nan')), dict(lon=181), dict(timestamp='2026-01-01'), dict(fr24_id='')]:
            with self.subTest(replacement=replacement):
                with self.assertRaises(FlightDataError):normalize_records([dict(self.raw, **replacement)])
        with self.assertRaises(FlightDataError):parse_upload(b'garbage\n', 'bad.csv')
        with self.assertRaises(FlightDataError):parse_upload(b'{}', 'bad.json')
        with self.assertRaises(FlightDataError):parse_upload(b'x'*5_000_001, 'large.json')

    def test_official_json_and_csv_aliases(self):
        self.assertEqual(parse_upload(json.dumps({'data':[self.raw]}).encode(), 'fr24.json')[0]['fr24_id'], 'fixture-flight')
        rows = parse_upload(b'flight_id,timestamp,latitude,longitude,source\nA,1767268800,0,0,MLAT\n', 'snapshot.csv')
        self.assertEqual(rows[0]['source'], 'MLAT')
        self.assertTrue(rows[0]['timestamp'].endswith('+00:00'))

    def test_historical_request_uses_capture_time_and_bounded_area(self):
        response = Mock(status_code=200);response.json.return_value = {'data':[self.raw]}
        get = Mock(return_value=response)
        result = fetch_historical(self.candidate, 'test-token-placeholder', get=get)
        args = get.call_args
        self.assertEqual(args.args[0], API_URL)
        self.assertIn('/historic/', args.args[0])
        self.assertEqual(args.kwargs['params']['timestamp'], 1767268800)
        self.assertEqual(args.kwargs['params']['limit'], 100)
        self.assertEqual(args.kwargs['headers']['Accept-Version'], 'v1')
        self.assertFalse(args.kwargs['allow_redirects'])
        north, south, west, east = map(float, args.kwargs['params']['bounds'].split(','))
        self.assertLess(south, self.candidate['latitude']);self.assertGreater(north, self.candidate['latitude'])
        self.assertLess(west, self.candidate['longitude']);self.assertGreater(east, self.candidate['longitude'])
        self.assertEqual(len(result['records']), 1)
        response.close.assert_called_once()

    def test_failed_request_cannot_become_no_match_or_leak_token(self):
        for code in [302, 401, 402, 403, 429, 500]:
            response = Mock(status_code=code)
            with self.assertRaises(FlightDataError) as caught:
                fetch_historical(self.candidate, 'test-token-placeholder', get=Mock(return_value=response))
            self.assertNotIn('test-token-placeholder', str(caught.exception))
            response.json.assert_not_called()
        with self.assertRaises(FlightDataError) as caught:
            fetch_historical(self.candidate, 'test-token-placeholder', get=Mock(side_effect=requests.ConnectionError('test-token-placeholder')))
        self.assertNotIn('test-token-placeholder', str(caught.exception))

    def test_result_cap_is_reported_and_date_line_queries_rejected(self):
        response = Mock(status_code=200);response.json.return_value = {'data':[self.raw]*100}
        self.assertTrue(fetch_historical(self.candidate, 'test-placeholder', get=Mock(return_value=response))['incomplete'])
        with self.assertRaises(FlightDataError):query_bounds(dict(self.candidate, longitude=180), 500)
        with self.assertRaises(FlightDataError):fetch_historical(self.candidate, '')


class FlightComparisonUITests(unittest.TestCase):
    def test_synthetic_results_stay_in_session_and_no_token_disables_api(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {'FR24_API_TOKEN':''}):
            root = Path(td)/'app';root.mkdir()
            saved = Store(root)
            original = synthetic_assessment();original.update(source='test-fixture', data_origin='Test fixture')
            saved.setting('aircraft_awareness_latest', original)
            with patch.object(collection_store, 'ROOT', root), patch('flight_comparison.fetch_historical') as request:
                at = AppTest.from_file(str(Path(__file__).parent/'dashboard.py'), default_timeout=25).run()
                self.assertFalse(at.exception)
                self.assertTrue(at.button(key='flight_compare_api').disabled)
                at.toggle(key='aircraft_map_demo').set_value(True).run()
                at.button(key='flight_compare_demo').click().run()
                self.assertFalse(at.exception)
                self.assertFalse(at.error)
                self.assertTrue(any('POSSIBLE BROADCAST MATCH' in m.value for m in at.markdown))
                self.assertTrue(any('unconfirmed' in m.value for m in at.markdown))
                at.number_input(key='flight_match_radius').set_value(1000).run()
                self.assertTrue(any('settings changed' in i.value for i in at.info))
                at.toggle(key='aircraft_map_demo').set_value(False).run()
                self.assertFalse(any('POSSIBLE BROADCAST MATCH' in m.value for m in at.markdown))
                self.assertEqual(saved.setting('aircraft_awareness_latest'), original)
                self.assertIsNone(saved.setting('flight_comparisons'))
                request.assert_not_called()


    def test_api_only_runs_on_click_and_failure_clears_old_match(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {'FR24_API_TOKEN':'test-placeholder'}):
            root = Path(td)/'app';root.mkdir()
            saved = Store(root);assessment = synthetic_assessment()
            saved.setting('aircraft_awareness_latest', assessment)
            candidate = candidate_rows(assessment)[0][0]
            records = normalize_records([dict(fr24_id='TEST-ONLY', timestamp=candidate['acquired_at'],
                lat=candidate['latitude'], lon=candidate['longitude'], source='ADSB')])
            with patch.object(collection_store, 'ROOT', root), patch('flight_comparison_ui.fetch_historical') as request:
                request.return_value = dict(records=records, provider='Mock FR24', incomplete=False)
                at = AppTest.from_file(str(Path(__file__).parent/'dashboard.py'), default_timeout=25).run()
                request.assert_not_called()
                at.button(key='flight_compare_api').click().run()
                self.assertFalse(at.exception)
                request.assert_called_once()
                self.assertTrue(any('POSSIBLE BROADCAST MATCH' in m.value for m in at.markdown))
                at.run()
                request.assert_called_once()
                request.side_effect = FlightDataError('Provider unavailable')
                at.button(key='flight_compare_api').click().run()
                self.assertFalse(at.exception)
                self.assertTrue(at.error)
                self.assertFalse(any('POSSIBLE BROADCAST MATCH' in m.value for m in at.markdown))
                self.assertFalse(any('NO MATCHING FLIGHT RECORD' in m.value for m in at.markdown))


if __name__ == '__main__':unittest.main()

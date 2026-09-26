from pathlib import Path
import unittest

class DashboardContractTests(unittest.TestCase):
    def test_satellite_awareness_is_the_active_workflow(self):
        text=(Path(__file__).parent/'dashboard.py').read_text(encoding='utf-8')
        self.assertIn("'Aircraft Awareness':aircraft_awareness", text)
        self.assertIn('parse_detections', text)
        self.assertIn('infer_geotiff', text)
        self.assertIn('Detect aircraft candidates in image', text)
        self.assertIn('NO OBSERVATIONS', text)
        self.assertNotIn("'Live Visual & UAV Monitor':", text)
        self.assertNotIn("'Thermal & RF Sensors':", text)

    def test_supervisor_starts_only_satellite_workflow(self):
        text=(Path(__file__).resolve().parents[1]/'start_dashboard.py').read_text(encoding='utf-8')
        self.assertIn("'multisatellite':[str(runtime),'multisatellite_worker.py']", text)
        self.assertNotIn("'rf_live':", text)
        self.assertNotIn("'realtime':", text)

if __name__ == '__main__': unittest.main()

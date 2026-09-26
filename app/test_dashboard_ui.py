"""Optional Streamlit AppTest smoke checks; requires dashboard dependencies."""
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
import collection_store


class DashboardTests(unittest.TestCase):
    def test_current_pages_render(self):
        logging.disable(logging.CRITICAL)
        pages=['Operations Center','Sentinal X Intelligence','Aircraft Awareness','Research & Validation','Setup & Auto Mode',
               'Satellite Detail','Processing Lab','Advanced Analysis','System Health']
        try:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d)/'app';root.mkdir();(root.parent/'AEROSENTINEL_BLUEPRINT.md').write_text('Blueprint')
                (root/'collection_config.json').write_text((Path(__file__).parent/'collection_config.json').read_text())
                with patch.object(collection_store,'ROOT',root):
                    at=AppTest.from_file(str(Path(__file__).parent/'dashboard.py'),default_timeout=25).run()
                    self.assertFalse(at.exception)
                    for page in pages:
                        at.sidebar.radio[0].set_value(page).run()
                        self.assertFalse(at.exception,msg=page)
        finally:
            logging.disable(logging.NOTSET)

if __name__=='__main__':unittest.main()

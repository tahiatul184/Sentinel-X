import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from terrain_service import fetch_terrain


class TerrainServiceTests(unittest.TestCase):
    def test_dem_context_computes_slope_and_writes_outputs(self):
        def searcher(bbox):
            return [{'id':'dem-1','assets':{'data':{'href':'https://example.blob.core.windows.net/dem.tif','type':'image/tiff'}}}]
        def fake_read(asset,grid):
            y,x=np.mgrid[0:grid['height'],0:grid['width']]
            arr=(100 + x*0.3 + y*0.1).astype('float32')
            return arr,np.ones(arr.shape,dtype=bool)
        with tempfile.TemporaryDirectory() as td:
            with patch('terrain_service._read_to_grid',side_effect=fake_read):
                result=fetch_terrain([90.39,23.79,90.41,23.81],td,grid_pixels=128,searcher=searcher)
            self.assertGreater(result['summary']['p90_slope_deg'],0)
            self.assertTrue(Path(result['raster_path']).exists())
            self.assertTrue(Path(result['preview_path']).exists())
            self.assertEqual(result['collection'],'cop-dem-glo-30')


if __name__=='__main__':unittest.main()

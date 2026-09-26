import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np

from aircraft_detector import _iou, infer_geotiff


class _Transform:
    def __mul__(self, point):
        return point[0] / 1000, point[1] / 1000


class _Raster:
    width = height = 512
    count = 3
    crs = 'EPSG:3857'
    transform = _Transform()

    def __enter__(self): return self
    def __exit__(self, *args): return False


class _Model:
    def __init__(self, boxes): self.boxes = boxes

    def predict(self, *_args, **_kwargs):
        boxes = [types.SimpleNamespace(cls=np.array(0), conf=np.array(score),
                 xyxy=np.array([coords])) for score, coords in self.boxes]
        return [types.SimpleNamespace(names={0:'aircraft'}, boxes=boxes)]


class DetectorTests(unittest.TestCase):
    def test_overlap_suppression(self):
        self.assertEqual(_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1)
        self.assertEqual(_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0)

    def test_inference_creates_georeferenced_rows(self):
        rasterio = types.ModuleType('rasterio')
        rasterio.open = lambda _: _Raster()
        windows = types.ModuleType('rasterio.windows')
        windows.Window = lambda *a: a
        warp = types.ModuleType('rasterio.warp')
        warp.transform = lambda _a, _b, xs, ys: (xs, ys)
        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / 'scene.tif'
            image.write_bytes(b'dummy')
            with patch.dict(sys.modules, {'rasterio':rasterio, 'rasterio.windows':windows, 'rasterio.warp':warp}), \
                 patch('direct_candidate._estimate_gsd', return_value=0.5), \
                 patch('aircraft_detector._rgb_tile', return_value=(np.zeros((512,512,3), dtype=np.uint8),1.0)):
                result = infer_geotiff(image, site_id='A', scene_id='S1',
                    acquired_at='2026-09-25T10:00:00Z',
                    model=_Model([(0.9, (20, 30, 80, 90))]))
                self.assertEqual(result['detections'], 1)
                self.assertEqual(result['awareness']['observations'][0]['presence'], 'CANDIDATE')
                self.assertAlmostEqual(result['rows'][0]['longitude'], 0.05)
                empty = infer_geotiff(image, site_id='A', scene_id='S2',
                    acquired_at='2026-09-26T10:00:00Z', model=_Model([]))
                self.assertEqual(empty['state'], 'NO DETECTION / ABSENCE UNDETERMINED')
                self.assertEqual(empty['awareness']['state'], 'NO OBSERVATIONS')

    def test_timezone_is_required_even_when_model_has_no_detection(self):
        with self.assertRaisesRegex(ValueError, 'timezone'):
            infer_geotiff('unused.tif', site_id='A', scene_id='S',
                          acquired_at='2026-09-25T10:00:00', model=_Model([]))


if __name__ == '__main__': unittest.main()

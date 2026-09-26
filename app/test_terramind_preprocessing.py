import tempfile
import unittest
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from terramind_job import _center_s2, _reproject_to_grid, _normalise


class TerraMindPreprocessingTests(unittest.TestCase):
    def _write(self,path,count,data_value,transform=None):
        transform=transform or from_origin(90.0,24.0,0.0001,0.0001)
        arr=np.full((count,240,240),data_value,dtype='float32')
        with rasterio.open(path,'w',driver='GTiff',height=240,width=240,count=count,dtype='float32',crs='EPSG:4326',transform=transform) as dst:
            dst.write(arr)

    def test_alignment_and_normalization(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            s2=td/'s2.tif';s1=td/'s1.tif';dem=td/'dem.tif'
            self._write(s2,12,2000.0)
            self._write(s1,2,-15.0)
            self._write(dem,1,8.0)
            s2_arr,crs,tr,bounds=_center_s2(s2,'scaled_10000')
            s1_arr=_reproject_to_grid(s1,2,crs,tr,(224,224),value_range=(-70,30),label='S1')
            dem_arr=_reproject_to_grid(dem,1,crs,tr,(224,224),value_range=(-600,9000),label='DEM')
            out=_normalise(s2_arr,s1_arr,dem_arr)
            self.assertEqual(out['S2L2A'].shape,(12,224,224))
            self.assertEqual(out['S1GRD'].shape,(2,224,224))
            self.assertEqual(out['DEM'].shape,(1,224,224))
            self.assertTrue(all(np.isfinite(v).all() for v in out.values()))


if __name__=='__main__':
    unittest.main()

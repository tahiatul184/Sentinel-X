from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin

from direct_candidate import analyze_highres, scan_inbox


class DirectCandidateTests(unittest.TestCase):
    def test_resolution_gated_candidate_screening(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'hi.tif'
            a=np.zeros((128,128),dtype='float32')+.2
            a[60:64,60:64]=1.0
            with rasterio.open(p,'w',driver='GTiff',width=128,height=128,count=1,dtype='float32',
                               crs='EPSG:32646',transform=from_origin(500000,2600000,.3,.3)) as dst:
                dst.write(a,1)
            result=analyze_highres(p,gsd_m=.3)
            self.assertTrue(result['feasibility']['feasible'])
            self.assertGreaterEqual(result['candidate_count'],1)
            self.assertEqual(result['identity'],'UNKNOWN / NOT CLASSIFIED')
            self.assertTrue(Path(result['segmentation_mask_path']).exists())

    def test_inbox_dedup(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);box=root/'highres_inbox';box.mkdir()
            p=box/'hi.tif';a=np.ones((32,32),dtype='float32')
            with rasterio.open(p,'w',driver='GTiff',width=32,height=32,count=1,dtype='float32',
                               crs='EPSG:32646',transform=from_origin(500000,2600000,.3,.3)) as dst:dst.write(a,1)
            state,changed=scan_inbox(root,{'geo_x_highres_inbox':'highres_inbox'})
            self.assertTrue(changed)
            state2,changed2=scan_inbox(root,{'geo_x_highres_inbox':'highres_inbox'},state)
            self.assertFalse(changed2)
            self.assertEqual(state['latest']['sha256'],state2['latest']['sha256'])

if __name__=='__main__':unittest.main()

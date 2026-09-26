import unittest
import numpy as np

from spectral_indices import compute_optical_indices, screening_layers
from image_processing import enhance_rgb, lee_filter, gradient_texture
from quality_control import image_quality_metrics
from change_detection import composite_change


class ProcessingModuleTests(unittest.TestCase):
    def test_spectral_index_suite(self):
        shape=(20,20); valid=np.ones(shape,bool)
        bands={
            'red':np.full(shape,.2,dtype='float32'),
            'green':np.full(shape,.25,dtype='float32'),
            'nir':np.full(shape,.4,dtype='float32'),
            'swir16':np.full(shape,.3,dtype='float32'),
            'swir22':np.full(shape,.1,dtype='float32'),
        }
        idx=compute_optical_indices(bands,valid)
        self.assertAlmostEqual(float(np.nanmean(idx['ndvi'])),1/3,places=5)
        self.assertIn('ndmi',idx);self.assertIn('mndwi',idx);self.assertIn('nbr2',idx)
        layers=screening_layers(idx)
        self.assertEqual(layers['water'].shape,shape)

    def test_rgb_enhancement_and_texture(self):
        y,x=np.mgrid[:64,:64]
        base=(x+y).astype('float32')
        rgb=enhance_rgb(base,base*.8,base*.6,np.ones((64,64),bool))
        self.assertEqual(rgb.shape,(64,64,3));self.assertEqual(rgb.dtype,np.uint8)
        texture=gradient_texture(base,np.ones((64,64),bool))
        self.assertTrue(np.isfinite(texture).all())

    def test_lee_filter_reduces_speckle_variance(self):
        rng=np.random.default_rng(7)
        source=np.ones((96,96),dtype='float32')
        noisy=source*rng.gamma(shape=1.4,scale=1/1.4,size=source.shape).astype('float32')
        filtered=lee_filter(np.log1p(noisy),np.ones(source.shape,bool),radius=2)
        self.assertLess(float(np.nanvar(filtered)),float(np.nanvar(np.log1p(noisy))))

    def test_quality_metrics_are_bounded(self):
        y,x=np.mgrid[:64,:64]
        rgb=np.stack([(x*4)%256,(y*4)%256,((x+y)*2)%256],axis=-1).astype('uint8')
        q=image_quality_metrics(rgb,np.ones((64,64),bool),cloud_cover_percent=10)
        self.assertGreaterEqual(q['processing_readiness_score'],0)
        self.assertLessEqual(q['processing_readiness_score'],1)
        self.assertGreater(q['entropy_bits'],1)

    def test_multisignal_change(self):
        shape=(32,32);valid=np.ones(shape,bool)
        previous={'valid':valid,'water':np.zeros(shape,'float32'),'ndvi':np.zeros(shape,'float32')}
        latest={'valid':valid,'water':np.full(shape,.7,'float32'),'ndvi':np.full(shape,.4,'float32')}
        change,summary=composite_change(latest,previous)
        self.assertEqual(summary['available_signal_count'],2)
        self.assertTrue(np.isfinite(change).all())
        self.assertGreater(summary['mean_composite_change_screening'],.5)

if __name__=='__main__':unittest.main()

class ThermalProcessingTests(unittest.TestCase):
    def test_uncertainty_weighted_thermal_diagnostics(self):
        from thermal_processing import thermal_diagnostics
        temp=np.array([[300.0,300.2],[300.1,310.0]],dtype='float32')
        unc=np.array([[0.5,0.5],[0.5,5.0]],dtype='float32')
        emis=np.full((2,2),.96,dtype='float32')
        cdist=np.array([[3.0,3.0],[3.0,.2]],dtype='float32')
        d=thermal_diagnostics(temp,np.ones((2,2),bool),uncertainty_k=unc,emissivity=emis,cloud_distance_km=cdist)
        self.assertLess(d['stats']['uncertainty_weighted_mean_surface_c'],28.0)
        self.assertGreater(d['stats']['st_qa_uncertainty']['max_k'],4.9)
        self.assertTrue(d['near_cloud_flag'][1,1])
        self.assertTrue(d['elevated_uncertainty_flag'][1,1])
        # Very hot but highly uncertain pixels are deliberately not promoted
        # to an anomaly candidate.
        self.assertFalse(d['thermal_anomaly_candidate'][1,1])
        d2=thermal_diagnostics(temp,np.ones((2,2),bool),uncertainty_k=np.full((2,2),.5,dtype='float32'))
        self.assertTrue(d2['thermal_anomaly_candidate'][1,1])

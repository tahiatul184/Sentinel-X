import copy
import json
from pathlib import Path
import unittest
from aircraft_awareness import analyze
from sentinal_validation import evaluate, representation_change

class SentinalTests(unittest.TestCase):
    def row(self, **changes):
        row=dict(site_id='A',scene_id='1',acquired_at='2026-01-01T00:00:00Z',modality='optical',latitude=23,longitude=90,confidence=.9,gsd_m=.5,object_length_m=20,registration_error_m=2,model_id='optical-v1',quality=1,cloud_fraction=0)
        row.update(changes);return row
    def test_context_cannot_rescue_weak_detector(self):
        result=analyze([self.row(confidence=.1),self.row(modality='thermal',confidence=1,model_id='thermal-v1')])
        self.assertEqual(result['state'],'INSUFFICIENT EVIDENCE')
    def test_native_resolution_overrides_enhanced_grid(self):
        self.assertEqual(analyze([self.row(native_gsd_m=10,enhancement='superresolved')])['state'],'INSUFFICIENT EVIDENCE')
    def test_duplicate_import_does_not_add_evidence(self):
        self.assertEqual(analyze([self.row(),self.row()])['observations'][0]['evidence_count'],1)
    def test_missing_registration_and_long_gap(self):
        nextrow=self.row(scene_id='2',acquired_at='2026-01-02T00:00:00Z',latitude=23.001,registration_error_m=None)
        result=analyze([self.row(),nextrow]);self.assertFalse(result['anomalies'])
        self.assertEqual(result['movement_candidates'][0]['state'],'REGISTRATION UNKNOWN')
        nextrow['acquired_at']='2026-02-01T00:00:00Z'
        self.assertFalse(analyze([self.row(),nextrow])['movement_candidates'])
    def document(self):
        return json.loads((Path(__file__).parent.parent/'research/evaluation_example.json').read_text())
    def test_metrics_include_missed_and_duplicate_objects(self):
        doc=self.document();p=copy.deepcopy(doc['methods']['optical_baseline'][0]);doc['methods']['optical_baseline'].append(p)
        r=evaluate(doc)['methods'];self.assertEqual(r['optical_baseline']['overall']['fp'],1)
        self.assertEqual(r['fusion_ablation']['overall']['fn'],1)
        self.assertEqual(r['fusion_ablation']['overall']['ap'],0)
    def test_split_leakage_and_incomplete_labels(self):
        doc=self.document();other=copy.deepcopy(doc['scenes'][0]);other.update(scene_id='train',split='train');doc['scenes'].append(other)
        with self.assertRaisesRegex(ValueError,'leakage'):evaluate(doc)
        doc=self.document();doc['scenes'][0]['fully_reviewed']=False
        with self.assertRaisesRegex(ValueError,'fully_reviewed'):evaluate(doc)
    def test_incompatible_feature_spaces_rejected(self):
        pair=json.loads((Path(__file__).parent.parent/'research/feature_pair_example.json').read_text())
        self.assertGreater(representation_change(pair['before'],pair['after'])['cosine_change'],0)
        pair['after']['checkpoint_sha256']='other'
        with self.assertRaisesRegex(ValueError,'checkpoint'):representation_change(pair['before'],pair['after'])

if __name__=='__main__':unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from imagery_archive import ensure_storage_layout, archive_multisatellite_run, backfill_multisatellite_archive
from collection_store import DEFAULTS


class ImageryArchiveTests(unittest.TestCase):
    def test_storage_layout_explains_highres_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); cfg=dict(DEFAULTS)
            layout=ensure_storage_layout(root,cfg)
            self.assertTrue(Path(layout['satellite_archive']).is_dir())
            self.assertTrue((Path(layout['highres_inbox'])/'README_ADD_HIGH_RES_IMAGES.txt').exists())
            self.assertFalse(layout['highres_auto_download'])

    def test_multisatellite_run_populates_user_visible_archive_and_log(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); cfg=dict(DEFAULTS)
            out=root/'data/multisatellite/runs/r1'; out.mkdir(parents=True)
            s=out/'analysis_stack.tif'; s.write_bytes(b'fake-tif')
            p=out/'preview.png'; p.write_bytes(b'fake-png')
            f=out/'multisatellite_fusion.tif'; f.write_bytes(b'fusion')
            fp=out/'multisatellite_fusion_preview.png'; fp.write_bytes(b'preview')
            run={'started_at':'2026-09-12T00:00:00+00:00','finished_at':'2026-09-12T00:01:00+00:00',
                 'scene_results':[{'satellite':'sentinel-2','scene_id':'scene-1','collection':'sentinel-2-l2a',
                                   'platform':'sentinel-2b','acquired_at':'2026-09-11T05:00:00+00:00',
                                   'native_gsd_m':10,'cloud_cover_percent':5,'valid_percent':98,
                                   'stack_path':str(s),'preview_path':str(p)}],
                 'fusion':{'fusion_path':str(f),'preview_path':str(fp),'available_satellites':['sentinel-2'],
                           'acquisition_skew_hours':0}}
            rows=archive_multisatellite_run(root,cfg,run)
            self.assertEqual(len(rows),2)
            archive=root/'data/imagery'
            self.assertTrue((archive/'image_log.jsonl').exists())
            self.assertTrue((archive/'latest_sentinel-2.json').exists())
            self.assertTrue((archive/'latest_fusion.json').exists())
            self.assertTrue(any(x.name=='analysis_stack.tif' for x in archive.rglob('analysis_stack.tif')))
            # Re-archiving the same run must not duplicate the log.
            before=(archive/'image_log.jsonl').read_text().splitlines()
            self.assertEqual(archive_multisatellite_run(root,cfg,run),[])
            after=(archive/'image_log.jsonl').read_text().splitlines()
            self.assertEqual(before,after)

    def test_backfill_indexes_existing_run_json(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); cfg=dict(DEFAULTS)
            run_dir=root/'data/multisatellite/runs/oldrun';run_dir.mkdir(parents=True)
            scene_dir=run_dir/'sentinel-1'/'old-scene';scene_dir.mkdir(parents=True)
            s=scene_dir/'analysis_stack.tif';s.write_bytes(b'tif')
            p=scene_dir/'preview.png';p.write_bytes(b'png')
            run={'started_at':'2026-09-10T00:00:00+00:00','finished_at':'2026-09-10T00:01:00+00:00',
                 'scene_results':[{'satellite':'sentinel-1','scene_id':'old-scene','collection':'sentinel-1-grd',
                                   'platform':'sentinel-1a','acquired_at':'2026-09-09T00:00:00+00:00',
                                   'native_gsd_m':10,'cloud_cover_percent':None,'valid_percent':95,
                                   'stack_path':'C:/old/TRUST_GEO/app/data/multisatellite/runs/oldrun/sentinel-1/old-scene/analysis_stack.tif',
                                   'preview_path':'C:/old/TRUST_GEO/app/data/multisatellite/runs/oldrun/sentinel-1/old-scene/preview.png'}],
                 'fusion':{}}
            (run_dir/'run.json').write_text(json.dumps(run),encoding='utf-8')
            status=backfill_multisatellite_archive(root,cfg)
            self.assertEqual(status['runs_checked'],1)
            self.assertEqual(status['entries_archived'],1)
            self.assertTrue((root/'data/imagery/image_log.jsonl').exists())

if __name__=='__main__': unittest.main()

"""Offline regression checks, including real loopback HTTP interruption/resume."""
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import requests
import satellite_downloader as mod
import resumable_transfer as transfer

TIFF = b'II\x2a\x00' + b'\x00' * (65536-4)
URL = 'https://sentinel-cogs.s3.us-west-2.amazonaws.com/test.tif'
ITEM = {'assets': {'visual': {'href': URL}}}

class Response:
    def __init__(self, chunks=None, *, status=200, total=len(TIFF), headers=None):
        self.chunks = [TIFF] if chunks is None else chunks
        self.status_code = status
        self.headers = {'Content-Length': str(total), 'ETag': '"version-one"'}
        self.headers.update(headers or {})
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self):
        if self.status_code >= 400: raise requests.HTTPError(str(self.status_code))
    def iter_content(self, **kwargs):
        for chunk in self.chunks:
            if isinstance(chunk, BaseException): raise chunk
            yield chunk

class Session:
    def __init__(self, *responses): self.responses=list(responses); self.calls=[]
    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.responses.pop(0)

class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder/'test.tif.download'
        self.cfg = {'min_free_gb':0, 'auto_download_connections':1}
        self.events=[]
        self.disk=patch.object(transfer.shutil, 'disk_usage', return_value=type('Disk', (), {'free':10*1024**3})())
        self.disk.start(); self.addCleanup(self.disk.stop)
    def run_transfer(self, session, **kwargs):
        return transfer.transfer_tiff(URL, self.path, self.cfg, session,
                                      lambda **x:self.events.append(x), lambda:False, **kwargs)
    def interrupt(self, offset=16384):
        with self.assertRaises(requests.ConnectionError):
            self.run_transfer(Session(Response([TIFF[:offset], requests.ConnectionError('dropped')])) )
        self.assertEqual(self.path.read_bytes(), TIFF[:offset])
    def resume_response(self, offset=16384, **kwargs):
        return Response([TIFF[offset:]], status=206, total=len(TIFF)-offset,
            headers={'Content-Range':f'bytes {offset}-{len(TIFF)-1}/{len(TIFF)}', **kwargs})
    def test_complete_transfer_and_progress(self):
        self.run_transfer(Session(Response()))
        self.assertEqual(self.path.read_bytes(), TIFF)
        self.assertEqual(self.events[-1]['downloaded_bytes'],len(TIFF))
        self.assertEqual(self.events[-1]['transfer_phase'],'crop')
    def test_resume_after_network_failure(self):
        self.interrupt()
        session=Session(self.resume_response())
        self.run_transfer(session)
        self.assertEqual(self.path.read_bytes(), TIFF)
        self.assertEqual(session.calls[0][1]['headers']['Range'],'bytes=16384-')
        self.assertEqual(session.calls[0][1]['headers']['If-Range'],'"version-one"')
        self.assertEqual(self.events[-1]['transfer_attempt'],2)
    def test_truncated_body_retains_progress(self):
        with self.assertRaisesRegex(RuntimeError,'ended'):
            self.run_transfer(Session(Response([TIFF[:1000]])))
        self.assertEqual(self.path.stat().st_size,1000)
    def test_server_ignores_range_replaces_instead_of_appending(self):
        self.interrupt()
        self.run_transfer(Session(Response()))
        self.assertEqual(self.path.read_bytes(),TIFF)
        self.assertTrue(any('declined resume' in e['message'] for e in self.events))
    def test_changed_etag_full_response(self):
        self.interrupt()
        changed=TIFF[:8]+b'X'*(len(TIFF)-8)
        self.run_transfer(Session(Response([changed],headers={'ETag':'"version-two"'})))
        self.assertEqual(self.path.read_bytes(),changed)
    def test_mismatched_range_never_appended(self):
        self.interrupt()
        with self.assertRaisesRegex(transfer.InvalidTransfer,'mismatched'):
            self.run_transfer(Session(self.resume_response(**{'Content-Range':f'bytes 99-{len(TIFF)-1}/{len(TIFF)}'})))
        self.assertFalse(self.path.exists())
    def test_changed_etag_partial_response_rejected(self):
        self.interrupt()
        with self.assertRaises(transfer.InvalidTransfer):
            self.run_transfer(Session(self.resume_response(ETag='"changed"')))
        self.assertFalse(self.path.exists())
    def test_http_error_keeps_previous_bytes(self):
        self.interrupt()
        with self.assertRaises(requests.HTTPError):
            self.run_transfer(Session(Response(status=503)))
        self.assertEqual(self.path.read_bytes(),TIFF[:16384])
    def test_pause_retains_bytes_and_next_call_resumes(self):
        stopped=False
        class PausingResponse(Response):
            def iter_content(self, **kwargs):
                nonlocal stopped
                yield TIFF[:16384]
                stopped=True
                yield TIFF[16384:]
        with self.assertRaises(transfer.TransferPaused):
            transfer.transfer_tiff(URL,self.path,self.cfg,Session(PausingResponse()),lambda **x:None,lambda:stopped)
        self.assertEqual(self.path.read_bytes(),TIFF[:16384])
        self.run_transfer(Session(self.resume_response()))
        self.assertEqual(self.path.read_bytes(),TIFF)
    def test_elapsed_time_over_15_minutes_does_not_abort(self):
        now=[0]
        def clock(): now[0]+=1000; return now[0]
        self.run_transfer(Session(Response([TIFF[:16384],TIFF[16384:]])),clock=clock)
        self.assertEqual(self.path.read_bytes(),TIFF)
        self.assertGreater(self.events[-1]['transfer_elapsed_seconds'],900)
    def test_non_tiff_removed(self):
        with self.assertRaisesRegex(transfer.InvalidTransfer,'non-TIFF'):
            self.run_transfer(Session(Response([b'<html>error</html>'])))
        self.assertFalse(self.path.exists())
    def test_size_limit(self):
        with self.assertRaisesRegex(transfer.InvalidTransfer,'limit'):
            self.run_transfer(Session(Response()),max_bytes=100)
        self.assertFalse(self.path.exists())
    def test_low_storage_keeps_previous_progress(self):
        self.interrupt()
        with patch.object(transfer.shutil,'disk_usage',return_value=type('Disk',(),{'free':1})()):
            with self.assertRaisesRegex(RuntimeError,'storage'):
                self.run_transfer(Session(self.resume_response()))
        self.assertEqual(self.path.read_bytes(),TIFF[:16384])
    def test_completed_cache_reused_without_http(self):
        self.run_transfer(Session(Response()))
        session=Session()
        self.run_transfer(session)
        self.assertEqual(session.calls,[])
    def test_url_change_does_not_reuse_bytes(self):
        self.interrupt()
        session=Session(Response())
        transfer.transfer_tiff(URL+'?different',self.path,self.cfg,session,lambda **x:None,lambda:False)
        self.assertNotIn('Range',session.calls[0][1]['headers'])
    def test_missing_etag_does_not_append(self):
        with self.assertRaises(RuntimeError):
            self.run_transfer(Session(Response([TIFF[:16384]],headers={'ETag':''})))
        session=Session(Response())
        self.run_transfer(session)
        self.assertNotIn('Range',session.calls[0][1]['headers'])
        self.assertEqual(self.path.read_bytes(),TIFF)
    def test_fallback_reuses_cache_and_skips_repeated_remote_failure(self):
        remote_calls=[]
        output=self.folder/'crop.tif.part'
        def crop(item,bbox,out,cfg,source,**kwargs):
            if kwargs['remote']:
                remote_calls.append(source)
                raise mod.RemoteRasterReadError('range read failed')
            self.assertEqual(source.read_bytes(),TIFF)
            out.write_bytes(b'crop')
            return {'asset_url':URL}
        with patch.object(mod,'crop_source',side_effect=crop):
            with self.assertRaisesRegex(RuntimeError,'dropped'):
                mod.download_crop(ITEM,[],output,self.cfg,session=Session(Response([TIFF[:16384],requests.ConnectionError('dropped')])))
            result=mod.download_crop(ITEM,[],output,self.cfg,session=Session(self.resume_response()))
        self.assertEqual(len(remote_calls),1)
        self.assertEqual(result['transfer_method'],'resumable_https_then_local_crop')
        self.assertEqual(list((self.folder/'satellite_transfer_cache').iterdir()),[])
    def test_downloader_restart_preserves_progress_and_registers_completed_crop(self):
        from collection_store import DEFAULTS
        from datetime import datetime, timezone
        now = datetime(2026,9,12,tzinfo=timezone.utc).timestamp()
        cfg = dict(DEFAULTS, auto_collect_enabled=True, auto_location_ready=True, auto_download_connections=1)
        item = dict(ITEM, id='test-scene', collection='sentinel-2-c1-l2a',
                    properties={'datetime':'2026-09-10T04:00:00Z','eo:cloud_cover':10})
        def crop(item,bbox,out,cfg,source,**kwargs):
            if kwargs['remote']:raise mod.RemoteRasterReadError('remote failed')
            self.assertEqual(source.read_bytes(),TIFF)
            out.write_bytes(b'completed-crop')
            return {'asset_url':URL}
        first = Session(Response([TIFF[:16384],requests.ConnectionError('dropped')]))
        first.headers = {}
        second = Session(self.resume_response()); second.headers = {}
        with patch.object(mod,'search_scenes',return_value=[item]), patch.object(mod,'crop_source',side_effect=crop), patch.object(mod.logging,'exception'):
            worker=mod.Downloader(self.folder,session=first,clock=lambda:now)
            worker.tick(cfg)
            status=worker.store.setting('satellite_download')
            self.assertEqual(status['state'],'ERROR')
            self.assertEqual(status['downloaded_bytes'],16384)
            self.assertEqual(status['next_check'],now+15)
            # A new downloader object has no memory from the first worker.
            restarted=mod.Downloader(self.folder,session=second,clock=lambda:now+30)
            restarted.tick(cfg)
            status=restarted.store.setting('satellite_download')
            self.assertEqual(status['state'],'WAITING')
            self.assertEqual(status['collected_last_check'],1)
            self.assertEqual(status['transfer_attempt'],2)
        self.assertEqual(second.calls[0][1]['headers']['Range'],'bytes=16384-')

    def test_fast_path_no_full_transfer(self):
        session=Session()
        with patch.object(mod,'crop_source',return_value={'ok':True}):
            self.assertEqual(mod.download_crop(ITEM,[],self.path,self.cfg,session=session),{'ok':True})
        self.assertEqual(session.calls,[])
    def test_validation_failure_no_transfer(self):
        session=Session()
        with patch.object(mod,'crop_source',side_effect=ValueError('no pixels')):
            with self.assertRaises(ValueError):mod.download_crop(ITEM,[],self.path,self.cfg,session=session)
        self.assertEqual(session.calls,[])
    def test_nested_errors_preserved(self):
        try:
            try:raise OSError('TLS detail')
            except OSError as exc:raise RuntimeError('Read failed') from exc
        except RuntimeError as exc:self.assertIn('TLS detail',mod.exception_detail(exc))
    def test_real_http_disconnect_and_resume(self):
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                headers=dict(self.headers); calls.append(headers)
                offset=int(self.headers.get('Range','bytes=0-').split('=')[1].split('-')[0])
                self.send_response(206 if offset else 200)
                self.send_header('Content-Length',str(len(TIFF)-offset))
                self.send_header('ETag','"version-one"')
                if offset:self.send_header('Content-Range',f'bytes {offset}-{len(TIFF)-1}/{len(TIFF)}')
                self.end_headers()
                self.wfile.write(TIFF[offset:] if offset else TIFF[:32768])
                self.wfile.flush()
                self.close_connection=True
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            url=f'http://127.0.0.1:{server.server_port}/image'
            with requests.Session() as session:
                session.trust_env=False
                with self.assertRaises(requests.RequestException):
                    transfer.transfer_tiff(url,self.path,self.cfg,session,lambda **x:None,lambda:False)
            self.assertEqual(self.path.stat().st_size,32768)
            # Fresh client reopens only the disk state left by the first attempt.
            with requests.Session() as session:
                session.trust_env=False
                transfer.transfer_tiff(url,self.path,self.cfg,session,lambda **x:None,lambda:False)
            self.assertEqual(self.path.read_bytes(),TIFF)
            self.assertEqual(calls[1]['Range'],'bytes=32768-')
            self.assertEqual(calls[1]['If-Range'],'"version-one"')
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)

if __name__=='__main__':unittest.main()

"""Concurrent and resumable transfer regressions using a real local HTTP server."""
import re
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import requests
import parallel_transfer as transfer
from resumable_transfer import InvalidTransfer, TransferPaused

PART=128*1024
DATA=b'II\x2a\x00'+bytes(range(256))*2047+b'\x01'*252
assert len(DATA)==4*PART

class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'image.download'
        self.events=[]
        self.state=dict(active=0,peak=0,requests=[],drop=False,dropped=False,
                        unsupported=False,decline_parts=False,bad_range=False,
                        etag='"test-v1"',data=DATA)
        self.lock=threading.Lock()
        state,lock=self.state,self.lock
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                request_range=self.headers.get('Range')
                data=state['data']
                is_probe=request_range=='bytes=0-0'
                with lock:state['requests'].append((request_range,self.headers.get('If-Range')))
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',request_range or '')
                supported=not state['unsupported'] and not (state['decline_parts'] and not is_probe)
                if match and supported:
                    first=int(match.group(1));last=int(match.group(2)) if match.group(2) else len(data)-1
                    status=206
                else:first,last,status=0,len(data)-1,200
                payload=data[first:last+1]
                self.send_response(status)
                self.send_header('Content-Length',str(len(payload)))
                self.send_header('ETag',state['etag'])
                if status==206:
                    actual_first=first+1 if state['bad_range'] and not is_probe else first
                    self.send_header('Content-Range',f'bytes {actual_first}-{last}/{len(data)}')
                self.end_headers()
                if is_probe and supported:
                    self.wfile.write(payload);return
                with lock:
                    state['active']+=1;state['peak']=max(state['peak'],state['active'])
                    drop=state['drop'] and not state['dropped'] and first==0
                    if drop:state['dropped']=True
                try:
                    if drop:payload=payload[:64*1024]
                    for i in range(0,len(payload),16384):
                        self.wfile.write(payload[i:i+16384]);self.wfile.flush()
                        time.sleep(0.005)
                except (BrokenPipeError,ConnectionResetError):pass
                finally:
                    self.close_connection=True
                    with lock:state['active']-=1
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self.shutdown)
        self.url=f'http://127.0.0.1:{self.server.server_port}/image'
        self.session=requests.Session();self.session.trust_env=False;self.addCleanup(self.session.close)
        self.disk=patch.object(transfer.shutil,'disk_usage',return_value=type('Disk',(),{'free':10*1024**3})())
        self.disk.start();self.addCleanup(self.disk.stop)
    def shutdown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
    def run_transfer(self,connections=4,cancelled=lambda:False,progress=None):
        transfer.transfer_tiff(self.url,self.path,{'min_free_gb':0,'auto_download_connections':connections},
            self.session,progress or (lambda **x:self.events.append(x)),cancelled,
            _part_bytes=PART,_min_parallel_bytes=1)
    def test_four_connections_fetch_correct_nonoverlapping_bytes(self):
        self.run_transfer()
        self.assertEqual(self.path.read_bytes(),DATA)
        self.assertGreaterEqual(self.state['peak'],2)
        self.assertLessEqual(self.state['peak'],4)
        ranges=[r for r,tag in self.state['requests'] if r!='bytes=0-0']
        self.assertEqual(set(ranges),{f'bytes={i*PART}-{(i+1)*PART-1}' for i in range(4)})
        self.assertEqual(self.events[-1]['downloaded_bytes'],len(DATA))
        self.assertEqual(self.events[-1]['download_connections'],4)
        self.assertFalse(transfer.parts_folder(self.path).exists())
    def test_interrupted_part_resumes_at_exact_saved_byte(self):
        self.state['drop']=True
        with self.assertRaises(requests.RequestException):self.run_transfer()
        first=transfer.parts_folder(self.path)/'00000.part'
        self.assertEqual(first.stat().st_size,64*1024)
        self.assertFalse(self.path.exists())
        self.state['requests'].clear()
        self.run_transfer()
        self.assertIn((f'bytes={64*1024}-{PART-1}','"test-v1"'),self.state['requests'])
        self.assertEqual(self.path.read_bytes(),DATA)
        self.assertEqual(self.events[-1]['transfer_attempt'],2)
    def test_changed_image_does_not_mix_old_parts(self):
        self.state['drop']=True
        with self.assertRaises(requests.RequestException):self.run_transfer()
        self.state['etag']='"test-v2"';self.state['data']=b'II\x2a\x00'+b'Z'*(len(DATA)-4)
        self.run_transfer()
        self.assertEqual(self.path.read_bytes(),self.state['data'])
    def test_probe_without_ranges_falls_back_to_one_connection(self):
        self.state['unsupported']=True
        self.run_transfer()
        self.assertEqual(self.path.read_bytes(),DATA)
        self.assertEqual(self.events[-1]['download_connections'],1)
    def test_server_declines_parts_after_probe_falls_back(self):
        self.state['decline_parts']=True
        self.run_transfer()
        self.assertEqual(self.path.read_bytes(),DATA)
        self.assertEqual(self.events[-1]['download_connections'],1)
    def test_invalid_range_not_published(self):
        self.state['bad_range']=True
        with self.assertRaises(InvalidTransfer):self.run_transfer()
        self.assertFalse(self.path.exists())
        self.assertFalse(transfer.parts_folder(self.path).exists())
        self.assertEqual(self.events[-1]['downloaded_bytes'],0)
    def test_single_connection_setting(self):
        self.run_transfer(connections=1)
        self.assertEqual(self.path.read_bytes(),DATA)
        self.assertEqual(self.state['peak'],1)
        self.assertEqual(self.events[-1]['download_connections'],1)
    def test_pause_preserves_parts(self):
        calls=[0]
        def cancelled():
            calls[0]+=1
            return calls[0]>=3
        with self.assertRaises(TransferPaused):self.run_transfer(cancelled=cancelled)
        self.assertTrue(transfer.parts_folder(self.path).exists())
        self.run_transfer()
        self.assertEqual(self.path.read_bytes(),DATA)
    def test_existing_sequential_partial_is_retained(self):
        from collection_store import atomic_json
        from resumable_transfer import metadata_path
        self.path.write_bytes(DATA[:65536])
        atomic_json(metadata_path(self.path),dict(url=self.url,total=len(DATA),etag='"test-v1"',attempts=1))
        self.run_transfer()
        self.assertEqual(self.path.read_bytes(),DATA)
        self.assertEqual(self.state['requests'][0],('bytes=65536-','"test-v1"'))
    def test_storage_guard_avoids_starting_parts(self):
        with patch.object(transfer.shutil,'disk_usage',return_value=type('Disk',(),{'free':1})()):
            with self.assertRaisesRegex(RuntimeError,'storage'):self.run_transfer()
        self.assertEqual(len(self.state['requests']),1)
    def test_connection_cap(self):
        with self.assertRaisesRegex(ValueError,'1, 2 or 4'):self.run_transfer(connections=8)
        self.assertFalse(self.state['requests'])

if __name__=='__main__':unittest.main()

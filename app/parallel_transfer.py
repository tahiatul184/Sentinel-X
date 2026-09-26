"""Bounded parallel byte-range transfers with persistent, separately saved parts."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time

import requests
from collection_store import atomic_json
from resumable_transfer import (transfer_tiff as single_transfer, strong_etag,
    discard_transfer as discard_single, metadata_path, TIFF_HEADERS,
    TransferPaused, InvalidTransfer)


class RangeUnsupported(RuntimeError):
    pass


def parts_folder(destination):
    return destination.with_name(destination.name + '.parts')


def has_transfer(destination):
    return destination.exists() or parts_folder(destination).exists()


def discard_transfer(destination):
    discard_single(destination)
    shutil.rmtree(parts_folder(destination), ignore_errors=True)


def probe_asset(url, session, max_bytes):
    with session.get(url, stream=True, timeout=(10, 30), allow_redirects=False,
                     headers={'Range':'bytes=0-0','Accept-Encoding':'identity'}) as response:
        response.raise_for_status()
        match = re.fullmatch(r'bytes 0-0/(\d+)', response.headers.get('Content-Range',''))
        tag = strong_etag(response.headers)
        if response.status_code != 206 or not match or not tag:
            return None
        total = int(match.group(1))
        if total < 8 or total > max_bytes:
            raise InvalidTransfer('Provider image exceeds the permitted TIFF size.')
        if response.headers.get('Content-Encoding','identity').lower() != 'identity':
            return None
        # Check the one-byte probe without reading an ignored full-file response.
        probe = iter(response.iter_content(chunk_size=2))
        if next(probe, b'') not in (b'I',b'M') or next(probe, b''):
            raise InvalidTransfer('Image server did not return a TIFF header.')
        return total, tag


def new_session(parent):
    """Separate Requests sessions for concurrent workers; preserve TLS/proxy settings."""
    session = requests.Session()
    session.headers.update(parent.headers)
    session.proxies.update(parent.proxies)
    session.verify, session.cert, session.trust_env = parent.verify, parent.cert, parent.trust_env
    return session


def transfer_tiff(url, destination, cfg, session, progress, cancelled,
                  *, max_bytes=1024**3, clock=time.monotonic,
                  _part_bytes=8*1024**2, _min_parallel_bytes=4*1024**2):
    destination = Path(destination)
    connections = int(cfg.get('auto_download_connections',4))
    if connections not in (1,2,4):
        raise ValueError('Download connections must be 1, 2 or 4.')
    folder = parts_folder(destination)

    def single(message=None):
        if message:progress(message=message, download_connections=1)
        def one_progress(**fields):progress(download_connections=1, **fields)
        single_transfer(url,destination,cfg,session,one_progress,cancelled,max_bytes=max_bytes,clock=clock)
        shutil.rmtree(folder,ignore_errors=True)

    # Continue an existing sequential file without discarding its saved bytes.
    if destination.exists() or (connections == 1 and not folder.exists()):
        return single()
    if cancelled():raise TransferPaused('Download paused; saved parts are retained.')
    info = probe_asset(url,session,max_bytes)
    if info is None:
        return single('This server does not support parallel ranges. Using one resumable connection.')
    total, tag = info
    if total < _min_parallel_bytes:
        return single()
    folder.mkdir(parents=True,exist_ok=True)
    manifest = folder/'manifest.json'
    try:
        meta = json.loads(manifest.read_text(encoding='utf-8'))
    except (OSError,ValueError):meta={}
    identity = dict(url=url,total=total,etag=tag,part_bytes=_part_bytes)
    if any(meta.get(k) != v for k,v in identity.items()):
        shutil.rmtree(folder)
        folder.mkdir()
        meta = dict(identity,attempts=0)
    if meta.get('single_only'):
        return single('Parallel ranges were declined for this image; using one connection.')
    meta['attempts'] = int(meta.get('attempts',0)) + 1
    atomic_json(manifest,meta)
    count = math.ceil(total/_part_bytes)
    paths=[folder/f'{i:05d}.part' for i in range(count)]
    lengths=[min(_part_bytes,total-i*_part_bytes) for i in range(count)]
    sizes=[]
    for path,length in zip(paths,lengths):
        size=path.stat().st_size if path.exists() else 0
        if size>length:path.unlink();size=0
        sizes.append(size)
    resumed=sum(sizes)
    reserve=float(cfg['min_free_gb'])*1024**3 + 800_000_000
    # The assembled file temporarily coexists with all downloaded parts.
    if shutil.disk_usage(folder).free < reserve + (total-resumed) + total:
        raise RuntimeError('Not enough free storage for parallel image parts and the completed TIFF.')
    lock=threading.Lock()
    stop=threading.Event()
    started=clock()
    sample_time,sample_bytes=started,resumed
    speed=0.0

    def report(message,phase='download'):
        nonlocal sample_time,sample_bytes,speed
        now=clock()
        with lock:received=sum(sizes)
        elapsed=now-sample_time
        if elapsed>0:speed=max(0.0,(received-sample_bytes)/elapsed)
        if elapsed>=10:sample_time,sample_bytes=now,received
        progress(message=message,downloaded_bytes=received,total_bytes=total,
                 speed_bps=speed,eta_seconds=(total-received)/speed if speed>0 else None,
                 transfer_attempt=meta['attempts'],resumed_bytes=resumed,
                 transfer_phase=phase,transfer_elapsed_seconds=now-started,
                 download_connections=min(connections,count))

    def fetch(index):
        path=paths[index]
        offset=sizes[index]
        first=index*_part_bytes+offset
        last=index*_part_bytes+lengths[index]-1
        if first>last:return
        if stop.is_set():raise TransferPaused('Download stopped; parts are saved.')
        with new_session(session) as client:
            with client.get(url,stream=True,timeout=(10,30),allow_redirects=False,
                            headers={'Range':f'bytes={first}-{last}', 'If-Range':tag,
                                     'Accept-Encoding':'identity'}) as response:
                if response.status_code==200:
                    raise RangeUnsupported('Server declined a parallel byte range.')
                if response.status_code in (412,416):
                    raise InvalidTransfer('Image changed or its byte range is no longer valid.')
                response.raise_for_status()
                expected=f'bytes {first}-{last}/{total}'
                response_tag=strong_etag(response.headers)
                if (response.status_code!=206 or response.headers.get('Content-Range')!=expected or
                        (response_tag is not None and response_tag!=tag) or
                        response.headers.get('Content-Encoding','identity').lower()!='identity'):
                    raise InvalidTransfer('Parallel image range or version did not match the request.')
                raw_length=response.headers.get('Content-Length')
                if raw_length is not None and raw_length!=str(last-first+1):
                    raise InvalidTransfer('Parallel response size did not match the requested range.')
                with path.open('ab',buffering=0) as out:
                    disk_check=offset
                    for chunk in response.iter_content(chunk_size=64*1024):
                        if stop.is_set():raise TransferPaused('Download stopped; parts are saved.')
                        if not chunk:continue
                        if offset+len(chunk)>lengths[index]:
                            raise InvalidTransfer('A parallel image response exceeded its range.')
                        out.write(chunk)
                        offset+=len(chunk)
                        with lock:sizes[index]=offset
                        if offset-disk_check>=1024**2:
                            if shutil.disk_usage(folder).free<reserve+total:
                                raise RuntimeError('Storage became too low. Downloaded parts are saved.')
                            disk_check=offset
                    os.fsync(out.fileno())
                if offset!=lengths[index]:
                    raise RuntimeError('An image connection ended early. Saved parts will resume.')

    message=f'Downloading with up to {connections} connections. Each part is saved for retries.'
    report(message)
    errors=[]
    pending=set()
    try:
        with ThreadPoolExecutor(max_workers=connections) as pool:
            pending={pool.submit(fetch,i) for i,size in enumerate(sizes) if size<lengths[i]}
            while pending:
                if cancelled():stop.set()
                done,pending=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in done:
                    try:future.result()
                    except Exception as exc:
                        errors.append(exc);stop.set()
                report(message if not stop.is_set() else 'Stopping connections; downloaded parts are saved.')
        invalid=next((e for e in errors if isinstance(e,InvalidTransfer)),None)
        if invalid:
            shutil.rmtree(folder)
            raise invalid
        unsupported=next((e for e in errors if isinstance(e,RangeUnsupported)),None)
        if unsupported:
            meta['single_only']=True
            atomic_json(manifest,meta)
            return single('The server declined parallel downloading; using one connection.')
        if cancelled():raise TransferPaused('Download paused; saved parts will resume.')
        if errors:
            raise next((e for e in errors if not isinstance(e,TransferPaused)),errors[0])
        report('All parts downloaded. Combining the satellite image locally.','assemble')
        assembled=folder/'assembled.tmp'
        try:
            with assembled.open('wb') as out:
                for path,length in zip(paths,lengths):
                    if cancelled():raise TransferPaused('Assembly paused; downloaded parts are saved.')
                    if path.stat().st_size!=length:raise InvalidTransfer('Saved image part size is incorrect.')
                    with path.open('rb') as source:shutil.copyfileobj(source,out,length=1024**2)
                out.flush();os.fsync(out.fileno())
            with assembled.open('rb') as source:
                if source.read(4) not in TIFF_HEADERS:raise InvalidTransfer('Assembled image is not a TIFF.')
            # Publish metadata first. On a crash before rename, all parts remain.
            atomic_json(metadata_path(destination),dict(url=url,total=total,etag=tag,attempts=meta['attempts']))
            os.replace(assembled,destination)
            shutil.rmtree(folder)
        finally:assembled.unlink(missing_ok=True)
        report('Download complete. Creating your area crop locally.','crop')
    except InvalidTransfer:
        stop.set()
        discard_transfer(destination)
        sizes[:] = [0]*len(sizes)
        report('Image validation failed. A fresh transfer is required.','error')
        raise
    except BaseException:
        stop.set()
        report('Transfer interrupted. Valid image parts are saved for the next attempt.','waiting')
        raise

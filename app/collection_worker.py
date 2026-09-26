"""Imagery inspection only: native GeoTIFF provenance, masks and previews."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import signal
import threading
import time
from collection_store import ROOT,Store,load_config,local_path,sha256
from imagery_archive import archive_inspected_scene, ensure_storage_layout


@contextmanager
def worker_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt;msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl;fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:raise RuntimeError('This AEROSENTINEL service is already running.') from exc
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def inspect_scene(scene,root=ROOT):
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from PIL import Image
    path=Path(scene['path'])
    with rasterio.open(path) as src:
        if not src.crs:raise ValueError('GeoTIFF has no coordinate reference system.')
        if src.count<3:raise ValueError('At least three bands are needed for the imagery preview.')
        height=max(1,round(src.height*min(1,768/src.width,768/src.height)))
        width=max(1,round(src.width*min(1,768/src.width,768/src.height)))
        data=src.read([1,2,3],out_shape=(3,height,width),masked=True,resampling=Resampling.bilinear).astype(float)
        valid=~np.any(np.ma.getmaskarray(data),axis=0)&np.isfinite(data.filled(np.nan)).all(axis=0)
        if not valid.any():raise ValueError('No valid sampled pixels.')
        rgb=np.zeros((height,width,3),dtype=np.uint8)
        for i in range(3):
            values=data[i].filled(0).astype(float)
            lo,hi=np.percentile(values[valid],[2,98]);hi=max(hi,lo+1)
            rgb[:,:,i]=np.where(valid,np.clip((values-lo)/(hi-lo)*255,0,255),0).astype('uint8')
        folder=root/'data/collection/previews';folder.mkdir(parents=True,exist_ok=True)
        preview=folder/(scene['id']+'.png');Image.fromarray(rgb).save(preview)
        bounds=transform_bounds(src.crs,'EPSG:4326',*src.bounds)
        tags=src.tags();meta=scene.get('input_metadata',{})
        return dict(preview=str(preview),width=src.width,height=src.height,bands=src.count,crs=str(src.crs),
            bounds=list(bounds),center={'lat':(bounds[1]+bounds[3])/2,'lon':(bounds[0]+bounds[2])/2},
            pixel_size=list(src.res),pixel_units=src.crs.linear_units if src.crs.is_projected else 'degrees',
            valid_percent_sampled=float(valid.mean()*100),cloud_cover=meta.get('cloud_cover_percent',tags.get('CLOUD_COVER')),
            acquired_at=meta.get('acquired_at',tags.get('ACQUISITION_TIME')),
            preview_note='Independent contrast stretches for viewing; not a quantitative change product.')


def main():
    store=Store(ROOT);stop=threading.Event();seen={}
    try: ensure_storage_layout(ROOT, load_config(ROOT))
    except Exception: pass
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.set())
    with worker_lock(ROOT/'data/collection/imagery_worker.lock'):
        while not stop.is_set():
            try:
                cfg=load_config(ROOT);folder=local_path(ROOT,cfg['watch_dir']);folder.mkdir(parents=True,exist_ok=True)
                store.setting('imagery_worker',dict(state='PAUSED' if store.setting('paused') else 'WATCHING',time=time.time()))
                if not store.setting('paused'):
                    for path in folder.iterdir():
                        if not path.is_file() or path.suffix.lower() not in ['.tif','.tiff']:continue
                        stat=path.stat();identity=(stat.st_size,stat.st_mtime_ns)
                        previous=seen.get(path)
                        if previous and previous[0]==identity and time.time()-previous[1]>=cfg['settle_seconds']:
                            meta_path=path.with_suffix(path.suffix+'.json')
                            metadata=json.loads(meta_path.read_text()) if meta_path.exists() else {}
                            store.register(path,sha256(path),metadata)
                            seen[path]=(identity,time.time()+86400)
                        elif not previous or previous[0]!=identity:seen[path]=(identity,time.time())
                    for scene in store.scenes():
                        if scene['state']=='QUEUED':
                            try:
                                quality=inspect_scene(scene)
                                store.update(scene['id'],'IMAGERY_READY','GeoTIFF inspected and preview created.',quality=quality)
                                try:
                                    archive_inspected_scene(ROOT, cfg, scene, quality)
                                except Exception as archive_exc:
                                    store.update(scene['id'],'IMAGERY_READY','Preview created; imagery archive/log update failed.',
                                                 quality=quality,archive_error=f'{type(archive_exc).__name__}: {archive_exc}')
                            except Exception as exc:store.update(scene['id'],'FAILED',str(exc),error=str(exc))
            except Exception as exc:store.setting('imagery_worker',dict(state='ERROR',time=time.time(),error=str(exc)))
            stop.wait(3)

if __name__=='__main__':main()

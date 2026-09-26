"""Optional, isolated Prithvi feature extraction for declared HLS reflectance."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import numpy as np
from collection_store import atomic_json,sha256
from trust_core import timestamp

MODEL='prithvi_eo_v2_tiny_tl'
MODEL_CARD='https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL'
MEAN=np.array([1087,1342,1433,2734,1958,1363],dtype=np.float32)
STD=np.array([2248,2179,2178,1850,1242,1049],dtype=np.float32)
BANDS=['blue','green','red','narrow_nir','swir1','swir2']


def normalize(data,units):
    data=np.asarray(data,dtype=np.float32)
    if data.shape!=(6,224,224):raise ValueError('Prithvi input must be six bands on a 224 × 224 native grid.')
    if not np.isfinite(data).all():raise ValueError('The input contains missing or non-finite reflectance pixels.')
    if units=='reflectance':data=data*10000
    elif units!='scaled_10000':raise ValueError('Unknown reflectance units.')
    if data.min() < -2000 or data.max()>16000:raise ValueError('Reflectance range does not match the declared units.')
    return (data-MEAN[:,None,None])/STD[:,None,None]


def ready():return all(importlib.util.find_spec(m) is not None for m in ['torch','terratorch'])


def extract(input_path,output,units,acquired):
    import rasterio
    import torch
    from rasterio.windows import Window
    from rasterio.warp import transform
    from terratorch.registry import BACKBONE_REGISTRY
    from importlib.metadata import version
    acquired=timestamp(acquired)
    with rasterio.open(input_path) as src:
        if src.count!=6 or src.width<224 or src.height<224:
            raise ValueError('Supply a six-band HLS crop of at least 224 × 224 pixels. RGB downloads are not accepted.')
        if not src.crs or not src.crs.is_projected or src.crs.linear_units not in ['metre','meter'] or any(abs(v-30)>1 for v in src.res):
            raise ValueError('This integration requires an HLS grid in metres at approximately 30 m/pixel.')
        window=Window((src.width-224)//2,(src.height-224)//2,224,224)
        data=src.read(window=window,masked=True)
        if np.ma.getmaskarray(data).any():raise ValueError('The center patch contains nodata. Supply a valid HLS patch with QA reviewed.')
        x=normalize(data,units)
        cx,cy=src.xy(window.row_off+112,window.col_off+112)
        lon,lat=transform(src.crs,'EPSG:4326',[cx],[cy])
        crs=str(src.crs)
    torch.set_num_threads(2)
    model=BACKBONE_REGISTRY.build(MODEL,pretrained=True,num_frames=1)
    model.eval()
    weights_hash=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        weights_hash.update(name.encode())
        weights_hash.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    with torch.inference_mode():
        features=model.forward_features(torch.from_numpy(x)[None,:,None],
            temporal_coords=torch.tensor([[[acquired.year,acquired.timetuple().tm_yday]]],dtype=torch.float32),
            location_coords=torch.tensor([[lat[0],lon[0]]],dtype=torch.float32))
        value=features[-1] if isinstance(features,(list,tuple)) else features
        if value.ndim==3:vector=value[:,1:,:].mean(dim=1)
        elif value.ndim==4:vector=value.flatten(2).mean(dim=2)
        else:raise RuntimeError(f'Unexpected Prithvi feature shape: {tuple(value.shape)}')
        vector=vector[0].detach().cpu().numpy()
        if not np.isfinite(vector).all():raise RuntimeError('Model returned non-finite features.')
    result=dict(model=MODEL,model_weights_sha256=weights_hash.hexdigest(),model_card=MODEL_CARD,origin='Model-derived',input_sha256=sha256(Path(input_path)),
        acquired_at=acquired.isoformat(),bands=BANDS,units=units,center_latitude=lat[0],center_longitude=lon[0],
        crs=crs,patch=[224,224],embedding=vector.tolist(),dimension=len(vector),
        runtime={'torch':version('torch'),'terratorch':version('terratorch')},
        interpretation='Feature vector only: no flood, damage or operational-readiness classifier is attached.',
        limitations='Single-date HLS patch; QA is operator-declared. No local task calibration or field validation.')
    atomic_json(Path(output),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('request');args=parser.parse_args()
    request=json.loads(Path(args.request).read_text())
    status=Path(request['status'])
    try:
        atomic_json(status,dict(state='RUNNING',pid=os.getpid(),time=time.time(),message='Loading Prithvi; first run downloads model weights.'))
        extract(request['input'],request['output'],request['units'],request['acquired'])
        atomic_json(status,dict(state='COMPLETE',time=time.time(),output=request['output']))
    except Exception as exc:
        atomic_json(status,dict(state='FAILED',time=time.time(),error=f'{type(exc).__name__}: {exc}'))
        raise

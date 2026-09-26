"""Optional isolated model environment, launched explicitly from Foundation lab."""
from pathlib import Path
import os
import subprocess
import sys
import time
import venv
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'app'))
from collection_store import atomic_json
from collection_worker import worker_lock
status=ROOT/'app/data/foundation/setup-status.json'
try:
    with worker_lock(ROOT/'app/data/foundation/setup.lock'):
        atomic_json(status,dict(state='RUNNING',pid=os.getpid(),time=time.time(),message='Installing optional foundation environment.'))
        folder=ROOT/'.foundation-venv';python=folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        if not python.exists():venv.EnvBuilder(with_pip=True).create(folder)
        subprocess.run([str(python),'-m','pip','install','torch>=2.6,<3','torchvision','--index-url','https://download.pytorch.org/whl/cpu'],check=True)
        subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'app/requirements-foundation.txt')],check=True)
        subprocess.run([str(python),'-c','import torch,terratorch,rasterio; from terratorch.registry import BACKBONE_REGISTRY'],check=True)
        atomic_json(folder/'ready.json',dict(installed_at=time.time()))
        atomic_json(status,dict(state='COMPLETE',time=time.time(),message='Environment installed. Weights download on first feature extraction.'))
except Exception as exc:
    atomic_json(status,dict(state='FAILED',time=time.time(),error=str(exc)));raise

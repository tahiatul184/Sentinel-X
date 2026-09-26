"""Install and supervise AEROSENTINEL's local imagery and dashboard processes."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import venv
import webbrowser
from logging.handlers import RotatingFileHandler
import logging

PROJECT=Path(__file__).resolve().parent;APP=PROJECT/'app'
sys.path.insert(0,str(APP))
from collection_store import Store,atomic_json,load_config
from imagery_archive import ensure_storage_layout, backfill_multisatellite_archive
from collection_worker import worker_lock


def bootstrap(setup_only=False,no_browser=False):
    if sys.version_info<(3,12):raise RuntimeError('Python 3.12 or newer is required.')
    folder=PROJECT/'.dashboard-venv';python=folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():venv.EnvBuilder(with_pip=True).create(folder)
    requirements=APP/'requirements-dashboard.txt';digest=hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker=folder/'setup.json'
    previous=json.loads(marker.read_text()) if marker.exists() else {}
    probe=subprocess.run([str(python),'-c','import streamlit,rasterio,numpy,pandas,PIL,requests,cv2,ultralytics,huggingface_hub'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if previous.get('digest')!=digest or probe.returncode:
        print('Installing AEROSENTINEL imagery and dashboard packages...',flush=True)
        subprocess.run([str(python),'-m','pip','install','-r',str(requirements)],check=True)
        subprocess.run([str(python),'-c','import streamlit,rasterio,numpy,pandas,PIL,requests,cv2,ultralytics,huggingface_hub'],check=True)
        atomic_json(marker,dict(digest=digest))
    if setup_only:return 0
    command=[str(python),str(Path(__file__)),'--no-bootstrap']
    if no_browser:command.append('--no-browser')
    return subprocess.call(command)


def healthy():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=1) as response:return response.status==200
    except Exception:return False


def stop_process(process):
    if process.poll() is not None:return
    if os.name=='nt':
        subprocess.run([str(Path(os.environ['SystemRoot'])/'System32/taskkill.exe'),'/PID',str(process.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW)
    else:os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:process.kill()


def log_stream(name,stream):
    logger=logging.getLogger(name);logger.setLevel(logging.INFO)
    handler=RotatingFileHandler(APP/f'data/collection/logs/{name}.log',maxBytes=3_000_000,backupCount=2,encoding='utf-8');logger.addHandler(handler)
    try:
        for line in stream:logger.info(line.rstrip())
    finally:stream.close();handler.close();logger.removeHandler(handler)


def run(no_browser=False,stop_event=None,install_signals=True):
    store=Store(APP);stopped=stop_event or threading.Event();processes={}
    try:
        cfg=load_config(APP);ensure_storage_layout(APP,cfg);store.setting('storage_backfill',backfill_multisatellite_archive(APP,cfg))
    except Exception as exc:
        store.setting('storage_backfill',{'error':f'{type(exc).__name__}: {exc}'})
    (APP/'data/collection/logs').mkdir(parents=True,exist_ok=True)
    with worker_lock(APP/'data/collection/supervisor.lock'):
        with socket.socket() as s:
            try:s.bind(('127.0.0.1',8501))
            except OSError as exc:raise RuntimeError('Port 8501 is in use. Stop the earlier app before launching AEROSENTINEL.') from exc
        runtime=Path(sys.executable)
        if runtime.name.lower()=='pythonw.exe':runtime=runtime.with_name('python.exe')
        specs={'imagery':[str(runtime),'collection_worker.py'],'multisatellite':[str(runtime),'multisatellite_worker.py'],
            'highres':[str(runtime),'highres_worker.py'],
            'operations':[str(runtime),'ops_worker.py'],
            'dashboard':[str(runtime),'-m','streamlit','run','dashboard.py','--server.address','127.0.0.1','--server.port','8501',
                         '--server.headless','true','--global.developmentMode','false','--server.maxUploadSize','300','--browser.gatherUsageStats','false']}
        env=dict(os.environ,PYTHONUNBUFFERED='1',STREAMLIT_BROWSER_GATHER_USAGE_STATS='false')
        history={name:deque() for name in specs}
        def launch(name):
            p=subprocess.Popen(specs[name],cwd=APP,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0,start_new_session=os.name!='nt')
            processes[name]=p;threading.Thread(target=log_stream,args=(name,p.stdout),daemon=True).start()
        if install_signals:
            for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stopped.set())
        try:
            for name in specs:launch(name)
            opened=False;started=time.monotonic()
            while not stopped.wait(2):
                states={}
                for name,p in list(processes.items()):
                    if p.poll() is not None:
                        now=time.monotonic()
                        while history[name] and now-history[name][0]>300:history[name].popleft()
                        if len(history[name])<3:history[name].append(now);launch(name);states[name]='RESTARTING'
                        else:states[name]='FAILED'
                    else:states[name]='RUNNING'
                ready=healthy() and states['dashboard']=='RUNNING'
                store.setting('supervisor',dict(time=time.time(),services=states,dashboard_ready=ready,pid=os.getpid()))
                if ready and not opened:
                    if not no_browser:webbrowser.open('http://127.0.0.1:8501')
                    opened=True
                if not opened and time.monotonic()-started>120:raise RuntimeError('Dashboard did not become ready. Open app/data/collection/logs/dashboard.log.')
        finally:
            for p in processes.values():stop_process(p)
            store.setting('supervisor',dict(time=time.time(),state='STOPPED'))
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--setup-only',action='store_true');parser.add_argument('--no-bootstrap',action='store_true');parser.add_argument('--no-browser',action='store_true');args=parser.parse_args()
    try:raise SystemExit(run(args.no_browser) if args.no_bootstrap else bootstrap(args.setup_only,args.no_browser))
    except Exception as exc:print(f'AEROSENTINEL: {exc}',file=sys.stderr);raise SystemExit(1)

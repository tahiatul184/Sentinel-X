"""Local deployment-readiness checks for AEROSENTINEL v2.8.0."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time

from collection_store import Store, load_config

PLANETARY_STAC = 'https://planetarycomputer.microsoft.com/api/stac/v1/'
OPEN_METEO = 'https://api.open-meteo.com/v1/forecast'


def _age_minutes(epoch, now):
    try:
        return max(0.0, (float(now)-float(epoch))/60.0)
    except (TypeError, ValueError):
        return None


def local_readiness(root, now=None):
    root=Path(root);now=time.time() if now is None else float(now)
    cfg=load_config(root);store=Store(root)
    supervisor=store.setting('supervisor') or {}
    ops=store.setting('ops_status') or {}
    sat=store.setting('multisatellite_download') or {}
    report=store.setting('ops_latest_report') or {}
    weather=store.setting('weather_latest') or {}
    geo_x=store.setting('aerosentinel_latest') or store.setting('geo_x_latest') or {}
    disk=shutil.disk_usage(root)
    services=supervisor.get('services') or {}
    required={'imagery','multisatellite','highres','operations','dashboard'}
    running=sum(1 for k in required if services.get(k)=='RUNNING')
    report_age=_age_minutes(report.get('epoch'),now)
    weather_age=_age_minutes(weather.get('_fetched_epoch'),now)
    report_limit=max(3,int(cfg.get('ops_report_interval_minutes',5))+3)
    checks=[
        {'check':'Device/AOI location saved','ok':bool(cfg.get('auto_location_ready')),'detail':cfg.get('auto_area_name')},
        {'check':'Automation enabled','ok':bool(cfg.get('auto_collect_enabled') and cfg.get('ops_automation_enabled') and not store.setting('paused')),'detail':'running' if not store.setting('paused') else 'paused'},
        {'check':'Background services','ok':running==len(required),'detail':f'{running}/{len(required)} running'},
        {'check':'Free disk space','ok':disk.free>=2*1024**3,'detail':f'{disk.free/1024**3:.1f} GB free'},
        {'check':'5-minute report exporter','ok':report_age is not None and report_age<=report_limit,'detail':'no report yet' if report_age is None else f'latest {report_age:.1f} min ago'},
        {'check':'Weather context','ok':weather_age is not None and weather_age<=20,'detail':'not fetched yet' if weather_age is None else f'latest {weather_age:.1f} min ago'},
        {'check':'Satellite worker','ok':sat.get('state') not in {'ERROR','FAILED'},'detail':str(sat.get('state') or 'starting')},
        {'check':'Operations worker','ok':ops.get('state') not in {'ERROR','FAILED'},'detail':str(ops.get('state') or 'starting')},
        {'check':'AEROSENTINEL research layer','ok':(not cfg.get('geo_x_enabled',True)) or bool(geo_x),
         'detail':'disabled' if not cfg.get('geo_x_enabled',True) else str((geo_x.get('operational_aviation_risk') or {}).get('status') or (geo_x.get('predictive_early_warning') or {}).get('status') or 'waiting for first assessment')},
        {'check':'AEROSENTINEL deployment metrics','ok':(not cfg.get('geo_x_enabled',True)) or not geo_x or bool(geo_x.get('deployment_readiness')),
         'detail':'waiting for first assessment' if not geo_x else ('available' if geo_x.get('deployment_readiness') else 'missing')},

    ]
    passed=sum(1 for c in checks if c['ok'])
    state='READY' if passed==len(checks) else ('DEGRADED' if passed>=len(checks)-2 else 'NOT_READY')
    return {'state':state,'passed':passed,'total':len(checks),'checks':checks,'time':datetime.fromtimestamp(now,timezone.utc).isoformat()}


def live_provider_probe(root, timeout=10, get=None):
    cfg=load_config(Path(root))
    if get is None:
        import requests
        get=requests.get
    results=[]
    try:
        r=get(PLANETARY_STAC,timeout=timeout);r.raise_for_status()
        results.append({'provider':'Planetary Computer STAC','ok':True,'detail':f'HTTP {r.status_code}'})
    except Exception as exc:
        results.append({'provider':'Planetary Computer STAC','ok':False,'detail':f'{type(exc).__name__}: {exc}'})
    if cfg.get('auto_location_ready'):
        try:
            r=get(OPEN_METEO,params={'latitude':float(cfg['auto_latitude']),'longitude':float(cfg['auto_longitude']),
                    'current':'temperature_2m','forecast_days':1,'timezone':'UTC'},timeout=timeout)
            r.raise_for_status()
            results.append({'provider':'Open-Meteo','ok':True,'detail':f'HTTP {r.status_code}'})
        except Exception as exc:
            results.append({'provider':'Open-Meteo','ok':False,'detail':f'{type(exc).__name__}: {exc}'})
    else:
        results.append({'provider':'Open-Meteo','ok':False,'detail':'save/allow device location first'})
    return {'ok':all(x['ok'] for x in results),'results':results}

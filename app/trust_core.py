"""AEROSENTINEL: transparent scenario calculations and persistent research evidence."""
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
import math
import uuid
import numpy as np
from collection_store import Store

VERSION='2.8.0'


def utc_now():return datetime.now(timezone.utc).isoformat()


def number(value,low,high,name):
    value=float(value)
    if not math.isfinite(value) or not low<=value<=high:raise ValueError(f'{name} must be between {low} and {high}.')
    return value


def timestamp(value):
    result=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('Timestamps must include a timezone, for example 2026-09-12T06:00:00Z.')
    return result.astimezone(timezone.utc)


class ResearchStore:
    def __init__(self,root):
        self.store=Store(root)
        with self.store.db() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS research_records(
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, created TEXT NOT NULL,
                title TEXT NOT NULL, origin TEXT NOT NULL, payload TEXT NOT NULL,
                digest TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS research_reviews(
                id TEXT PRIMARY KEY, record_id TEXT NOT NULL, created TEXT NOT NULL,
                reviewer TEXT NOT NULL, decision TEXT NOT NULL, note TEXT NOT NULL);''')
    def add(self,kind,title,origin,payload):
        if origin not in ['Observed','Imported','Simulated','Model-derived']:raise ValueError('Unknown evidence origin.')
        if not str(title).strip():raise ValueError('Give the record a title.')
        raw=json.dumps(payload,sort_keys=True,allow_nan=False)
        ident=uuid.uuid4().hex[:16]
        with self.store.db() as db:
            db.execute('INSERT INTO research_records VALUES(?,?,?,?,?,?,?)',
                (ident,kind,utc_now(),title.strip(),origin,raw,hashlib.sha256(raw.encode()).hexdigest()))
        return ident
    def records(self,kind=None):
        with self.store.db() as db:
            rows=db.execute('SELECT * FROM research_records '+('WHERE kind=? ' if kind else '')+'ORDER BY created DESC',
                (kind,) if kind else ()).fetchall()
        return [dict(dict(r),payload=json.loads(r['payload'])) for r in rows]
    def review(self,ident,reviewer,decision,note):
        if not reviewer.strip():raise ValueError('A reviewer name is required.')
        if decision not in ['Needs follow-up','Reviewed','Rejected']:raise ValueError('Invalid review decision.')
        with self.store.db() as db:
            if not db.execute('SELECT id FROM research_records WHERE id=?',(ident,)).fetchone():raise ValueError('Record not found.')
            db.execute('INSERT INTO research_reviews VALUES(?,?,?,?,?,?)',
                       (uuid.uuid4().hex,ident,utc_now(),reviewer.strip(),decision,note))
    def export(self):
        with self.store.db() as db:reviews=[dict(r) for r in db.execute('SELECT * FROM research_reviews ORDER BY created')]
        return dict(framework='AEROSENTINEL',version=VERSION,exported_at=utc_now(),records=self.records(),reviews=reviews,
                    integrity_note='SHA-256 identifies record content; local records are not tamper-proof or authenticated.')


def parse_sensors(text):
    rows=list(csv.DictReader(io.StringIO(text)))
    if not rows or len(rows)>10000:raise ValueError('Supply 1–10,000 readings.')
    needed={'timestamp','site','rain_mm_h','water_level_m','source'}
    if not needed.issubset(rows[0]):raise ValueError('CSV needs: timestamp, site, rain_mm_h, water_level_m, source.')
    result=[]
    for row in rows:
        if not row['site'].strip() or not row['source'].strip():raise ValueError('Every reading needs a site and source.')
        result.append(dict(timestamp=timestamp(row['timestamp']).isoformat(),site=row['site'].strip(),
            rain_mm_h=number(row['rain_mm_h'],0,500,'Rainfall'),
            water_level_m=number(row['water_level_m'],-100,100,'Water level'),source=row['source'].strip()))
    keys=[(r['site'],r['timestamp']) for r in result]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate site/time readings must be resolved before import.')
    return sorted(result,key=lambda r:r['timestamp'])


def validate_asset(asset):
    if not asset.get('name','').strip():raise ValueError('Asset name is required.')
    if not str(asset.get('site','')).strip():raise ValueError('Site identifier is required.')
    return dict(name=asset['name'].strip(),type=asset.get('type','Other'),
        latitude=number(asset['latitude'],-90,90,'Latitude'),longitude=number(asset['longitude'],-180,180,'Longitude'),
        criticality=number(asset['criticality'],1,5,'Criticality'),
        threshold_mm=number(asset['threshold_mm'],1,1000,'Scenario threshold'),
        site=str(asset.get('site','')).strip())


def water_balance(rainfall,area_ha,runoff,infiltration_mm_h,drain_m3_s,initial_mm=0,step_minutes=15):
    """Lumped catchment mass balance, not a spatial flood or runway-safety model."""
    rain=np.asarray(rainfall,dtype=float)
    if rain.ndim!=1 or not 1<=len(rain)<=384 or not np.isfinite(rain).all() or (rain<0).any() or (rain>500).any():
        raise ValueError('Rainfall must contain 1–384 finite hourly intensities between 0 and 500 mm/h.')
    area=number(area_ha,0.01,10000,'Catchment area')*10000
    coeff=number(runoff,0,1,'Runoff coefficient')
    infiltration=number(infiltration_mm_h,0,100,'Infiltration')
    drain=number(drain_m3_s,0,10000,'Drain capacity')
    depth=number(initial_mm,0,1000,'Initial surface storage')
    dt=number(step_minutes,1,60,'Step minutes')/60
    volume=depth*area/1000
    result=[]
    for i,r in enumerate(rain):
        inflow=r*dt*area/1000*coeff
        available=volume+inflow
        infiltrated=min(available,infiltration*dt*area/1000)
        drained=min(available-infiltrated,drain*dt*3600)
        volume=available-infiltrated-drained
        result.append(dict(hour=(i+1)*dt,rain_mm_h=float(r),surface_storage_mm=volume/area*1000,
                           inflow_m3=float(inflow),infiltrated_m3=float(infiltrated),drained_m3=float(drained),stored_m3=float(volume)))
    return result


def simulate(parameters,rainfall=None,seed=2026):
    hours=int(number(parameters['hours'],1,48,'Duration'))
    rain=np.asarray(rainfall if rainfall is not None else [parameters['rain_mm_h']]*(hours*4),dtype=float)
    kwargs={k:parameters[k] for k in ['area_ha','runoff','infiltration_mm_h','drain_m3_s','initial_mm']}
    center=water_balance(rain,**kwargs)
    uncertainty=number(parameters.get('uncertainty_percent',20),0,80,'Sensitivity range')/100
    rng=np.random.default_rng(seed)
    paths=[]
    for _ in range(128):
        factor=rng.uniform(1-uncertainty,1+uncertainty,3)
        varied=dict(kwargs,runoff=min(1,kwargs['runoff']*factor[1]),drain_m3_s=kwargs['drain_m3_s']*factor[2])
        paths.append([v['surface_storage_mm'] for v in water_balance(np.clip(rain*factor[0],0,500),**varied)])
    bands=np.percentile(paths,[10,50,90],axis=0)
    for i,row in enumerate(center):
        row.update(p10_mm=float(bands[0,i]),p50_mm=float(bands[1,i]),p90_mm=float(bands[2,i]))
    return dict(parameters=parameters,series=center,peak_mm=max(r['surface_storage_mm'] for r in center),
        model='Lumped rainfall-runoff-storage balance v1',origin='Simulated',seed=seed,
        assumptions='Constant catchment parameters; no terrain routing, river backwater, tides or drainage-network hydraulics.',
        uncertainty='P10/P50/P90 summarize uniformly perturbed scenarios, not calibrated forecast probabilities.')


def evaluation(rows,threshold=0.5):
    threshold=number(threshold,0,1,'Classification threshold')
    if not rows or len(rows)>100000:raise ValueError('Provide 1–100,000 labelled examples.')
    y=np.array([number(r['label'],0,1,'Binary label') for r in rows])
    if not np.isin(y,[0,1]).all():raise ValueError('Labels must be exactly 0 or 1.')
    p=np.array([number(r['probability'],0,1,'Probability') for r in rows])
    pred=p>=threshold
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
    div=lambda a,b:a/b if b else None
    bins=[];ece=0
    for i in range(10):
        mask=(p>=i/10)&(p<(i+1)/10) if i<9 else (p>=.9)&(p<=1)
        if mask.any():
            confidence=float(p[mask].mean());frequency=float(y[mask].mean())
            ece+=abs(confidence-frequency)*int(mask.sum())/len(y)
            bins.append(dict(bin=i,count=int(mask.sum()),mean_probability=confidence,event_frequency=frequency))
    return dict(n=len(y),threshold=threshold,tp=tp,fp=fp,fn=fn,tn=tn,
        precision=div(tp,tp+fp),recall=div(tp,tp+fn),iou=div(tp,tp+fp+fn),f1=div(2*tp,2*tp+fp+fn),
        accuracy=(tp+tn)/len(y),brier=float(np.mean((p-y)**2)),ece=float(ece),calibration=bins,
        scope='Metrics assess supplied labels and predictions; they do not establish deployment readiness.')


def recommendation_checks(scenes,sensors,assets,model_ready=False):
    checks=[]
    checks.append(('Satellite evidence',bool(scenes),'Collect an image for the study site.'))
    checks.append(('Ground observations',bool(sensors),'Import timestamped readings with site and source identifiers.'))
    checks.append(('Asset inventory',bool(assets),'Record drainage and access assets with locally agreed scenario thresholds.'))
    checks.append(('Foundation-model runtime',bool(model_ready),'Set up TerraMind for multimodal S1+S2+DEM inference; Prithvi remains available for optical-only experiments.'))
    checks.append(('Site validation',False,'Evaluate on independent sites and storm events before operational interpretation.'))
    return [dict(item=k,state='Available' if ok else 'Needed',next_step='' if ok else why) for k,ok,why in checks]

"""Sentinal X 1.0.0 multimodal aviation-safety research workspace."""
from datetime import datetime,timezone
import csv
import io
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from collection_store import ROOT,Store,load_config,local_path,atomic_json,sha256,validate_config
from current_location import location_picker
from satellite_collection_ui import download_status,collection_settings
from trust_core import ResearchStore,simulate,evaluation,parse_sensors,validate_asset,recommendation_checks,utc_now
from multimodal_pipeline import (parse_multimodal_csv,run_multimodal_pipeline,asset_decision_rows,
    decision_surface_points,parse_foundation_records,template_csv)
from multisource_catalog import discover_multisource
from deployment_health import local_readiness, live_provider_probe
from aerosentinel import (resolution_research_table, fit_temperature,
    domain_reliability_evaluation, domain_evaluation_template_csv)
from direct_candidate import analyze_highres
from imagery_archive import ensure_storage_layout
from aircraft_awareness import TEMPLATE, parse_detections, analyze
from aircraft_detector import infer_geotiff, MODEL_ID
from image_processing import enhance_rgb, denoise_rgb
from quality_control import image_quality_metrics
from sentinal_x_ui import render as sentinal_research
from aircraft_map import render as render_aircraft_map

PROJECT=ROOT.parent
store=Store(ROOT);research=ResearchStore(ROOT)
st.set_page_config(page_title='Sentinal X 1.0.0 | Satellite Aircraft Awareness',page_icon='🌍',layout='wide')
st.markdown('''<style>
.block-container{max-width:1500px;padding-top:2.5rem} [data-testid="stMetric"]{background:#132631;padding:18px;border:1px solid #2c444e;border-radius:14px}
.eyebrow{color:#68d6c0;font-size:12px;letter-spacing:3px;font-weight:700}.hero{font-size:42px;letter-spacing:-1.5px;font-weight:700;margin:5px 0}.subtitle{color:#a9bdc7;margin-bottom:22px;max-width:920px}
</style>''',unsafe_allow_html=True)
with st.sidebar:
    st.title('Sentinal X 1.0.0')
    page=st.radio('Go to',['Aircraft Map','Operations Center','Sentinal X Intelligence','Aircraft Awareness','Research & Validation','Setup & Auto Mode','Satellite Detail','Processing Lab','Advanced Analysis','System Health'],key='navigation')
    st.caption('AUTOMATED RESILIENCE RESEARCH · Local workstation')
    paused=bool(store.setting('paused'))
    if st.button('▶ Resume all automation' if paused else '⏸ Pause all automation',width='stretch'):
        store.setting('paused',not paused);st.rerun()
    st.caption('Observed / Imported / Simulated / Model-derived evidence is kept separate.')
    st.caption('AEROSENTINEL does not automate interception, jamming, targeting, weapons employment, hostile-actor classification, or autonomous flight-release decisions.')


if page != 'Aircraft Map':
    st.markdown('<div class="eyebrow">SENTINAL X / SATELLITE AVIATION RESEARCH / 1.0.0</div><div class="hero">Observe. Understand. Detect. Predict. Trust.</div><div class="subtitle">Satellite-only aircraft awareness from optical, SAR, thermal, temporal and foundation-model observations. Presence and movement remain review candidates bounded by image resolution, acquisition time and model quality.</div>',unsafe_allow_html=True)

@st.fragment(run_every='10s')
def aircraft_map_page():
    render_aircraft_map(store)


def save_upload(upload,folder='evidence'):
    target=ROOT/'data'/folder/(uuid.uuid4().hex+Path(upload.name).suffix.lower())
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(upload.getvalue())
    return target


def assets():return research.records('asset')


def foundation_ready():
    folder=PROJECT/'.foundation-venv'
    return (folder/'ready.json').exists() and (folder/('Scripts/python.exe' if os.name=='nt' else 'bin/python')).exists()


def overview():
    scenes=store.scenes();records=research.records();simulations=research.records('scenario')
    cols=st.columns(4)
    for col,label,value in zip(cols,['Collected scenes','Ground evidence records','Registered assets','Saved scenarios'],
            [len(scenes),len([r for r in records if r['kind'] in ['observation','sensors']]),len(assets()),len(simulations)]):col.metric(label,value)
    left,right=st.columns([1.5,1])
    with left:
        st.subheader('Latest satellite evidence')
        preview=next((s.get('quality',{}).get('preview') for s in scenes if s.get('quality',{}).get('preview')),None)
        if preview and Path(preview).exists():st.image(preview,width='stretch')
        else:st.info('Open Satellite imagery to collect a scene or upload your downloaded GeoTIFF.')
    with right:
        st.subheader('What is available')
        st.dataframe(pd.DataFrame(recommendation_checks(scenes,research.records('sensors'),assets(),foundation_ready())),hide_index=True)
    st.subheader('Latest activity')
    st.dataframe(pd.DataFrame([{k:r[k] for k in ['created','kind','title','origin']} for r in records[:12]]),hide_index=True,width='stretch')
    st.info('Start with the study site and sensor evidence. Run a clearly labelled scenario, review its assumptions, and save a result for comparison.')


def satellite():
    first,second,third=st.tabs(['Library','Collect & upload','Storage & logs'])
    with second:
        location_picker(store,ROOT);download_status(store,ROOT,'trust')
        collection_settings(store,ROOT)
        upload=st.file_uploader('Import a GeoTIFF from an earlier installation or provider',type=['tif','tiff'])
        acquired=st.text_input('Acquisition time, if known (ISO UTC)',placeholder='2026-09-12T06:00:00Z')
        if st.button('Import imagery',disabled=upload is None):
            if acquired:
                from trust_core import timestamp;timestamp(acquired)
            path=save_upload(upload,'imagery_imports')
            store.register(path,sha256(path),dict(source='manual_upload',acquired_at=acquired or None,context_only=True))
            st.success('Registered. The imagery worker will create a preview.')
    with first:
        scenes=store.scenes()
        if not scenes:st.info('No images collected yet. Use Collect & upload.');return
        st.dataframe(pd.DataFrame([dict(name=s['filename'],source=s.get('input_metadata',{}).get('satellite_source') or s.get('input_metadata',{}).get('source'),state=s['state'],acquired=s.get('quality',{}).get('acquired_at'),id=s['id']) for s in scenes]),hide_index=True,width='stretch')
        selected=st.selectbox('Inspect image',scenes,format_func=lambda s:s['filename'])
        q=selected.get('quality',{})
        left,right=st.columns([1.7,1])
        with left:
            if q.get('preview') and Path(q['preview']).exists():st.image(q['preview'],width='stretch')
            else:st.info('Waiting for a preview. Refresh this page after processing.')
        with right:
            st.write('**Processing:**',selected['state']);st.write('**Acquired:**',q.get('acquired_at') or 'Unknown')
            st.write('**CRS:**',q.get('crs'));st.write('**Native pixel size:**',q.get('pixel_size'),q.get('pixel_units'))
            st.write('**Valid sampled pixels:**',q.get('valid_percent_sampled'));st.write('**Provider tile cloud cover:**',q.get('cloud_cover'))
            st.caption('Valid pixels do not imply cloud-free ground visibility. Cloud metadata describes the provider tile.')
            st.code(selected['sha256'],language=None,wrap_lines=True)
        if selected.get('error'):st.error(selected['error'])
        if selected['state']=='FAILED' and st.button('Retry image inspection'):
            store.update(selected['id'],'QUEUED','User requested inspection retry.',error=None);st.rerun()
        if q.get('center'):st.map(pd.DataFrame([q['center']]))
        st.download_button('Export image evidence JSON',json.dumps(selected,indent=2),file_name='scene_evidence.json')
    with third:
        cfg=load_config(ROOT);layout=ensure_storage_layout(ROOT,cfg)
        st.subheader('Where imagery is actually stored')
        if not cfg.get('auto_location_ready',False):
            st.warning('Automatic satellite collection is waiting for a saved AOI/location. Go to Setup & Auto Mode and save device/manual coordinates first.')
        st.markdown(f"**Satellite analytical archive:** `{layout['satellite_archive']}`")
        st.markdown(f"**Satellite image log:** `{layout['satellite_image_log']}`")
        st.markdown(f"**Raw/local GeoTIFF inbox:** `{local_path(ROOT,cfg['watch_dir'])}`")
        st.markdown(f"**High-resolution INPUT inbox:** `{layout['highres_inbox']}`")
        st.markdown(f"**High-resolution processed results:** `{layout['highres_output']}`")
        st.info('Important: the high-resolution inbox is an INPUT folder. AEROSENTINEL does not automatically obtain commercial/fine-resolution imagery. Sentinel-1/Sentinel-2/Landsat products are written to the satellite analytical archive instead.')
        archive=Path(layout['satellite_archive'])
        files=[x for x in archive.rglob('*') if x.is_file() and x.name not in {'archive_index.json'}] if archive.exists() else []
        c1,c2,c3=st.columns(3)
        c1.metric('Archived files',len(files))
        log=archive/'image_log.jsonl'
        lines=[]
        if log.exists():
            try: lines=log.read_text(encoding='utf-8').splitlines()[-50:]
            except Exception: lines=[]
        c2.metric('Image-log entries',len(log.read_text(encoding='utf-8').splitlines()) if log.exists() else 0)
        hs=store.setting('geo_x_highres_state') or {}
        c3.metric('High-res inputs',int(hs.get('input_file_count',0) or 0))
        if lines:
            rows=[]
            for line in reversed(lines):
                try:
                    r=json.loads(line);rows.append({k:r.get(k) for k in ['kind','satellite','scene_id','acquired_at','archived_at']})
                except Exception: pass
            if rows: st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
        else:
            st.info('No satellite archive entries yet. Check the location status and multi-satellite worker in System Health.')


def compare():
    st.subheader('Before / after visual review')
    st.caption('Compare acquisition times and cloud conditions before attributing a difference to physical change.')
    scenes=[s for s in store.scenes() if s.get('quality',{}).get('preview')]
    if len(scenes)<2:st.info('Collect or import at least two images to compare.');return
    a,b=st.columns(2)
    before=a.selectbox('Before',scenes,format_func=lambda s:s['filename'],index=1)
    after=b.selectbox('After',scenes,format_func=lambda s:s['filename'],index=0)
    a.image(before['quality']['preview'],caption=str(before['quality'].get('acquired_at')),width='stretch')
    b.image(after['quality']['preview'],caption=str(after['quality'].get('acquired_at')),width='stretch')
    same=(before['quality'].get('crs')==after['quality'].get('crs') and before['quality'].get('bounds')==after['quality'].get('bounds'))
    st.caption('Bounds and CRS agree.' if same else 'These images have different grids or footprints. This is side-by-side review, not registered change detection.')
    note=st.text_area('Describe visible changes and alternative explanations')
    if st.button('Save comparison',disabled=before['id']==after['id'] or not note.strip()):
        research.add('comparison','Before / after review','Observed',dict(before_id=before['id'],after_id=after['id'],note=note,grid_agreement=same))
        st.success('Comparison saved to the evidence timeline.')


def site_sensors():
    st.subheader('Site asset register')
    with st.form('asset'):
        cols=st.columns(3)
        name=cols[0].text_input('Asset name');site=cols[0].text_input('Site identifier')
        kind=cols[1].selectbox('Asset type',['Runway surface','Taxiway','Apron','Drainage','Access road','Power','Communications','Building','Medical / relief staging','Water supply','Other'])
        criticality=cols[1].slider('Research priority',1,5,3)
        cfg=load_config(ROOT)
        lat=cols[2].number_input('Latitude',-90.0,90.0,float(cfg['auto_latitude']),format='%.5f')
        lon=cols[2].number_input('Longitude',-180.0,180.0,float(cfg['auto_longitude']),format='%.5f')
        threshold=st.number_input('Scenario review threshold (mm of catchment surface storage)',1.0,1000.0,50.0)
        st.caption('User-defined scenario threshold; not a certified asset tolerance or aviation operating limit.')
        save=st.form_submit_button('Register asset')
    if save:
        value=validate_asset(dict(name=name,site=site,type=kind,latitude=lat,longitude=lon,criticality=criticality,threshold_mm=threshold))
        research.add('asset',name,'Imported',value);st.success('Asset registered.')
    items=assets()
    if items:
        df=pd.DataFrame([dict(id=r['id'],**r['payload']) for r in items]);st.dataframe(df,hide_index=True,width='stretch')
        st.map(df.rename(columns={'latitude':'lat','longitude':'lon'}))
    st.subheader('Ground sensor readings')
    st.caption('CSV import: timestamp, site, rain_mm_h, water_level_m, source. This is an imported snapshot, not a live sensor connection.')
    template='timestamp,site,rain_mm_h,water_level_m,source\n2026-09-12T06:00:00Z,EXAMPLE_SITE,12.5,1.2,EXAMPLE_GAUGE\n'
    st.download_button('Download CSV template (example row)',template,file_name='sensor_template.csv')
    upload=st.file_uploader('Import sensor CSV',type=['csv'])
    if st.button('Validate & save readings',disabled=upload is None):
        rows=parse_sensors(upload.getvalue().decode('utf-8-sig'));research.add('sensors',upload.name,'Imported',dict(readings=rows))
        st.success(f'{len(rows)} readings saved.')
    readings=research.records('sensors')
    if readings:
        selected=st.selectbox('Reading set',readings,format_func=lambda r:r['title'])
        df=pd.DataFrame(selected['payload']['readings']);st.dataframe(df,hide_index=True)
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(df['timestamp'].max())).total_seconds()/3600
        st.caption(f'Newest reading is {age:.1f} hours old. Source reliability and gauge calibration need independent confirmation.')
        st.line_chart(df.set_index('timestamp')[['rain_mm_h']])



def multimodal_twin():
    st.subheader('Multimodal predictive digital twin')
    st.info('FULL SOFTWARE PIPELINE · Sentinel-1 SAR + Sentinel-2 optical + DEM + weather/rainfall + historical imagery feed a multimodal representation, temporal analysis, uncertainty/reliability audit, predictive-twin state and decision support. A strict learned-foundation run requires timestamp-matched TerraMind feature records.')

    with st.expander('Architecture contract implemented on this page', expanded=False):
        st.code('''MULTI-SOURCE EARTH OBSERVATION
  Sentinel-1 SAR + Sentinel-2 Optical + DEM
  + Weather / Rainfall + Historical imagery
                    ↓
       Multimodal Foundation Model
                    ↓
          Temporal Representation
                    ↓
     Change Detection | Anomaly Detection
          | Land/Surface Understanding
                    ↓
          Uncertainty Estimation
                    ↓
             RELIABILITY AUDIT
                    ↓
 Current State | Future Prediction | Risk
                    ↓
        PREDICTIVE DIGITAL TWIN
                    ↓
          Decision-Support Map
                    ↓
 Airfield Risk | Disaster Risk | Infrastructure Resilience''', language=None)
        st.caption('TerraMind provides the learned spatial EO backbone for Sentinel-1 + Sentinel-2 + DEM. Weather/rainfall and historical-image context enter explicit adapters in the same hybrid foundation stage before temporal processing. If no valid TerraMind record is supplied, the software uses a transparent fallback and marks the strict architecture run as incomplete.')

    with st.expander('0 · Discover all five EO source groups', expanded=False):
        cfg=load_config()
        d1,d2,d3=st.columns(3)
        lat=d1.number_input('AOI latitude',-90.0,90.0,float(cfg.get('auto_latitude',0.0)),format='%.6f',key='mm_lat')
        lon=d1.number_input('AOI longitude',-180.0,180.0,float(cfg.get('auto_longitude',0.0)),format='%.6f',key='mm_lon')
        radius=d1.slider('Discovery radius (km)',1.0,50.0,5.0,.5,key='mm_radius')
        today=datetime.now(timezone.utc).date()
        start_date=d2.date_input('Start date',today-pd.Timedelta(days=30),key='mm_start')
        end_date=d2.date_input('End date',today-pd.Timedelta(days=6),key='mm_end')
        d3.caption('Discovery queries Earth Search for Sentinel-1, Sentinel-2 and Copernicus DEM; Open-Meteo provides rainfall context; older Sentinel-2 scenes provide historical-imagery inventory. Discovery is not the same as analysis-ready preprocessing.')
        if d3.button('Discover multi-source EO',type='primary'):
            try:
                st.session_state['multisource_discovery']=discover_multisource(lat,lon,start_date,end_date,radius)
            except Exception as exc:
                st.error(f'Discovery failed: {type(exc).__name__}: {exc}')
        discovered=st.session_state.get('multisource_discovery')
        if discovered:
            counts=[dict(source=k,count=v['count'],collection=v['collection']) for k,v in discovered['source_groups'].items()]
            st.dataframe(pd.DataFrame(counts),hide_index=True,width='stretch')
            st.download_button('Download source-discovery manifest',json.dumps(discovered,indent=2),file_name='trust_geo_multisource_discovery.json')
            st.caption(discovered['processing_note'])

    left,right=st.columns([1.25,1])
    with left:
        st.markdown('**1 · Load aligned multi-source features**')
        st.caption('Each row is one observation time. The template includes all five architecture source groups and optional forecast rainfall. Blank cells remain missing and reduce reliability.')
        st.download_button('Download multimodal CSV template',template_csv(),file_name='multimodal_feature_template.csv')
        upload=st.file_uploader('Aligned multimodal feature CSV',type=['csv'],key='multimodal_features')
        foundation_uploads=st.file_uploader('Optional TerraMind feature JSON record(s)',type=['json'],accept_multiple_files=True,key='multimodal_foundation_records')
        st.caption('Use Foundation lab → TerraMind multimodal to create these learned S1+S2+DEM records. Exact timestamps are matched to the feature table.')
    with right:
        st.markdown('**2 · Set reliability / projection controls**')
        horizon=st.slider('Short-horizon projection (table steps)',1,12,1)
        abstain=st.slider('Abstain below reliability',0.0,1.0,.45,.05)
        st.caption('Future prediction uses recent state dynamics plus optional forecast rainfall. Sensitivity bands are not calibrated confidence intervals.')

    if st.button('Run full multimodal architecture',type='primary',disabled=upload is None):
        rows=parse_multimodal_csv(upload.getvalue().decode('utf-8-sig'))
        foundation_records=[]
        for item in foundation_uploads or []:
            foundation_records.extend(parse_foundation_records(item.getvalue()))
        result=run_multimodal_pipeline(rows,horizon,abstain,foundation_records=foundation_records)
        st.session_state['multimodal_result']=result

    result=st.session_state.get('multimodal_result')
    if not result:
        st.warning('Load the multi-source feature table to run the architecture. Add TerraMind records when you want the learned multimodal foundation-model path rather than the transparent fallback.')
        return

    compliance=pd.DataFrame([dict(stage=k,status=v) for k,v in result['architecture_compliance'].items()])
    strict=bool(result['architecture_compliance']['strict_architecture_run'])
    if strict:
        st.success('STRICT ARCHITECTURE RUN: every timestamp has all five source groups and a matching TerraMind S1GRD + S2L2A + DEM record.')
    else:
        st.warning('The full software architecture is present, but this run is not strict. Every timestamp must have all five source groups and a matching TerraMind record containing S1GRD + S2L2A + DEM.')
    with st.expander('Architecture compliance for this run',expanded=True):
        st.dataframe(compliance,hide_index=True,width='stretch')
        st.caption(result['encoder'])

    latest=result['latest']
    st.markdown('**3 · Reliability-audited predictive twin output**')
    cols=st.columns(5)
    cols[0].metric('Current state',f"{latest['current_state']:.3f}")
    cols[1].metric('Future projection',f"{latest['future_prediction']:.3f}")
    cols[2].metric('Risk screening',f"{latest['risk_screening']:.3f}")
    cols[3].metric('Reliability',f"{latest['reliability']:.3f}")
    cols[4].metric('Audit',latest['audit'])
    st.caption(f"Risk sensitivity band: {latest['risk_sensitivity_p10']:.3f} – {latest['risk_sensitivity_p90']:.3f}. This is perturbation sensitivity, not a calibrated probability interval.")
    if latest['audit']=='ABSTAIN':
        st.error('Reliability audit says ABSTAIN. Downstream screening remains visible for traceability but is not supported for interpretation.')
    elif latest['audit']=='REVIEW':
        st.warning('Reliability audit says REVIEW. Inspect missing modalities, source disagreement, perturbation sensitivity and temporal gaps.')
    else:
        st.success('Reliability audit passed the prototype checks. This still does not establish deployment readiness or calibrated risk probability.')

    summary=pd.DataFrame([
        dict(source=name,coverage=v['coverage'],rows_available=v['rows_available'],features=', '.join(v['features']))
        for name,v in result['source_summary'].items()
    ])
    st.dataframe(summary,hide_index=True,width='stretch')

    twin=pd.DataFrame(result['predictive_digital_twin']).set_index('timestamp')
    change=pd.DataFrame([{'timestamp':r['timestamp'],'change_score':r['change_score']} for r in result['change_detection']]).set_index('timestamp')
    anomaly=pd.DataFrame([{'timestamp':r['timestamp'],'anomaly_score':r['anomaly_score']} for r in result['anomaly_detection']]).set_index('timestamp')
    unc=pd.DataFrame(result['uncertainty_estimation']).set_index('timestamp')
    chart=twin[['current_state','future_prediction','risk_screening','infrastructure_resilience_screening']].join(change).join(anomaly).join(unc[['uncertainty_score','reliability_score','perturbation_sensitivity']])
    st.line_chart(chart)

    a,b,c=st.columns(3)
    with a:
        st.markdown('**Change detection**')
        st.dataframe(change.tail(12),width='stretch')
    with b:
        st.markdown('**Anomaly detection**')
        st.dataframe(anomaly.tail(12),width='stretch')
    with c:
        st.markdown('**Uncertainty & audit**')
        st.dataframe(unc[['source_count','source_coverage','source_disagreement','perturbation_sensitivity','uncertainty_score','reliability_score','audit']].tail(12),width='stretch')

    surface=pd.DataFrame(result['land_surface_understanding']).set_index('timestamp')
    with st.expander('Land / surface understanding'):
        st.dataframe(surface,width='stretch')
        st.caption('These are interpretable relative wetness, SAR-water, vegetation-stress, terrain and historical-deviation proxies. They are not semantic class probabilities.')

    st.markdown('**4 · Decision-support map and domain screening**')
    domain=st.columns(3)
    domain[0].metric('Airfield risk screening',f"{latest['airfield_risk_screening']:.3f}",delta=f"future {latest['airfield_future_screening']:.3f}")
    domain[1].metric('Disaster risk screening',f"{latest['disaster_risk_screening']:.3f}",delta=f"future {latest['disaster_future_screening']:.3f}")
    domain[2].metric('Infrastructure resilience',f"{latest['infrastructure_resilience_screening']:.3f}",delta=f"future {latest['infrastructure_future_resilience']:.3f}")
    site_assets=assets()
    if site_assets:
        decisions=asset_decision_rows(site_assets,latest)
        map_df=pd.DataFrame(decisions)
        surface_points=decision_surface_points(decisions)
        try:
            import pydeck as pdk
            layers=[]
            if len(surface_points)>1:
                layers.append(pdk.Layer('HeatmapLayer',data=surface_points,get_position='[longitude, latitude]',get_weight='priority',radius_pixels=55))
            layers.append(pdk.Layer('ScatterplotLayer',data=decisions,get_position='[longitude, latitude]',get_radius=35,get_fill_color='[40, 190, 170, 190]',pickable=True))
            center_lat=float(map_df['latitude'].mean());center_lon=float(map_df['longitude'].mean())
            deck=pdk.Deck(layers=layers,initial_view_state=pdk.ViewState(latitude=center_lat,longitude=center_lon,zoom=11),tooltip={'text':'{asset}\nDomain: {domain}\nPriority: {screening_priority}'})
            st.pydeck_chart(deck,use_container_width=True)
        except Exception:
            st.map(map_df.rename(columns={'latitude':'lat','longitude':'lon'}))
        st.dataframe(map_df,hide_index=True,width='stretch')
        st.caption('The heat surface is an inverse-distance visualization of registered asset review priority. It is NOT an inundation, damage or structural-failure raster.')
    else:
        st.info('Register site assets under Site & sensors to populate the decision-support map.')

    c1,c2=st.columns(2)
    if c1.button('Save multimodal twin evidence'):
        research.add('multimodal_twin','Multimodal predictive twin run','Model-derived',result)
        st.success('Multimodal architecture output saved to the research evidence timeline.')
    c2.download_button('Download multimodal twin JSON',json.dumps(result,indent=2),file_name='trust_geo_multimodal_twin.json')
    st.caption(result['score_scope'])

def scenario():
    st.subheader('Rainfall & drainage scenario twin')
    st.warning('SIMULATION · Catchment water balance, not a calibrated flood forecast or an airfield operating decision.')
    with st.form('scenario'):
        title=st.text_input('Scenario name','Rainfall and drainage sensitivity')
        site=st.selectbox('Scenario site',sorted({a['payload']['site'] for a in assets()}) or ['Hypothetical site'])
        a,b,c=st.columns(3)
        rain=a.number_input('Rainfall intensity (mm/h)',0.0,300.0,30.0)
        hours=a.slider('Duration (hours)',1,48,6)
        area=a.number_input('Catchment area (ha)',0.01,10000.0,25.0)
        runoff=b.slider('Runoff coefficient',0.0,1.0,.8)
        infiltration=b.number_input('Infiltration capacity (mm/h)',0.0,100.0,5.0)
        drain=b.number_input('Drain capacity (m³/s)',0.0,10000.0,.5,step=.1)
        initial=c.number_input('Initial surface storage (mm)',0.0,1000.0,0.0)
        uncertainty=c.slider('Assumption variation (±%)',0,80,20)
        mitigation=c.number_input('Alternative drain capacity (m³/s)',0.0,10000.0,1.0,step=.1)
        source=st.text_input('Basis of inputs','User-defined hypothetical scenario')
        run=st.form_submit_button('Run scenario and alternative',type='primary')
    if run:
        parameters=dict(rain_mm_h=rain,hours=hours,area_ha=area,runoff=runoff,infiltration_mm_h=infiltration,drain_m3_s=drain,initial_mm=initial,uncertainty_percent=uncertainty)
        st.session_state['scenario_result']=dict(title=title,site=site,basis=source,baseline=simulate(parameters),alternative=simulate(dict(parameters,drain_m3_s=mitigation)))
    result=st.session_state.get('scenario_result')
    if result:
        baseline,alternative=result['baseline'],result['alternative']
        a,b=st.columns(2);a.metric('Baseline peak storage',f"{baseline['peak_mm']:.1f} mm");b.metric('Alternative peak storage',f"{alternative['peak_mm']:.1f} mm")
        df=pd.DataFrame(baseline['series']).set_index('hour');df['alternative_mm']=[r['surface_storage_mm'] for r in alternative['series']]
        st.line_chart(df[['surface_storage_mm','p10_mm','p90_mm','alternative_mm']])
        st.caption(baseline['uncertainty']);st.caption(baseline['assumptions'])
        st.dataframe(df.reset_index(),hide_index=True,width='stretch')
        site_assets=[a for a in assets() if a['payload']['site']==result.get('site')]
        if site_assets:
            st.caption('Screening the same catchment scenario against user-set asset review thresholds. This is not spatial inundation analysis.')
            st.dataframe(pd.DataFrame([dict(asset=a['title'],threshold_mm=a['payload']['threshold_mm'],
                review='Review scenario exposure' if baseline['peak_mm']>=a['payload']['threshold_mm'] else 'Below entered threshold') for a in site_assets]),hide_index=True)
        if st.button('Save scenario evidence'):
            research.add('scenario',result['title'],'Simulated',result);st.success('Scenario and assumptions saved.')
        st.download_button('Download scenario JSON',json.dumps(result,indent=2),file_name='trust_geo_scenario.json')
    saved=research.records('scenario')
    if saved:
        with st.expander('Previous scenarios'):
            selected=st.selectbox('Saved scenario',saved,format_func=lambda r:r['title']);st.json(selected['payload'],expanded=False)


def launch_job(command,status):
    previous=json.loads(status.read_text()) if status.exists() else {}
    if previous.get('state') in ['STARTING','RUNNING']:
        if previous.get('pid'):
            try:os.kill(previous['pid'],0)
            except OSError:pass
            else:raise ValueError('A model/setup job is already running. Check its log before starting another.')
        elif time.time()-previous.get('time',0)<120:
            raise ValueError('The previous job is still starting. Refresh this page shortly.')
    log=status.with_suffix('.log');log.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(status,dict(state='STARTING',time=time.time(),log=str(log)))
    try:
        with log.open('w',encoding='utf-8') as stream:
            subprocess.Popen(command,cwd=PROJECT,stdout=stream,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    except OSError as exc:
        atomic_json(status,dict(state='FAILED',time=time.time(),error=str(exc)))
        raise


def foundation():
    st.subheader('Multimodal foundation-model research')
    st.info('TerraMind is the preferred multimodal backbone for the architecture. Prithvi is retained as an optical-only legacy experiment. Neither output is a task probability without a validated downstream head.')
    runtime=PROJECT/'.foundation-venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    ready=foundation_ready()
    st.write('**Foundation runtime:**','Installed; live inference still requires valid inputs and model weights.' if ready else 'Not installed')
    st.caption('Optional setup downloads CPU PyTorch, TerraTorch and dependencies into a separate environment. Model weights download on first use. Core non-learned analysis works without this environment.')
    setup_status=ROOT/'data/foundation/setup-status.json'
    if st.button('Install / update foundation-model environment',disabled=False if not ready else False):
        launch_job([sys.executable,str(PROJECT/'setup_foundation.py')],setup_status);st.success('Setup started. Refresh this page to see its status.')
    if setup_status.exists():st.json(json.loads(setup_status.read_text()),expanded=False)

    tm_tab,prithvi_tab=st.tabs(['TerraMind multimodal','Prithvi optical legacy'])
    with tm_tab:
        st.link_button('Official TerraMind / TerraTorch guide','https://github.com/torchgeo/terratorch/blob/main/docs/guide/terramind.md')
        st.markdown('Provide **co-located inputs** for one timestamp: a 12-band Sentinel-2 L2A stack, a calibrated two-band Sentinel-1 VV/VH stack in dB, and a one-band DEM in metres. Sentinel-1 and DEM are reprojected onto the Sentinel-2 center 224 × 224 grid.')
        a,b,c=st.columns(3)
        s2_upload=a.file_uploader('Sentinel-2 L2A 12-band GeoTIFF',type=['tif','tiff'],key='tm_s2')
        s1_upload=b.file_uploader('Sentinel-1 VV/VH dB GeoTIFF',type=['tif','tiff'],key='tm_s1')
        dem_upload=c.file_uploader('DEM elevation GeoTIFF',type=['tif','tiff'],key='tm_dem')
        s2_units=st.selectbox('Sentinel-2 units',['scaled_10000','reflectance'],key='tm_s2_units')
        acquired_tm=st.text_input('Acquisition time (ISO UTC)',placeholder='2026-09-12T06:00:00Z',key='tm_acquired')
        confirmed_tm=st.checkbox('I checked S2 band order/scaling, S1 radiometric calibration to dB, DEM units, geolocation and quality masks.',key='tm_confirm')
        tm_status=ROOT/'data/foundation/terramind-job-status.json'
        if st.button('Extract TerraMind multimodal features',type='primary',disabled=not (ready and s2_upload and s1_upload and dem_upload and acquired_tm and confirmed_tm)):
            from trust_core import timestamp;timestamp(acquired_tm)
            s2_path=save_upload(s2_upload,'foundation/terramind_inputs')
            s1_path=save_upload(s1_upload,'foundation/terramind_inputs')
            dem_path=save_upload(dem_upload,'foundation/terramind_inputs')
            output=s2_path.with_suffix('.terramind.json')
            request=s2_path.with_suffix('.terramind-request.json')
            atomic_json(request,dict(s2_input=str(s2_path),s1_input=str(s1_path),dem_input=str(dem_path),output=str(output),s2_units=s2_units,acquired=acquired_tm,status=str(tm_status)))
            launch_job([str(runtime),str(ROOT/'app/terramind_job.py'),str(request)],tm_status)
            st.success('TerraMind extraction started. Refresh for status.')
        if tm_status.exists():
            state=json.loads(tm_status.read_text());st.json(state,expanded=False)
            if state.get('state')=='COMPLETE' and Path(state['output']).exists():
                result=json.loads(Path(state['output']).read_text())
                st.write('**Model:**',result['model'],' · **Feature dimensions:**',result['dimension'])
                st.caption(result['interpretation'])
                st.download_button('Download TerraMind feature record',json.dumps(result,indent=2),file_name='terramind_multimodal_features.json')
                if st.button('Save TerraMind record to evidence'):
                    research.add('foundation','TerraMind multimodal feature extraction','Model-derived',result);st.success('Model-derived TerraMind record saved.')
        st.caption('Weather/rainfall and historical-image context are fused through explicit context adapters in the overall hybrid foundation stage. They are not falsely presented as TerraMind pre-trained raw modalities.')

    with prithvi_tab:
        st.link_button('Official Prithvi model card','https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL')
        st.markdown('Supply a **six-band HLS reflectance GeoTIFF**, ordered blue, green, red, narrow NIR, SWIR1, SWIR2, on a 30 m grid. The center 224 × 224 patch is used.')
        upload=st.file_uploader('HLS input GeoTIFF',type=['tif','tiff'],key='hls')
        units=st.selectbox('Input units',['scaled_10000','reflectance'],key='prithvi_units')
        acquired=st.text_input('HLS acquisition time (ISO UTC)',placeholder='2026-09-12T06:00:00Z',key='prithvi_acquired')
        confirmed=st.checkbox('I checked the band order, reflectance units and cloud/quality mask.',key='prithvi_confirm')
        job_status=ROOT/'data/foundation/job-status.json'
        if st.button('Extract Prithvi optical features',disabled=not (ready and upload and acquired and confirmed)):
            from trust_core import timestamp;timestamp(acquired)
            path=save_upload(upload,'foundation/inputs');output=path.with_suffix('.features.json')
            request=path.with_suffix('.request.json');atomic_json(request,dict(input=str(path),output=str(output),units=units,acquired=acquired,status=str(job_status)))
            launch_job([str(runtime),str(ROOT/'foundation_job.py'),str(request)],job_status);st.success('Prithvi extraction started. Refresh for status.')
        if job_status.exists():
            state=json.loads(job_status.read_text());st.json(state,expanded=False)
            if state.get('state')=='COMPLETE' and Path(state['output']).exists():
                result=json.loads(Path(state['output']).read_text());st.write('**Feature dimensions:**',result['dimension']);st.caption(result['interpretation'])
                st.download_button('Download Prithvi feature record',json.dumps(result,indent=2),file_name='prithvi_features.json')
                if st.button('Save Prithvi record to evidence'):
                    research.add('foundation','Prithvi feature extraction','Model-derived',result);st.success('Model-derived record saved.')
        st.caption('Prithvi is optical-only in this package and therefore does not satisfy the strict multimodal-foundation stage by itself.')

def evaluate():
    st.subheader('Offline reliability evaluation')
    st.caption('Import binary flood/damage labels and model probabilities. A value of 1 means the named event is present. No model is run on this page.')
    template='sample_id,label,probability\nEXAMPLE_A,0,0.1\nEXAMPLE_B,1,0.8\n'
    st.download_button('Download evaluation template (example rows)',template,file_name='evaluation_template.csv')
    task=st.text_input('Task and model identifier','Flood presence — external predictions')
    upload=st.file_uploader('Labelled predictions CSV',type=['csv'],key='eval')
    threshold=st.slider('Decision threshold',0.0,1.0,.5)
    if st.button('Calculate metrics',disabled=upload is None):
        rows=list(csv.DictReader(io.StringIO(upload.getvalue().decode('utf-8-sig'))))
        result=evaluation(rows,threshold);st.session_state['evaluation_result']=dict(task=task,**result)
    result=st.session_state.get('evaluation_result')
    if result:
        cols=st.columns(4)
        for col,key in zip(cols,['precision','recall','f1','brier']):col.metric(key.title(),'Undefined' if result[key] is None else f'{result[key]:.3f}')
        st.dataframe(pd.DataFrame(result['calibration']),hide_index=True,width='stretch')
        st.json({k:v for k,v in result.items() if k!='calibration'},expanded=False)
        if st.button('Save evaluation record'):
            research.add('evaluation',result['task'],'Imported',result);st.success('Evaluation saved.')
        st.download_button('Export evaluation JSON',json.dumps(result,indent=2),file_name='trust_geo_evaluation.json')


def roadmap():
    st.subheader('Recommendations and readiness')
    st.markdown('''1. **Start with one civil study site and one hazard:** rainfall, drainage and access disruption.
2. **Add real ground truth:** timestamped gauges, inspections and surveyed drainage/terrain.
3. **Use suitable data for each task:** RGB for viewing; quality-masked multispectral or SAR data for validated hazard models.
4. **Benchmark before claiming reliability:** compare a simple baseline, a trained conventional model and a fine-tuned foundation model on held-out sites and storms.
5. **Calibrate uncertainty and abstention:** test cloud, missing data and sensor outage cases; report what the system cannot infer.
6. **Upgrade the twin after calibration:** connect a validated hydraulic model and measured asset dependencies.
7. **Prepare multi-user deployment separately:** authenticated roles, protected audit records, backups and independent review.''')
    st.dataframe(pd.DataFrame(recommendation_checks(store.scenes(),research.records('sensors'),assets(),foundation_ready())),hide_index=True,width='stretch')
    st.subheader('Deployment readiness')
    readiness=local_readiness(ROOT)
    st.metric('Local deployment state',readiness['state'],f"{readiness['passed']}/{readiness['total']} checks passing")
    st.dataframe(pd.DataFrame(readiness['checks']),hide_index=True,width='stretch')
    if st.button('Run live provider connectivity test'):
        with st.spinner('Checking public satellite and weather providers…'):
            probe=live_provider_probe(ROOT)
        st.dataframe(pd.DataFrame(probe['results']),hide_index=True,width='stretch')
        if probe['ok']:st.success('Provider connectivity is available from this workstation.')
        else:st.warning('One or more providers are unavailable. Auto Mode will keep retrying and preserve prior evidence.')
    st.caption('The live weather probe sends the saved monitoring coordinate to Open-Meteo. The satellite service uses the saved AOI when searching the STAC catalog.')
    st.subheader('Local service status')
    for key in ['supervisor','imagery_worker','multisatellite_download','ops_status','weather_latest','geo_x_latest','ops_latest_report']:
        with st.expander(key):st.json(store.setting(key) or {'state':'No report'})
    st.caption('The satellite catalog and model downloads need internet. A green process status is not evidence of model accuracy or field readiness.')
    logs=sorted((ROOT/'data/collection/logs').glob('*.log')) if (ROOT/'data/collection/logs').exists() else []
    if logs:
        path=st.selectbox('Service log',logs,format_func=lambda p:p.name)
        st.code(path.read_text(encoding='utf-8',errors='replace')[-12000:],language=None,wrap_lines=True)
    st.download_button('Download research blueprint',(PROJECT/'AEROSENTINEL_BLUEPRINT.md').read_text(),file_name='AEROSENTINEL_BLUEPRINT.md')


def _short_time(value):
    if not value:return 'Not yet'
    try:return datetime.fromisoformat(str(value).replace('Z','+00:00')).astimezone(timezone.utc).strftime('%d %b %H:%M UTC')
    except Exception:return str(value)


def _write_auto_config(**updates):
    cfg=dict(load_config(ROOT));cfg.update(updates);validate_config(cfg);atomic_json(ROOT/'collection_config.json',cfg);return cfg


def _start_auto_mode(cfg):
    updated=_write_auto_config(auto_collect_enabled=True,ops_automation_enabled=True,ops_weather_enabled=True,
        ops_auto_report_enabled=True,ops_report_interval_minutes=5,ops_weather_interval_minutes=5,geo_x_enabled=True,geo_x_highres_enabled=True,
        auto_interval_minutes=max(30,int(cfg.get('auto_interval_minutes',30))),
        multi_satellites=cfg.get('multi_satellites') or ['sentinel-1','sentinel-2','landsat-8-9'])
    store.setting('paused',False);store.setting('multisatellite_request',uuid.uuid4().hex);store.setting('ops_request',uuid.uuid4().hex)
    return updated


def _aircraft_summary():
    result=store.setting('aircraft_awareness_latest') or {}
    st.subheader('Satellite aircraft awareness')
    highres=store.setting('highres_collection_status') or {}
    st.caption('Automatic high-resolution collection: '+str(highres.get('state','STARTING')).replace('_',' ')+' · Configure on Aircraft Awareness.')
    st.caption('Only satellite detector outputs are used. Collection and surface fusion alone cannot establish aircraft presence.')
    cols=st.columns(3)
    cols[0].metric('Candidate observations',sum(x.get('presence')=='CANDIDATE' for x in result.get('observations',[])))
    cols[1].metric('Possible movements',sum(x.get('state')=='POSSIBLE MOVEMENT' for x in result.get('movement_candidates',[])))
    cols[2].metric('Anomaly candidates',len(result.get('anomalies',[])))
    if not result:st.info('NO OBSERVATIONS — import validated satellite detector output on Aircraft Awareness.')


def aircraft_awareness():
    st.subheader('Satellite-only aircraft awareness')
    location_picker(store, ROOT)
    cfg=load_config(ROOT)
    st.markdown('### Automatic high-resolution collection')
    st.write('Planet SkySat archive → RGB GeoTIFF crop around your saved location → aircraft candidate detection. Checks for new accessible imagery every 30 minutes. Archive downloads consume your Planet quota; acquisition times depend on available coverage.')
    if not os.environ.get('PL_API_KEY','').strip():
        st.info('One-time setup: set the PL_API_KEY environment variable on this computer with an account entitled to SkySat downloads, then restart the app. See AUTO_HIGHRES_SETUP.md. Do not paste the key into chat.')
    with st.form('highres_auto_settings'):
        enabled=st.checkbox('Automatically collect SkySat archive imagery',value=bool(cfg.get('highres_auto_enabled',True)))
        infer=st.checkbox('Run aircraft detector after download',value=bool(cfg.get('highres_auto_infer',True)))
        radius=st.slider('Collection half-width around location (km)',0.1,1.0,float(cfg.get('highres_radius_km',0.5)),0.1)
        lookback=st.number_input('Maximum scene age (days)',1,365,int(cfg.get('highres_lookback_days',90)))
        clouds=st.slider('Maximum whole-scene cloud fraction',0.0,1.0,float(cfg.get('highres_max_cloud',0.2)),0.05)
        apply=st.form_submit_button('Save automatic collection settings')
    if apply:
        _write_auto_config(highres_auto_enabled=enabled,highres_auto_infer=infer,highres_radius_km=float(radius),
                           highres_lookback_days=int(lookback),highres_max_cloud=float(clouds))
        store.setting('highres_request',uuid.uuid4().hex)
        st.rerun()
    status=store.setting('highres_collection_status') or {'state':'STARTING'}
    st.metric('High-resolution collector',status.get('state','UNKNOWN').replace('_',' '))
    if status.get('message'):st.info(status['message'])
    if status.get('acquired_at'):
        st.caption(f"Scene acquired: {status['acquired_at']} · saved-location pixel valid: {status.get('location_pixel_valid','checking')} · whole crop valid: {status.get('full_aoi_valid','checking')}")
    if status.get('path') and Path(status['path']).is_file():
        st.download_button('Download collected RGB GeoTIFF',Path(status['path']).read_bytes(),file_name='location_rgb.tif',mime='image/tiff')
        st.caption('SkySat visual products use provider enhancement; the output pixel spacing is not independent native resolution. Cloud filtering is scene-wide and does not guarantee a clear view of the location.')
    if st.button('Check for new high-resolution imagery now'):
        store.setting('highres_request',uuid.uuid4().hex)
        st.success('Collection check requested. Refresh to see progress.')
    st.markdown('### Analyze a manual image')
    st.write('Run the published optical aircraft model on a high-resolution RGB GeoTIFF. The checkpoint downloads on first use. The collected Sentinel/Landsat scene products are environmental context and are generally too coarse for aircraft inference.')
    st.caption('Research checkpoint: '+MODEL_ID)
    with st.form('aircraft_geotiff_inference'):
        image=st.file_uploader('Georeferenced RGB satellite GeoTIFF (native GSD ≤ 2 m)',type=['tif','tiff'],key='aircraft_geotiff')
        site_id=st.text_input('Site ID',value=str(load_config(ROOT).get('auto_area_name') or 'study-area'))
        scene_id=st.text_input('Scene ID',help='Unique identifier for this acquisition')
        acquired_at=st.text_input('Actual satellite acquisition time (ISO-8601 with timezone)',placeholder='2026-09-25T10:00:00Z')
        confidence=st.slider('Minimum model score',0.10,0.90,0.35,0.05)
        cloud=st.slider('Optical cloud fraction in study image',0.0,1.0,0.0,0.05)
        run_image=st.form_submit_button('Detect aircraft candidates in image',disabled=image is None)
    if run_image:
        try:
            path=save_upload(image,'aircraft_inputs')
            with st.spinner('Running tiled optical inference; first use downloads the model checkpoint…'):
                detection=infer_geotiff(path,site_id=site_id,scene_id=scene_id,acquired_at=acquired_at,
                                       confidence=float(confidence),cloud_fraction=float(cloud))
            history=store.setting('aircraft_detector_rows') or []
            history=[r for r in history if (r.get('site_id'),r.get('scene_id')) != (site_id,scene_id)]
            history=(history+detection['rows'])[-10000:]
            store.setting('aircraft_detector_rows',history)
            result=analyze(history+(store.setting('aircraft_import_rows') or []))
            result.update(source='optical_geotiff_detector', model_id=detection['model_id'],
                          model_sha256=detection['model_sha256'], input_sha256=detection['source_sha256'],
                          analyzed_at=utc_now(), latest_scene=detection['state'], tiles_analyzed=detection['tiles_analyzed'])
            store.setting('aircraft_awareness_latest',result)
            research.add('satellite_aircraft_awareness',scene_id,'Model-derived',result)
            st.success(f"Analyzed {detection['tiles_analyzed']} tiles; {detection['detections']} model detections. {detection['state']}.")
        except (ValueError,RuntimeError,OSError) as exc:
            st.error(str(exc))
    st.markdown('### Import other satellite model outputs')
    st.caption('SAR, thermal and foundation outputs can be added with the CSV contract. Thermal or foundation evidence alone remains insufficient for a presence candidate.')
    st.download_button('Download detector CSV template',TEMPLATE,file_name='satellite_aircraft_detections.csv',mime='text/csv')
    upload=st.file_uploader('Satellite detector CSV',type=['csv'],key='aircraft_detector_upload')
    if st.button('Analyze satellite observations',disabled=upload is None):
        try:
            source_bytes=upload.getvalue()
            rows=parse_detections(source_bytes.decode('utf-8-sig'))
            store.setting('aircraft_import_rows',rows)
            result=analyze((store.setting('aircraft_detector_rows') or [])+rows)
            result['input_name']=upload.name
            result['input_sha256']=hashlib.sha256(source_bytes).hexdigest()
            result['analyzed_at']=utc_now()
            store.setting('aircraft_awareness_latest',result)
            research.add('satellite_aircraft_awareness',upload.name,'Imported',result)
            st.success('Satellite observation analysis saved for review.')
        except (ValueError,UnicodeError) as exc:
            st.error(str(exc))
    result=store.setting('aircraft_awareness_latest') or {}
    if result:
        st.button('Open Aircraft Map',on_click=lambda: st.session_state.update(navigation='Aircraft Map'))
        st.metric('Assessment state',result['state'])
        for title,key in [('Presence candidates','observations'),('Possible movement between acquisitions','movement_candidates'),('Anomaly events for review','anomalies')]:
            st.markdown('### '+title)
            if result.get(key):st.dataframe(pd.DataFrame(result[key]),hide_index=True,width='stretch')
            else:st.info('No '+title.lower()+' in this set.')
        st.warning(result.get('limitations',''))
        st.download_button('Download analysis JSON',json.dumps(result,indent=2),file_name='satellite_aircraft_awareness.json',mime='application/json')
    else:
        st.info('NO OBSERVATIONS. A missing detector output does not imply an empty sky.')
    st.caption('Separate scenes share site_id to compare acquisitions. Each scene_id identifies one acquisition. Optical cloud, GSD, quality, registration error and model provenance affect interpretation. Sparse revisits cannot identify an aircraft or reconstruct a route.')


def operations_center():
    cfg=load_config(ROOT);sat_status=store.setting('multisatellite_download') or {};ops_status=store.setting('ops_status') or {}
    assessment=store.setting('ops_latest_assessment') or {}
    geo_x=store.setting('aerosentinel_latest') or store.setting('geo_x_latest') or {}
    if not cfg.get('auto_location_ready'):
        st.warning('First-time setup: v2.8.0 will request your browser/device location. Allow it once and Auto Mode starts automatically.')
        location_picker(store,ROOT)
        st.caption('I do not preload or guess sensitive facility coordinates. Device location is used only after your browser grants permission.')
        with st.expander('Manual coordinate fallback'):
            with st.form('v5_quick_setup'):
                name=st.text_input('Area name','My monitoring area')
                c1,c2,c3=st.columns(3)
                lat=c1.number_input('Latitude',-79.0,83.0,float(cfg['auto_latitude']),format='%.5f')
                lon=c2.number_input('Longitude',-179.8,179.8,float(cfg['auto_longitude']),format='%.5f')
                radius=c3.number_input('Half-width (km)',0.5,20.0,float(cfg['auto_radius_km']),0.5)
                go=st.form_submit_button('Save area & START AUTO MODE',type='primary',width='stretch')
            if go:
                _write_auto_config(auto_area_name=name.strip() or 'My monitoring area',auto_latitude=lat,auto_longitude=lon,
                    auto_radius_km=radius,auto_location_ready=True,auto_location_source='manual',auto_collect_enabled=True,
                    ops_automation_enabled=True,ops_weather_enabled=True,ops_auto_report_enabled=True,geo_x_enabled=True,geo_x_highres_enabled=True,
                    ops_report_interval_minutes=5,ops_weather_interval_minutes=5,auto_interval_minutes=30,auto_lookback_days=30,
                    auto_max_cloud=70.0,multi_scenes_per_satellite=2,multi_concurrent_workers=3,multi_grid_pixels=512)
                store.setting('paused',False);store.setting('multisatellite_request',uuid.uuid4().hex);store.setting('ops_request',uuid.uuid4().hex)
                st.success('Auto Mode started. Reports will auto-save locally every 5 minutes.')
                st.rerun()
        return

    auto_on=cfg.get('auto_collect_enabled') and cfg.get('ops_automation_enabled') and not bool(store.setting('paused'))
    latest_report=store.setting('ops_latest_report') or {}
    awareness=store.setting('aircraft_awareness_latest') or {}
    top=st.columns([1.1,1,1,1,1,1])
    top[0].metric('AUTO MODE','RUNNING' if auto_on else 'PAUSED / OFF')
    top[1].metric('Satellite sources',len(cfg.get('multi_satellites',[])))
    scores=assessment.get('scores') or {}
    top[2].metric('Research reliability','—' if not scores else f"{scores.get('reliability',0):.0%}")
    top[3].metric('Aviation risk',(geo_x.get('operational_aviation_risk') or {}).get('status') or (geo_x.get('predictive_early_warning') or {}).get('status','—'))
    top[4].metric('Aircraft awareness',awareness.get('state','NO OBSERVATIONS'))
    top[5].metric('Last auto report',_short_time(latest_report.get('report_generated_at')))

    a,b,c=st.columns([1.2,1,1])
    if a.button('Run everything now',type='primary',width='stretch',disabled=not auto_on):
        store.setting('multisatellite_request',uuid.uuid4().hex);store.setting('ops_request',uuid.uuid4().hex)
        st.success('Requested: concurrent satellite refresh + weather refresh + automated resilience analysis.')
    if b.button('Pause automation' if auto_on else 'Resume automation',width='stretch'):
        if auto_on:store.setting('paused',True)
        else:
            store.setting('paused',False);_start_auto_mode(cfg)
        st.rerun()
    if c.button('Refresh this screen',width='stretch'):st.rerun()

    st.subheader('What AEROSENTINEL is doing')
    stages=[
        ('1 · Satellite search',sat_status.get('state','STARTING')),
        ('2 · Parallel processing',f"peak {sat_status.get('peak_workers',0)} workers"),
        ('3 · Weather context',ops_status.get('state','STARTING')),
        ('4 · Fusion + screening','READY' if assessment else 'WAITING'),
        ('5 · AEROSENTINEL',ops_status.get('aerosentinel_status') or ops_status.get('geo_x_status','WAITING')),
        ('6 · Aircraft awareness',awareness.get('state','NO OBSERVATIONS')),
    ]
    cols=st.columns(6)
    for col,(label,state) in zip(cols,stages):
        col.markdown(f'**{label}**');col.caption(str(state).replace('_',' '))
    st.caption('The three satellite sources are searched and processed concurrently. Their actual acquisition times are preserved; v2.8.0 never pretends they observed the area at the same instant.')

    _aircraft_summary()

    if not assessment:
        st.info('Auto Mode is running. The first complete screen appears after weather context and at least one satellite run are available.')
        if sat_status.get('error'):st.warning('Satellite service: '+str(sat_status['error']))
        if ops_status.get('weather_error'):st.warning('Weather service: '+str(ops_status['weather_error']))
        return

    levels=assessment.get('levels') or {}
    st.subheader('Simple decision-support view')
    cols=st.columns(4)
    cols[0].metric('Airfield environment',levels.get('airfield_environment','—'),f"screen {scores.get('airfield_environment_screening',0):.2f}")
    cols[1].metric('Disaster response',levels.get('disaster_response','—'),f"screen {scores.get('disaster_response_screening',0):.2f}")
    cols[2].metric('Infrastructure continuity',levels.get('infrastructure_continuity','—'),f"screen {scores.get('infrastructure_continuity_screening',0):.2f}")
    cols[3].metric('Uncertainty',f"{(assessment.get('drivers') or {}).get('fusion_uncertainty',1):.0%}")
    st.caption('These are research screening indicators—not flight clearance, certified pavement condition, or calibrated hazard probabilities.')
    context=assessment.get('response_context') or {}
    st.markdown('**Bangladesh-relevant resilience context**')
    bx=st.columns(3)
    bx[0].metric('Monsoon / flood context',context.get('monsoon_flood','—'),f"screen {scores.get('monsoon_flood_screening',0):.2f}")
    bx[1].metric('High-wind / storm context',context.get('high_wind_storm','—'),f"screen {scores.get('high_wind_storm_screening',0):.2f}")
    bx[2].metric('HADR access challenge',context.get('humanitarian_access_challenge','—'),f"screen {scores.get('humanitarian_access_screening',0):.2f}")

    left,right=st.columns([1.45,1])
    with left:
        preview=(assessment.get('satellite_summary') or {}).get('fusion_preview')
        if preview and Path(preview).exists():st.image(preview,caption='Latest fused satellite screen · red=uncertainty, green=wetness, blue=coverage',width='stretch')
        else:st.info('The fusion map will appear after a satellite run completes.')
    with right:
        w=assessment.get('weather_summary') or {}
        st.markdown('**Weather context (public, non-certified)**')
        wc1,wc2=st.columns(2)
        wc1.metric('Rain next 24 h',f"{w.get('rain_next_24h_mm',0):.1f} mm")
        wc2.metric('Max gust next 6 h','—' if w.get('max_gust_next_6h_kmh') is None else f"{w.get('max_gust_next_6h_kmh'):.0f} km/h")
        wc1.metric('Min visibility next 6 h','—' if w.get('min_visibility_next_6h_m') is None else f"{w.get('min_visibility_next_6h_m')/1000:.1f} km")
        wc2.metric('Thunderstorm hours / 24 h',int(w.get('thunderstorm_hours_next_24h',0)))
        t=assessment.get('terrain_summary') or {}
        if t:
            st.markdown('**Terrain / drainage context**')
            tc1,tc2=st.columns(2)
            tc1.metric('Mean terrain slope',f"{t.get('mean_slope_deg',0):.1f}°")
            tc2.metric('Relief (P05–P95)',f"{t.get('relief_p05_p95_m',0):.0f} m")
            st.caption('Copernicus DEM is a surface model, not surveyed runway/obstacle data.')
        st.caption('Always verify official aviation weather/METAR/TAF and local observations before operational use.')

    st.subheader('Alerts and recommended checks')
    alerts=assessment.get('alerts') or []
    st.dataframe(pd.DataFrame([{'Level':x.get('level'),'Area':x.get('category'),'Why':x.get('message'),'Check':x.get('suggested_check')} for x in alerts]),hide_index=True,width='stretch')

    asset_rows=assessment.get('asset_screening') or []
    if asset_rows:
        st.subheader('Registered asset inspection priority')
        st.dataframe(pd.DataFrame(asset_rows),hide_index=True,width='stretch')
        st.caption('Priority combines AOI-wide hazard screening with your own research criticality. It is not a vulnerability or targeting assessment.')

    latest_report=store.setting('ops_latest_report') or {}
    report_dir=ROOT/'data/reports/auto'
    st.subheader('Automatic 5-minute reports')
    if latest_report:
        st.caption(f"Latest report saved: {_short_time(latest_report.get('report_generated_at'))} · Next export approximately every {int(latest_report.get('interval_minutes',5))} min.")
    else:
        st.info('The first report will be auto-saved after the first automated screen.')
    d1,d2,d3=st.columns([1,1,1])
    if latest_report.get('latest_html') and Path(latest_report['latest_html']).exists():
        d1.download_button('Download latest HTML report',Path(latest_report['latest_html']).read_bytes(),file_name='AEROSENTINEL_v2_2_0_latest_report.html',mime='text/html',width='stretch')
    if latest_report.get('latest_json') and Path(latest_report['latest_json']).exists():
        d2.download_button('Download latest JSON report',Path(latest_report['latest_json']).read_bytes(),file_name='AEROSENTINEL_v2_2_0_latest_report.json',mime='application/json',width='stretch')
    if d3.button('Open auto-report folder',width='stretch'):
        report_dir.mkdir(parents=True,exist_ok=True)
        try:
            if os.name=='nt': os.startfile(str(report_dir))
            else: st.info(str(report_dir))
        except OSError as exc: st.warning(str(exc))
    st.caption('Browsers block repeated unsolicited downloads, so the background worker writes timestamped reports directly to this local folder every 5 minutes. The dashboard download buttons always point to the newest copy.')


def early_warning():
    cfg=load_config(ROOT)
    gx=store.setting('aerosentinel_latest') or store.setting('geo_x_latest') or {}
    st.subheader('AEROSENTINEL · Aviation Safety Intelligence')
    st.caption('Unified multimodal spatio-temporal research for airfield change detection, hazard prediction, uncertainty-aware operational risk screening and human decision support. It is not flight clearance or a certified aviation safety system.')
    if not cfg.get('geo_x_enabled',True):
        st.warning('AEROSENTINEL is disabled in Setup & Auto Mode.')
        return
    if not gx:
        st.info('Auto Mode has not produced an AEROSENTINEL assessment yet. Keep the background services running or press Run everything now in Operations Center.')
        return

    ew=gx.get('predictive_early_warning') or {}
    unc=gx.get('uncertainty_engine') or {}
    anom=gx.get('anomaly_path') or {}
    temp=gx.get('temporal_reasoning') or {}
    aviation=gx.get('operational_aviation_risk') or {}
    rare=gx.get('rare_event_detection') or {}
    cross=gx.get('cross_sensor_generalization') or {}
    deploy=gx.get('deployment_readiness') or {}
    avsafe=gx.get('aviation_safety_assessment') or {}

    cols=st.columns(7)
    cols[0].metric('Status',aviation.get('status') or ew.get('status','—'))
    cols[1].metric('Operational risk',f"{float(aviation.get('risk_score',ew.get('score',0)))*100:.0f}/100")
    cols[2].metric('Airfield change',f"{float(avsafe.get('airfield_change_detection',0))*100:.0f}/100")
    cols[3].metric('Rare-event',f"{float(rare.get('score',0))*100:.0f}/100")
    cols[4].metric('Reliability',f"{float(unc.get('reliability',0)):.0%}")
    cols[5].metric('Uncertainty',f"{float(unc.get('uncertainty',0)):.0%}")
    cols[6].metric('Deployment',f"{float(deploy.get('deployment_readiness_score',0)):.0%}")

    if unc.get('abstain') or str(aviation.get('status','')).startswith('ABSTAIN'):
        st.error('ABSTAIN: evidence quality/reliability is insufficient for automated interpretation. Obtain additional or better-aligned evidence and perform human review.')
    elif aviation.get('status') in {'ELEVATED','HIGH REVIEW'}:
        st.warning('Elevated aviation-safety research screen. Review airfield, weather, historical change, rare-event and data-quality evidence before acting.')
    else:
        st.info('Continue monitoring. AEROSENTINEL provides research screening and decision support—not certified operational authority.')

    tabs=st.tabs(['Evidence & foundation model','Temporal & hazard prediction','Aviation risk & explainability','Rare event & cross-sensor','Direct candidate path','Trust / validation','Deployment','Human review'])
    t1,t2,t3,t4,t5,t6,t7,t8=tabs

    with t1:
        left,right=st.columns(2)
        with left:
            st.markdown('### Observed evidence')
            st.json(gx.get('observed_evidence') or {},expanded=True)
            st.caption('Measured or transparently derived evidence from optical/SAR imagery, terrain, weather and historical observations.')
        with right:
            st.markdown('### Interpretation / hypothesis')
            st.json(gx.get('interpretation_hypothesis') or {},expanded=True)
            st.caption('Interpretation is kept separate from observations and must be reviewed by a human.')
        rep=gx.get('representation') or {}
        st.markdown('### Satellite foundation-model / self-supervised representation')
        r1,r2,r3,r4=st.columns(4)
        r1.metric('Backend',rep.get('backend','—'))
        r2.metric('Historical rows',int(rep.get('history_rows',0)))
        r3.metric('TerraMind','ACTIVE' if rep.get('foundation') else 'OPTIONAL / NOT MATCHED')
        r4.metric('Sources',len((gx.get('cross_sensor_generalization') or {}).get('available_sensors') or []))
        if rep.get('foundation'):st.json(rep['foundation'],expanded=False)
        st.caption('TerraMind is recorded only when a valid timestamp-matched S1GRD + S2L2A + DEM feature record exists. Otherwise AEROSENTINEL clearly labels the transparent unlabelled baseline.')
        with st.expander('Evidence provenance manifest',expanded=False):
            st.json(gx.get('provenance') or {},expanded=False)
            st.caption('Observed scenes, derived context and learned embeddings remain explicitly distinguishable; generated data must not be treated as independent sensor confirmation.')

        try:
            import pydeck as pdk
            lat=float(cfg['auto_latitude']);lon=float(cfg['auto_longitude'])
            status=aviation.get('status') or ew.get('status','NORMAL')
            colors={'NORMAL':[45,180,105,180],'MONITOR':[230,190,45,190],'ELEVATED':[240,130,30,200],'HIGH REVIEW':[220,60,60,210],'ABSTAIN / HUMAN REVIEW':[130,130,130,190]}
            data=[{'latitude':lat,'longitude':lon,'status':status,'score':float(aviation.get('risk_score',ew.get('score',0)))}]
            deck=pdk.Deck(layers=[pdk.Layer('ScatterplotLayer',data=data,get_position='[longitude, latitude]',get_radius=max(300,float(cfg.get('auto_radius_km',5))*450),get_fill_color=colors.get(status,[130,130,130,190]),pickable=True)],initial_view_state=pdk.ViewState(latitude=lat,longitude=lon,zoom=10),tooltip={'text':'AOI status: {status}\nAviation risk screen: {score}'})
            st.pydeck_chart(deck,use_container_width=True)
            st.caption('AOI-level decision-support map only; it does not provide target designation, flight clearance or runway certification.')
        except Exception:
            pass

    with t2:
        traj=pd.DataFrame(anom.get('trajectory') or [])
        if not traj.empty:
            st.markdown('### Spatio-temporal anomaly history')
            st.line_chart(traj.set_index('timestamp')[['anomaly_score','ood_score']])
            st.dataframe(traj.tail(30),hide_index=True,width='stretch')
        else:st.info('Historical trajectory appears after repeated automated observations accumulate.')
        forecast=gx.get('hazard_prediction') or {}
        frows=forecast.get('forecast_next_3_observations') or []
        if frows:
            st.markdown('### Hazard forecast — next three observation steps')
            fdf=pd.DataFrame(frows)
            st.line_chart(fdf.set_index('step')[['screening','lower','upper']])
            st.dataframe(fdf,hide_index=True,width='stretch')
        st.json({k:temp.get(k) for k in ['persistence','consecutive_anomalies','repeated_anomalies','trend','trend_slope','current_season','same_season_history_rows','seasonal_baseline']},expanded=False)
        st.caption('Forecasts are uncertainty-aware short-horizon screening projections, not certified weather forecasts or guaranteed operational outcomes.')

    with t3:
        st.markdown('### Operational aviation-risk screen')
        domains=aviation.get('domains') or {}
        if domains:
            ddf=pd.DataFrame([{'domain':k.replace('_',' ').title(),'score':v} for k,v in domains.items()])
            st.bar_chart(ddf.set_index('domain'))
            st.dataframe(ddf,hide_index=True,width='stretch')
        st.markdown('### Explainable decision support')
        exp=gx.get('explainability') or aviation.get('explainability') or {}
        drivers=pd.DataFrame(exp.get('top_risk_drivers') or [])
        if not drivers.empty:
            st.dataframe(drivers,hide_index=True,width='stretch')
        cf=exp.get('counterfactual_source_contribution') or {}
        if cf:
            st.markdown('#### Counterfactual source contribution')
            st.dataframe(pd.DataFrame([{'source':k,'relative_contribution':v} for k,v in cf.items()]),hide_index=True,width='stretch')
            st.caption('Leave-one-source-out sensitivity: how much the fused screening surface changes when each available source is removed. It is not causal attribution.')
        st.json({'boundary':aviation.get('boundary'),'recommendation':aviation.get('recommendation')},expanded=False)
        st.caption('The score is decomposed into visible risk domains and weighted contributions. It is a research screen, not a hidden black-box operational command.')

    with t4:
        left,right=st.columns(2)
        with left:
            st.markdown('### Rare-event detection')
            st.metric('State',rare.get('state','—'))
            st.metric('Rare-event score',f"{float(rare.get('score',0)):.0%}")
            st.json(rare,expanded=False)
            st.caption('Rare-event means statistically unusual relative to the learned local baseline; it does not determine cause or severity.')
        with right:
            st.markdown('### Cross-sensor generalization')
            st.metric('State',cross.get('state','—'))
            st.metric('Readiness',f"{float(cross.get('generalization_readiness_score',0)):.0%}")
            st.json(cross,expanded=False)
            st.caption('True cross-sensor generalization must be demonstrated with independent labelled regions, seasons and acquisition conditions.')

    with t5:
        direct=gx.get('direct_path') or {}
        st.metric('Direct small-object path',direct.get('status','—'))
        st.write(direct.get('interpretation',''))
        st.json(direct.get('standard_satellite_feasibility') or {},expanded=False)
        hi=direct.get('high_resolution') or {}
        mask=hi.get('segmentation_mask_path')
        if mask and Path(mask).exists():
            st.image(mask,caption='Image-space candidate segmentation mask (salient candidate pixels; not identity or operational classification)',width='stretch')
        st.markdown('#### Add fine-resolution imagery when legally/authoritatively available')
        hs=store.setting('geo_x_highres_state') or {}
        hcols=st.columns(4)
        hcols[0].metric('Input GeoTIFFs',int(hs.get('input_file_count',0) or 0))
        hcols[1].metric('Processed this scan',int(hs.get('processed_this_scan',0) or 0))
        hcols[2].metric('Logged results',len(hs.get('history') or []))
        hcols[3].metric('Automatic commercial download','NO')
        st.caption('Drop GeoTIFFs into `app/highres_inbox` or upload one here. Source GSD is checked before candidate screening.')
        up=st.file_uploader('High-resolution GeoTIFF',type=['tif','tiff'],key='gx_highres')
        gsd=st.number_input('Known GSD (m/pixel, 0 = read/estimate from GeoTIFF)',0.0,50.0,0.0,0.1,key='gx_gsd')
        if st.button('Analyze high-resolution research image',disabled=up is None):
            folder=ROOT/str(cfg.get('geo_x_highres_inbox','highres_inbox'));folder.mkdir(parents=True,exist_ok=True)
            path=folder/(uuid.uuid4().hex+Path(up.name).suffix.lower());path.write_bytes(up.getvalue())
            result=analyze_highres(path,gsd_m=(gsd or None))
            store.setting('geo_x_highres_state',{'latest':result,'seen_sha256':[result['sha256']],'inbox':str(folder)})
            store.setting('ops_request',uuid.uuid4().hex)
            st.success('Candidate screening complete. AEROSENTINEL will merge the evidence into the next assessment.')
            st.json(result,expanded=False)
        st.markdown('#### Resolution feasibility research matrix')
        st.dataframe(pd.DataFrame(resolution_research_table()),hide_index=True,width='stretch')

    with t6:
        st.markdown('### Registration & reliability-aware fusion')
        dq=gx.get('data_quality_and_fusion') or {}
        st.json({k:dq.get(k) for k in ['registration_quality','fusion_source_reliability','effective_source_count','source_reliability_weights','counterfactual_source_contribution']},expanded=False)
        st.caption('Residual registration confidence and source weights are quality gates, not calibrated probabilities of correctness.')
        st.markdown('### Reliability stress suite')
        stress=pd.DataFrame(gx.get('reliability_stress_suite') or [])
        if not stress.empty:st.dataframe(stress,hide_index=True,width='stretch')
        st.markdown('### OOD / calibration / abstention')
        u1,u2,u3,u4=st.columns(4)
        u1.metric('OOD score',f"{float(unc.get('ood_score',0)):.2f}")
        u2.metric('Reliability',f"{float(unc.get('reliability',0)):.0%}")
        u3.metric('Calibration',unc.get('calibration','—'))
        u4.metric('Decision','ABSTAIN' if unc.get('abstain') else 'REPORT')
        st.markdown('#### Optional calibration from independent labelled validation data')
        cal=st.file_uploader('Calibration CSV (`score,label`)',type=['csv'],key='gx_calibration')
        if st.button('Fit temperature calibration',disabled=cal is None):
            df=pd.read_csv(cal)
            if not {'score','label'}.issubset(df.columns):st.error('CSV needs score,label columns.')
            else:
                model=fit_temperature(df['score'].tolist(),df['label'].astype(int).tolist())
                store.setting('geo_x_calibration',model);store.setting('ops_request',uuid.uuid4().hex)
                st.success('Calibration saved. The next automatic run will apply it.');st.json(model)
        st.markdown('### Multi-region / season / domain-shift evaluation')
        st.download_button('Download validation CSV template',domain_evaluation_template_csv(),file_name='aerosentinel_domain_validation_template.csv',mime='text/csv')
        domain_file=st.file_uploader('Independent labelled validation cases',type=['csv'],key='gx_domain_eval')
        if domain_file is not None and st.button('Run domain reliability evaluation'):
            try:
                result=domain_reliability_evaluation(domain_file and pd.read_csv(domain_file).to_dict('records'),abstain_threshold=float(cfg.get('geo_x_abstain_threshold',0.45)))
                st.session_state['gx_domain_result']=result
            except Exception as exc:st.error(str(exc))
        if st.session_state.get('gx_domain_result'):
            dr=st.session_state['gx_domain_result'];st.json(dr['overall'],expanded=True);st.dataframe(pd.DataFrame(dr['groups']),hide_index=True,width='stretch')

    with t7:
        st.markdown('### Deployment constraints and real-time readiness')
        d=deploy
        c1,c2,c3,c4=st.columns(4)
        c1.metric('Assessment latency',f"{float(d.get('assessment_compute_latency_ms',0)):.0f} ms")
        c2.metric('Satellite pipeline',f"{float(d.get('satellite_pipeline_elapsed_seconds') or 0):.1f} s")
        c3.metric('Local analytical data',f"{float(d.get('local_analytical_output_megabytes',0)):.1f} MB")
        c4.metric('Readiness',f"{float(d.get('deployment_readiness_score',0)):.0%}")
        st.json(d,expanded=True)
        st.markdown('### Compute-adaptive edge/ground routing')
        st.json(gx.get('adaptive_compute_policy') or {},expanded=True)
        st.caption('This build recommends a processing tier (archive, ROI analysis, ground foundation model, or human/reacquisition path); it does not claim autonomous onboard execution.')
        st.caption('AEROSENTINEL reports measured local processing latency, concurrency and data footprint. Satellite revisit remains externally constrained; aircraft awareness is limited to actual satellite acquisition times.')

    with t8:
        st.markdown('### Human analyst / operator review')
        records=research.records('aerosentinel_assessment') or research.records('geo_x_assessment')
        if not records:
            st.info('No stored AEROSENTINEL assessment record yet.')
        else:
            latest=records[0]
            st.caption(f"Assessment record: {latest['id']} · {latest['created']}")
            with st.form('gx_human_review'):
                reviewer=st.text_input('Reviewer name / identifier')
                decision=st.selectbox('Review outcome',['Needs follow-up','Reviewed','Rejected'])
                note=st.text_area('Human review note',placeholder='Compare historical imagery, local/official aviation evidence, weather, field observations, alternate explanations, uncertainty and deployment limitations.')
                submit=st.form_submit_button('Save human review',type='primary')
            if submit:
                research.review(latest['id'],reviewer,decision,note);st.success('Human review saved to the research audit trail.')
        st.warning('Final interpretation belongs to an authorised human reviewer. AEROSENTINEL does not issue flight clearance, dispatch authority, autonomous safety-critical actions, or certified runway/meteorological decisions.')

def setup_auto_mode():
    cfg=load_config(ROOT)
    st.subheader('Setup & Auto Mode')
    st.write('Recommended use: save one authorised aviation study area, keep all three satellite sources selected, and let Sentinal X 1.0.0 handle collection, multimodal fusion, change/anomaly analysis, hazard forecasting, reliability checks and briefing generation automatically.')
    profiles={'Airfield resilience':'airfield_resilience','Disaster response / HADR':'disaster_response','Infrastructure continuity':'infrastructure_resilience','Balanced':'balanced'}
    rev={v:k for k,v in profiles.items()}
    with st.form('v5_automation_settings'):
        enabled=st.checkbox('Enable fully automated mode',bool(cfg.get('auto_collect_enabled') and cfg.get('ops_automation_enabled')))
        gx_enabled=st.checkbox('Enable AEROSENTINEL early-warning research layer',bool(cfg.get('geo_x_enabled',True)))
        gx_abstain=st.slider('AEROSENTINEL abstain below reliability',0.10,0.90,float(cfg.get('geo_x_abstain_threshold',0.45)),0.05)
        current_profile=rev.get(cfg.get('ops_profile'),'Airfield resilience')
        profile_label=st.selectbox('Primary dashboard emphasis',list(profiles),index=list(profiles).index(current_profile) if current_profile in profiles else 0)
        name=st.text_input('Area name',cfg['auto_area_name'])
        x,y,z=st.columns(3)
        lat=x.number_input('Latitude',-79.0,83.0,float(cfg['auto_latitude']),format='%.5f')
        lon=y.number_input('Longitude',-179.8,179.8,float(cfg['auto_longitude']),format='%.5f')
        radius=z.number_input('Half-width (km)',0.5,20.0,float(cfg['auto_radius_km']),0.5)
        st.markdown('**Automation frequency**')
        f1,f2,f3,f4=st.columns(4)
        sat_minutes=f1.number_input('Satellite catalog refresh (min)',30,1440,int(max(30,cfg.get('auto_interval_minutes',30))),30)
        weather_minutes=f2.number_input('Weather refresh (min)',5,180,int(cfg.get('ops_weather_interval_minutes',5)),5)
        report_minutes=f3.number_input('Auto-report interval (min)',1,60,int(cfg.get('ops_report_interval_minutes',5)),1)
        scenes=f4.number_input('Scenes / satellite',1,3,int(cfg.get('multi_scenes_per_satellite',2)),1)
        labels={'Sentinel-1 SAR':'sentinel-1','Sentinel-2 Optical':'sentinel-2','Landsat 8/9 Optical + Thermal (TIRS)':'landsat-8-9'}
        defaults=[k for k,v in labels.items() if v in cfg.get('multi_satellites',[])]
        selected=st.multiselect('Satellites processed simultaneously',list(labels),default=defaults)
        q1,q2,q3=st.columns(3)
        workers=q1.slider('Parallel workers',2,8,int(cfg.get('multi_concurrent_workers',3)))
        grid=q2.select_slider('Analysis grid',options=[256,384,512,768,1024],value=int(cfg.get('multi_grid_pixels',512)))
        max_cloud=q3.slider('Max optical cloud %',0,100,int(cfg.get('auto_max_cloud',70)))
        st.caption('Recommended: 3 workers, 512 grid, 2 scenes/satellite. More workers/grids use more RAM, network and CPU.')
        save=st.form_submit_button('Save settings & apply',type='primary',width='stretch')
    if save:
        sats=[labels[x] for x in selected]
        if len(sats)<2:st.error('Choose at least two satellite sources.')
        else:
            _write_auto_config(auto_collect_enabled=enabled,ops_automation_enabled=enabled,ops_weather_enabled=True,geo_x_enabled=gx_enabled,geo_x_highres_enabled=True,geo_x_abstain_threshold=float(gx_abstain),
                ops_profile=profiles[profile_label],ops_auto_report_enabled=True,ops_report_interval_minutes=int(report_minutes),auto_area_name=name.strip() or 'My monitoring area',auto_latitude=lat,auto_longitude=lon,
                auto_radius_km=radius,auto_location_ready=True,auto_location_source='manual',auto_interval_minutes=int(sat_minutes),
                ops_weather_interval_minutes=int(weather_minutes),multi_scenes_per_satellite=int(scenes),multi_satellites=sats,
                multi_concurrent_workers=int(workers),multi_grid_pixels=int(grid),auto_max_cloud=float(max_cloud))
            store.setting('paused',False if enabled else bool(store.setting('paused')))
            if enabled:
                store.setting('multisatellite_request',uuid.uuid4().hex);store.setting('ops_request',uuid.uuid4().hex)
            st.success('Saved. Background services pick up the new settings automatically—no restart needed.')
            st.rerun()

    st.subheader('What Sentinal X 1.0.0 automates')
    st.markdown("""- **Concurrent EO:** Sentinel-1, Sentinel-2 and Landsat search/processing in parallel; Landsat Level-2 thermal surface temperature is extracted when available.
- **Common grid fusion:** all available satellite evidence is aligned before fusion.
- **Weather context:** rainfall, wind/gust, visibility, cloud, convection and heat context refresh automatically.
- **Terrain context:** Copernicus DEM GLO-30 is downloaded/cached automatically for slope and relief screening.
- **Airfield-environment screening:** surface-water/drainage + weather + change + uncertainty.
- **Disaster/HADR screening:** flood/rain/wind context for rescue, relief and access planning.
- **Infrastructure continuity:** registered runway/taxiway/apron, drainage, access, power, communications and relief-staging assets can be prioritized for inspection.
- **Reliability audit:** missing sources, temporal mismatch and uncertainty lower confidence instead of being hidden.
- **5-minute reports:** timestamped HTML + JSON reports auto-save to `app/data/reports/auto` on the configured cadence, even when the browser is closed.
- **AEROSENTINEL intelligence:** satellite foundation/self-supervised representation, airfield change detection, open-world anomaly and rare-event screening, temporal persistence/forecasting, cross-sensor robustness, OOD/uncertainty, explainability and operational aviation-risk screening.
- **Aircraft awareness:** imported georeferenced satellite detector outputs are screened for presence and possible change between acquisitions.
- **Briefings:** new analytical assessments are distinguished from report export time, so repeated reports never pretend old satellite imagery is newly acquired.""")
    st.info('Sentinal X 1.0.0 is a research decision-support system. Official aviation weather, field inspection, engineering data and authorised operational procedures remain authoritative.')


def processing_lab():
    st.subheader('Data & Image Processing Lab')
    st.write('The processing stack improves visualization robustness, reduces noise sensitivity and expands physically interpretable features. It does not perform synthetic super-resolution or invent detail beyond the source sensor.')
    rows=[
        {'Module':'Optical enhancement','Automatic':'Yes','Purpose':'Robust percentile stretch, CLAHE local contrast and mild visualization sharpening','Output':'preview_enhanced.png'},
        {'Module':'Spectral index engine','Automatic':'Yes','Purpose':'NDVI, NDWI, MNDWI, NDMI, SAVI, NBR, NBR2 and SWIR1-preferred NDBI when bands exist','Output':'analysis_stack.tif'},
        {'Module':'SAR despeckling','Automatic':'Yes','Purpose':'Adaptive Lee-style filtering on log-amplitude to reduce speckle sensitivity','Output':'filtered SAR bands + enhanced preview'},
        {'Module':'Texture extraction','Automatic':'Yes','Purpose':'Gradient-magnitude texture screening for optical and SAR scenes','Output':'relative texture bands'},
        {'Module':'Quality control','Automatic':'Yes','Purpose':'Valid coverage, dynamic range, entropy, clipping and sharpness/readability checks','Output':'processing_quality metadata'},
        {'Module':'Multi-signal change','Automatic':'Yes','Purpose':'Combines water, vegetation, built-surface and heat change when common signals exist','Output':'temporal change fusion + per-signal summaries'},
        {'Module':'Thermal uncertainty processing','Automatic':'Yes','Purpose':'Uses Landsat ST_QA, emissivity, cloud distance and robust anomaly statistics when available','Output':'uncertainty-aware thermal bands + quality metadata'},
        {'Module':'Aircraft observation fusion','Automatic':'On satellite detector import','Purpose':'Resolution-gated optical/SAR/thermal/foundation candidate fusion and sparse temporal displacement screening','Output':'candidate observations and possible movement for human review'},
    ]
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')

    last=store.setting('multisatellite_last_result') or {}
    scenes=last.get('scene_results') or []
    if scenes:
        qrows=[]
        for scene in scenes:
            q=scene.get('processing_quality') or (scene.get('analysis') or {}).get('processing_quality') or {}
            qrows.append({
                'Satellite':scene.get('label'),'Scene':scene.get('scene_id'),'Acquired':scene.get('acquired_at'),
                'Valid %':scene.get('valid_percent'),'Readiness score':q.get('processing_readiness_score'),
                'Entropy bits':q.get('entropy_bits'),'Clipped fraction':q.get('clipped_fraction'),
            })
        st.markdown('### Latest automatic scene quality')
        st.dataframe(pd.DataFrame(qrows),hide_index=True,width='stretch')
        chosen=st.selectbox('Inspect processed satellite scene',scenes,format_func=lambda s:f"{s.get('label')} · {s.get('scene_id')}",key='processing_scene')
        c1,c2=st.columns(2)
        raw=Path(str(chosen.get('preview_path') or ''))
        enhanced=Path(str(chosen.get('enhanced_preview_path') or ''))
        with c1:
            st.markdown('**Standard preview**')
            if raw.exists():st.image(str(raw),width='stretch')
            else:st.info('Standard preview is not available in this stored run.')
        with c2:
            st.markdown('**Enhanced preview**')
            if enhanced.exists():st.image(str(enhanced),width='stretch')
            else:st.info('Enhanced preview will appear after a v2.8.0 multisatellite run.')
        st.json((chosen.get('analysis') or {}).get('processing_modules') or [],expanded=False)
    else:
        st.info('Run the multisatellite worker to populate automatic quality metrics and enhanced scene previews.')

    st.markdown('### Local image enhancement sandbox')
    st.caption('Use this only for authorised PNG/JPEG research images. The result is a contrast/noise-enhanced visualization, not a higher-resolution measurement.')
    upload=st.file_uploader('PNG/JPEG image',type=['png','jpg','jpeg'],key='processing_lab_upload')
    if upload is not None:
        try:
            original=np.asarray(Image.open(upload).convert('RGB'))
            denoised=denoise_rgb(original,strength=5)
            valid=np.ones(original.shape[:2],dtype=bool)
            enhanced=enhance_rgb(denoised[:,:,0],denoised[:,:,1],denoised[:,:,2],valid)
            before=image_quality_metrics(original,valid)
            after=image_quality_metrics(enhanced,valid)
            c1,c2=st.columns(2)
            with c1:st.image(original,caption='Original',width='stretch')
            with c2:st.image(enhanced,caption='Denoised + contrast-enhanced',width='stretch')
            st.dataframe(pd.DataFrame([
                {'Stage':'Original',**before},{'Stage':'Enhanced preview',**after}
            ]),hide_index=True,width='stretch')
        except (OSError,ValueError) as exc:
            st.error(str(exc))


def advanced_analysis():
    st.info('Advanced pages are optional. You do not need them for normal v2.8.0 Auto Mode.')
    tabs=st.tabs(['Before/after','Multimodal twin','Scenario','Foundation models','Evaluation'])
    with tabs[0]:compare()
    with tabs[1]:multimodal_twin()
    with tabs[2]:scenario()
    with tabs[3]:foundation()
    with tabs[4]:evaluate()

try:
    {'Aircraft Map':aircraft_map_page,'Operations Center':operations_center,'Sentinal X Intelligence':early_warning,'Aircraft Awareness':aircraft_awareness,'Research & Validation':sentinal_research,'Setup & Auto Mode':setup_auto_mode,'Satellite Detail':satellite,
     'Processing Lab':processing_lab,'Advanced Analysis':advanced_analysis,'System Health':roadmap}[page]()
except (ValueError,KeyError,OSError,RuntimeError) as exc:
    st.error(str(exc))
st.divider();st.caption('Sentinal X 1.0.0 · Satellite-only aircraft awareness research. Outputs support human review and are not certified aviation weather, flight clearance, runway certification, dispatch authority, or calibrated accident probabilities.')

"""Concurrent multi-satellite controls shared by AEROSENTINEL pages."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import uuid

import pandas as pd
import streamlit as st

from collection_store import atomic_json, load_config, validate_config
from satellite_downloader import area_bbox
from multisatellite_pipeline import SATELLITES


def utc(value):
    return datetime.fromtimestamp(value, timezone.utc).strftime('%Y-%m-%d %H:%M UTC') if value else 'Not yet'


@st.fragment(run_every='5s')
def download_status(store, root, key='download'):
    cfg = load_config(root)
    status = store.setting('multisatellite_download') or {}
    state = status.get('state', 'STARTING')
    st.subheader('Concurrent multi-satellite collection + analysis')
    st.caption('Selected satellite sources are searched in parallel. Their AOI imagery is streamed/cropped, '
               'preprocessed and analyzed by independent workers at the same time; cross-sensor fusion follows '
               'the completed workers. Satellite acquisition times themselves may differ and are reported.')
    if not cfg['auto_collect_enabled']:
        st.info('Automatic concurrent satellite processing is disabled. Enable it below.')
    elif not cfg['auto_location_ready']:
        st.info('Choose your current location above, or save coordinates below, to start multi-satellite processing.')
    elif store.setting('paused'):
        st.warning('Collection is paused. Resume from the sidebar or AEROSENTINEL Control Center.')
    elif state == 'ERROR':
        st.error(status.get('message') or 'Concurrent satellite processing failed.')
        if status.get('error'):
            st.code(status['error'], language=None, wrap_lines=True)
    elif state == 'PARTIAL_SUCCESS':
        st.warning(status.get('message') or 'The run completed with partial source availability.')
        if status.get('error'):
            st.caption(status['error'])
    elif state in {'SEARCHING', 'CONCURRENT_PROCESSING'}:
        st.info(status.get('message') or ('Searching satellite catalogs concurrently…' if state == 'SEARCHING' else
                                         'Downloading and analyzing satellite scenes concurrently…'))
    elif state == 'UP_TO_DATE':
        st.success(status.get('message') or 'Concurrent multi-satellite run complete.')
    elif state in {'PAUSED', 'DISABLED', 'LOCATION_REQUIRED', 'STOPPED'}:
        st.info(status.get('message') or state.replace('_', ' ').title())
    else:
        st.info('Starting the concurrent multi-satellite worker.')

    sources = status.get('sources') or {}
    if sources:
        rows = []
        for sat in cfg.get('multi_satellites', []):
            value = sources.get(sat, {})
            rows.append({
                'Satellite': SATELLITES.get(sat).label if sat in SATELLITES else sat,
                'Stage': value.get('stage', 'QUEUED'),
                'Matches': value.get('matches'),
                'Asset': value.get('asset'),
                'Scene': value.get('scene'),
                'Error': value.get('error'),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    a, b, c, d = st.columns(4)
    a.metric('Selected sources', len(cfg.get('multi_satellites', [])))
    b.metric('Peak scene workers', status.get('peak_workers', 0))
    c.metric('Analyzed scenes', status.get('processed_scenes', 0))
    skew = status.get('acquisition_skew_hours')
    d.metric('Acquisition skew', '—' if skew is None else f'{skew:.1f} h')
    if cfg['auto_location_ready']:
        st.caption(f"Area: {cfg['auto_area_name']} · center {cfg['auto_latitude']:.5f}, {cfg['auto_longitude']:.5f} · "
                   f"approximately {cfg['auto_radius_km'] * 2:g} × {cfg['auto_radius_km'] * 2:g} km")
    st.caption(f"Last check: {utc(status.get('last_check'))} · Last completed run: {utc(status.get('last_download'))} · "
               f"Next check: {utc(status.get('next_check'))}")

    preview = status.get('fusion_preview')
    if preview and Path(preview).is_file():
        st.image(preview, caption='Latest multi-satellite fusion preview · red=uncertainty, green=wetness screening, blue=source coverage', width='stretch')
    available = status.get('available_satellites')
    if available:
        st.caption('Fused sources: ' + ', '.join(SATELLITES.get(v).label if v in SATELLITES else v for v in available))

    last = store.setting('multisatellite_last_result')
    if last:
        with st.expander('Latest concurrent analysis details'):
            rows=[]
            for scene in last.get('scene_results', []):
                analysis=scene.get('analysis') or {}
                rows.append({
                    'Satellite': scene.get('label'), 'Platform': scene.get('platform'),
                    'Acquired': scene.get('acquired_at'), 'Valid %': round(scene.get('valid_percent',0),1),
                    'Cloud %': scene.get('cloud_cover_percent'),
                    'NDVI mean': (analysis.get('ndvi') or {}).get('mean') if isinstance(analysis.get('ndvi'),dict) else None,
                    'Surface temp °C': ((analysis.get('surface_temperature_celsius') or {}).get('mean')
                                        if isinstance(analysis.get('surface_temperature_celsius'),dict) else None),
                    'Heat screening mean': ((analysis.get('relative_heat_screening') or {}).get('mean')
                                            if isinstance(analysis.get('relative_heat_screening'),dict) else None),
                    'Water screening mean': ((analysis.get('water_screening') or {}).get('mean')
                                             if isinstance(analysis.get('water_screening'),dict) else
                                             (analysis.get('low_backscatter_screening') or {}).get('mean')
                                             if isinstance(analysis.get('low_backscatter_screening'),dict) else None),
                })
            if rows:
                st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            fusion=last.get('fusion') or {}
            st.json({k:v for k,v in fusion.items() if k not in {'fusion_path','preview_path'}},expanded=False)
            st.download_button('Download latest concurrent run JSON',
                               json.dumps(last,indent=2),
                               file_name='trust_geo_concurrent_multisatellite_run.json',
                               key=key+'_run_json')

    if (cfg['auto_collect_enabled'] and status.get('time') and
            (state == 'STOPPED' or time.time() - status['time'] > 180)):
        st.warning('The concurrent satellite worker has stopped or has not reported recently. Check Roadmap & health or restart RUN_ALL.bat.')
    busy = state in {'SEARCHING', 'CONCURRENT_PROCESSING'} and time.time() - status.get('time', 0) < 300
    if st.button('Run concurrent multi-satellite download + analysis now', key=key+'_now', type='primary',
                 disabled=not cfg['auto_collect_enabled'] or not cfg['auto_location_ready'] or bool(store.setting('paused')) or busy):
        store.setting('multisatellite_request', uuid.uuid4().hex)
        st.success('Concurrent run requested. The background service will start the selected satellite workers within a few seconds.')


def collection_settings(store, root):
    cfg = load_config(root)
    st.write('Configure one AOI and choose at least two satellite sources. The same AOI grid is used for every source so their analysis layers can be fused pixel-for-pixel.')
    label_to_key = {spec.label: key for key, spec in SATELLITES.items()}
    current_labels = [SATELLITES[k].label for k in cfg.get('multi_satellites', []) if k in SATELLITES]
    with st.form('automatic_satellite_settings'):
        enabled = st.checkbox('Enable automatic multi-satellite runs', cfg['auto_collect_enabled'])
        selected_labels = st.multiselect('Satellite sources processed concurrently', list(label_to_key), default=current_labels,
            help='Choose two or three. Sentinel-1 adds SAR; Sentinel-2 adds optical observations; Landsat 8/9 adds optical plus Level-2 thermal surface temperature when available.')
        name = st.text_input('Collection area name', cfg['auto_area_name'])
        a, b = st.columns(2)
        with a:
            lat = st.number_input('Collection center latitude', -79.0, 83.0, float(cfg['auto_latitude']), format='%.5f')
            lon = st.number_input('Collection center longitude', -179.8, 179.8, float(cfg['auto_longitude']), format='%.5f')
            radius = st.number_input('Half-width of collection area (km)', 0.5, 20.0, float(cfg['auto_radius_km']), 0.5)
            location_ready = st.checkbox('Use the entered coordinates as my collection area', cfg['auto_location_ready'])
        with b:
            interval = st.number_input('Automatic run every (minutes)', 1, 1440, int(cfg['auto_interval_minutes']), 1)
            lookback = st.number_input('Search the last (days)', 1, 365, int(cfg['auto_lookback_days']), 1)
            cloud = st.slider('Maximum optical scene cloud cover (%)', 0.0, 100.0, float(cfg['auto_max_cloud']), 1.0)
            scenes = st.number_input('Newest scenes per satellite', 1, 3, int(cfg.get('multi_scenes_per_satellite', 1)), 1,
                help='Use 2 or 3 when you want within-satellite temporal-change screening as well as the latest fusion.')
            workers = st.slider('Concurrent scene workers', 2, 8, int(cfg.get('multi_concurrent_workers', 3)), 1,
                help='Independent satellite scenes are streamed, preprocessed and analyzed in parallel. Higher values use more network, RAM and CPU.')
            grid = st.select_slider('Common analysis grid (maximum pixels)', options=[256, 384, 512, 768, 1024],
                value=int(cfg.get('multi_grid_pixels', 512)),
                help='All satellite layers are reprojected onto this shared local UTM grid. Larger grids preserve more detail but use more memory/network I/O.')
            tolerance = st.number_input('Temporal alignment tolerance (hours)', 1.0, 720.0,
                float(cfg.get('multi_temporal_tolerance_hours', 72.0)), 1.0,
                help='Fusion uncertainty increases when the latest observations from different satellites are farther apart in time.')
        st.caption('Concurrent means the computer processes independent satellite sources at the same time. It does not mean the satellites acquired the area at exactly the same moment. AEROSENTINEL reports the actual acquisition-time skew.')
        saved = st.form_submit_button('Save concurrent collection settings', type='primary')
    if saved:
        try:
            selected = [label_to_key[v] for v in selected_labels]
            updated = dict(load_config(root), auto_collect_enabled=enabled, auto_area_name=name.strip(),
                           auto_latitude=lat, auto_longitude=lon, auto_radius_km=radius,
                           auto_interval_minutes=interval, auto_lookback_days=lookback,
                           auto_max_cloud=cloud, auto_location_ready=location_ready,
                           multi_satellites=selected, multi_scenes_per_satellite=int(scenes),
                           multi_concurrent_workers=int(workers), multi_grid_pixels=int(grid),
                           multi_temporal_tolerance_hours=float(tolerance))
            if location_ready and (not cfg['auto_location_ready'] or lat != cfg['auto_latitude'] or lon != cfg['auto_longitude']):
                updated.update(auto_location_source='manual', auto_location_accuracy_m=None, auto_location_captured_at=None)
            if location_ready and updated['auto_area_name'] == 'Choose your location':
                updated['auto_area_name'] = 'My selected area'
            validate_config(updated)
            area_bbox(updated)
            atomic_json(root/'collection_config.json', updated)
            st.success('Saved. The concurrent worker applies these settings without restarting AEROSENTINEL.')
            st.rerun()
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    with st.expander('Collection area center on map'):
        if cfg['auto_location_ready']:
            st.map(pd.DataFrame([{'lat': cfg['auto_latitude'], 'lon': cfg['auto_longitude']}]))
    st.info('Optical analysis uses red/green/blue/NIR, optional SWIR and pixel QA where available. Landsat 8/9 additionally reads the Level-2 TIRS/LWIR surface-temperature asset when available and stores Kelvin, Celsius and relative heat-screening bands. Sentinel-1 GRD contributes a relative low-backscatter screening layer; it is not automatically presented as calibrated sigma0/gamma0.')
    st.link_button('Planetary Computer STAC documentation', 'https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/')

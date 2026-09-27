"""Observation map for the latest satellite aircraft assessment."""
from datetime import datetime, timezone
import hashlib
import json
import math


def candidate_rows(assessment):
    """Build stable, deduplicated map rows; never promote insufficient evidence."""
    found = {}
    rejected = 0
    for item in assessment.get('observations', []):
        if item.get('presence') != 'CANDIDATE':
            continue
        try:
            lat, lon = float(item['latitude']), float(item['longitude'])
            score = float(item['screening_score'])
            stamp = datetime.fromisoformat(item['acquired_at'].replace('Z', '+00:00'))
            if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(score)
                    and -90 <= lat <= 90 and -180 <= lon <= 180 and 0 <= score <= 1
                    and stamp.tzinfo is not None):
                raise ValueError('Invalid mapped observation')
            stamp = stamp.astimezone(timezone.utc).isoformat()
            site, scene = str(item['site_id']), str(item['scene_id'])
            models = sorted(str(m) for m in item.get('model_ids', []))
            identity = json.dumps([site, scene, stamp, lat, lon, models], ensure_ascii=True)
            ident = 'OBS-' + hashlib.sha256(identity.encode()).hexdigest()[:12].upper()
            row = dict(item, candidate_id=ident, latitude=lat, longitude=lon,
                       screening_score=score, acquired_at=stamp, site_id=site, scene_id=scene,
                       model_ids=models, source=assessment.get('source', 'Satellite assessment'),
                       data_origin=assessment.get('data_origin', 'Detector / imported evidence'))
            found[ident] = row
        except (KeyError, TypeError, ValueError, OverflowError):
            rejected += 1
    return sorted(found.values(), key=lambda r: (r['acquired_at'], r['candidate_id']), reverse=True), rejected


def filter_rows(rows, sites, minimum_score=0.5, latest_only=True):
    # Find latest capture before threshold filtering: never substitute an older
    # stronger detection for a newer, lower-scored observation.
    chosen = [r for r in rows if r['site_id'] in sites]
    latest = {}
    for row in chosen:
        latest[row['site_id']] = max(latest.get(row['site_id'], ''), row['acquired_at'])
    return [r for r in chosen if r['screening_score'] >= minimum_score
            and (not latest_only or r['acquired_at'] == latest[r['site_id']])]


def map_view(rows):
    if not rows:
        return dict(latitude=0, longitude=0, zoom=1)
    radians = [math.radians(r['longitude']) for r in rows]
    lon = math.degrees(math.atan2(sum(math.sin(x) for x in radians), sum(math.cos(x) for x in radians)))
    lat = sum(r['latitude'] for r in rows) / len(rows)
    spread = max(max(r['latitude'] for r in rows) - min(r['latitude'] for r in rows),
                 2 * max(abs((r['longitude'] - lon + 180) % 360 - 180) for r in rows))
    return dict(latitude=max(-85, min(85, lat)), longitude=lon,
                zoom=max(1, min(16, math.log2(180 / max(spread, 0.002)))))


def synthetic_assessment():
    return dict(source='Synthetic preview', data_origin='SYNTHETIC — NOT REAL DETECTIONS',
                observations=[dict(site_id='DEMO-SITE', scene_id='DEMO-SCENE',
                    acquired_at='2026-01-01T12:00:00+00:00', latitude=lat, longitude=lon,
                    presence='CANDIDATE', screening_score=score, model_confidence=score,
                    modalities=['optical'], model_ids=['synthetic-example'], gsd_m=0.5,
                    native_gsd_m=None, resolution_status='SYNTHETIC', validation_status='SYNTHETIC',
                    registration_error_m=None, evidence_count=1)
                    for lat, lon, score in [(0.001, 0.002, 0.91), (0.003, 0.005, 0.78), (0.0, 0.006, 0.62)]])


def render(store):
    import streamlit as st

    st.header('Aircraft Map')
    st.caption('Satellite observation locations · Select an aircraft marker to inspect the evidence.')
    st.info('Positions belong to the image capture time. Satellite imagery alone does not establish identity or transponder status. Use Flight comparison to inspect time-matched flight records.')
    demo = st.toggle('Preview synthetic example', key='aircraft_map_demo', value=False)
    if demo:
        st.warning('SYNTHETIC PREVIEW — these invented points near 0°, 0° are not real aircraft detections. Nothing is saved to your evidence.')
    # Fresh database read on each fragment refresh includes new worker results.
    assessment = synthetic_assessment() if demo else (store.setting('aircraft_awareness_latest') or {})
    rows, rejected = candidate_rows(assessment)
    status = store.setting('highres_collection_status') or {}
    if not demo:
        st.caption('Collector: ' + str(status.get('state', 'NOT STARTED')).replace('_', ' ')
                   + ' · Latest analysis source: ' + str(assessment.get('source', 'Not yet available')))
        if assessment.get('analyzed_at'):
            st.caption('Analysis updated: ' + str(assessment['analyzed_at']))
    if rejected:
        st.warning(f'{rejected} candidate observations have invalid location/time data and cannot be mapped.')
    if not rows:
        st.metric('Candidate observations', 0)
        st.info('No aircraft candidates to display. Use Aircraft Awareness to analyze a GeoTIFF, import detector CSV, or configure automatic collection. No candidates does not establish absence.')
        st.button('Refresh observations', key='aircraft_map_empty_refresh')
        return
    controls = st.columns([2, 2, 2, 1])
    sites = controls[0].multiselect('Sites', sorted({r['site_id'] for r in rows}),
                                    default=sorted({r['site_id'] for r in rows}), key='aircraft_map_sites')
    scope = controls[1].selectbox('Capture history', ['Latest candidate capture per site', 'All candidate captures'], key='aircraft_map_scope')
    minimum = controls[2].slider('Minimum screening score', 0.0, 1.0, 0.5, 0.05, key='aircraft_map_score')
    controls[3].button('Refresh', key='aircraft_map_refresh')
    visible = filter_rows(rows, sites, minimum, scope == 'Latest candidate capture per site')
    metrics = st.columns(3)
    metrics[0].metric('Candidate observations', len(visible))
    metrics[1].metric('Image acquisitions', len({(r['site_id'], r['scene_id'], r['acquired_at']) for r in visible}))
    latest = max((r['acquired_at'] for r in visible), default='')
    metrics[2].metric('Latest capture (UTC)', latest[:16].replace('T', ' ') if latest else '—')
    st.caption('Refreshes every 10 seconds while open. Screening scores are uncalibrated evidence scores. Different captures may contain the same aircraft; counts are observations, not unique aircraft.')
    if not visible:
        st.info('No candidates match these filters. Lower the score threshold, select a site, or view all captures.')
        return
    _map_and_details(visible, demo)


def _map_and_details(rows, demo):
    import pandas as pd
    import pydeck as pdk
    import streamlit as st

    identifiers = {r['candidate_id']: r for r in rows}
    selected_key = 'aircraft_map_selected'
    if st.session_state.get(selected_key) not in identifiers:
        st.session_state[selected_key] = rows[0]['candidate_id']
    points = [dict(r, marker='✈', score_label=f"{r['screening_score']:.2f}",
                   color=[104, 214, 192] if r['screening_score'] >= 0.8 else [255, 195, 95]) for r in rows]
    map_column, detail_column = st.columns([3, 1.35])
    with map_column:
        chart = pdk.Deck(map_provider='carto', map_style=pdk.map_styles.DARK,
            initial_view_state=pdk.ViewState(**map_view(rows)),
            layers=[pdk.Layer('TextLayer', points, id='aircraft-observations',
                get_position=['longitude', 'latitude'], get_text='marker', get_color='color',
                get_size=30, pickable=True, auto_highlight=True)],
            tooltip={'text': '{candidate_id}\n{site_id} / {scene_id}\nCaptured: {acquired_at}\nScreening score: {score_label}'})
        event = st.pydeck_chart(chart, height=540, use_container_width=True, key='aircraft_observation_map',
                                on_select='rerun', selection_mode='single-object')
        objects = event.get('selection', {}).get('objects', {}).get('aircraft-observations', [])
        clicked = objects[0].get('candidate_id') if objects else None
        # Resolve browser selection against current data, not browser-supplied details.
        if clicked in identifiers and clicked != st.session_state.get('aircraft_map_last_click'):
            st.session_state[selected_key] = clicked
        st.session_state['aircraft_map_last_click'] = clicked
        st.caption('Teal: screening score ≥ 0.80 · Amber: lower score. Marker orientation is decorative. Map tiles require internet; the observation table remains available below.')
    with detail_column:
        selected = st.selectbox('Selected observation', list(identifiers), key=selected_key)
        row = identifiers[selected]
        st.subheader('Aircraft candidate')
        overview, evidence, flight_data = st.tabs(['Detection', 'Evidence', 'Flight comparison'])
        with overview:
            st.code(selected, language=None)
            st.write('**Image capture (UTC):**', row['acquired_at'])
            st.write('**Location:**', f"{row['latitude']:.6f}, {row['longitude']:.6f}")
            st.write('**Site:**', row['site_id'])
            st.write('**Scene:**', row['scene_id'])
            st.metric('Screening score', f"{row['screening_score']:.2f}")
            st.caption('Imagery-only identity / speed / altitude / transponder status: Unknown. See Flight comparison for possible reported flight associations.')
        with evidence:
            confidence = row.get('model_confidence')
            st.write('**Model confidence:**', f'{confidence:.2f} (uncalibrated)' if isinstance(confidence, (int, float)) else 'Not recorded in this assessment')
            st.write('**Source:**', row['source'])
            st.write('**Sensors:**', ', '.join(row.get('modalities', [])) or 'Unknown')
            st.write('**Model:**', ', '.join(row['model_ids']) or 'Unknown')
            st.write('**Product pixel spacing:**', str(row.get('gsd_m', 'Unknown')) + ' m')
            st.write('**Native resolution:**', str(row.get('native_gsd_m')) + ' m' if row.get('native_gsd_m') is not None else 'Unverified')
            st.write('**Local validation:**', row.get('validation_status', 'Unknown'))
            st.write('**Registration error:**', str(row.get('registration_error_m')) + ' m' if row.get('registration_error_m') is not None else 'Unknown')
            st.caption('Coordinates are imagery-derived estimates; resolution and registration limit their accuracy.')
        with flight_data:
            from flight_comparison_ui import render as render_flight_comparison
            render_flight_comparison(row, demo)
        st.download_button('Download selected observation', json.dumps(row, indent=2),
                           file_name=('synthetic_' if demo else '') + selected + '.json', mime='application/json')
    with st.expander('Observation table and export', expanded=True):
        frame = pd.DataFrame(rows)
        fields = ['candidate_id', 'site_id', 'scene_id', 'acquired_at', 'latitude', 'longitude',
                  'screening_score', 'source', 'data_origin']
        st.dataframe(frame[fields], hide_index=True, width='stretch')
        st.download_button('Download visible observations (CSV)', frame[fields].to_csv(index=False),
                           file_name='synthetic_observations.csv' if demo else 'aircraft_observations.csv', mime='text/csv')

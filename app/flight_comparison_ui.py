"""Session-only flight comparison panel; API calls occur only on explicit clicks."""
import os
import time

from flight_comparison import (CSV_TEMPLATE, DISCLAIMER, FlightDataError, compare,
                               fetch_historical, normalize_records, parse_upload)

SESSION_TTL = 1800


def render(candidate, demo=False):
    import streamlit as st
    st.caption('Compare the satellite capture time and location with flight-position records. A match is a possible association, not confirmed identity.')
    radius = st.number_input('Matching radius (metres)', 50, 5000, 500, 50, key='flight_match_radius')
    tolerance = st.number_input('Time tolerance (seconds)', 1, 120, 15, 1, key='flight_match_time')
    cache = st.session_state.setdefault('flight_comparisons', {})
    now = time.time()
    for old in list(cache):
        if now - cache[old]['compared_at'] > SESSION_TTL:
            del cache[old]
    cache_key = ('demo:' if demo else 'real:') + candidate['candidate_id']
    if demo:
        st.info('Synthetic comparison only; no FlightRadar24 request or real flight data.')
        if st.button('Compare synthetic flight record', key='flight_compare_demo'):
            records = normalize_records([dict(fr24_id='DEMO-FLIGHT', timestamp=candidate['acquired_at'],
                lat=candidate['latitude'], lon=candidate['longitude'], callsign='DEMO001',
                reg='SYNTHETIC', source='ADSB', alt=0, gspeed=0, track=0, squawk='0000')])
            cache[cache_key] = compare(candidate, records, radius, tolerance, provider='SYNTHETIC EXAMPLE')
    else:
        mode = st.radio('Flight data source', ['FlightRadar24 API', 'Upload snapshot'], key='flight_data_source')
        if mode == 'FlightRadar24 API':
            token = os.environ.get('FR24_API_TOKEN', '').strip()
            if not token:
                st.info('Set FR24_API_TOKEN in the server environment. Historical access requires an eligible FlightRadar24 API subscription.')
            st.caption('One historical query at this image capture time, limited to 100 returned records. Uses your API credits and sends the selected location/time to FlightRadar24. No automatic API polling.')
            if st.button('Fetch historical flight records & compare', disabled=not token, key='flight_compare_api'):
                cache.pop(cache_key, None)
                try:
                    with st.spinner('Querying the image capture time…'):
                        snapshot = fetch_historical(candidate, token, radius)
                        cache[cache_key] = compare(candidate, snapshot['records'], radius, tolerance,
                            provider=snapshot['provider'], incomplete=snapshot['incomplete'])
                except FlightDataError as exc:
                    st.error(str(exc))
        else:
            st.download_button('Download flight CSV template', CSV_TEMPLATE, 'flight_snapshot_template.csv', 'text/csv')
            uploaded = st.file_uploader('Flight-position JSON or CSV', type=['json', 'csv'], key='flight_snapshot')
            st.caption('Use records you are entitled to use. JSON accepts the official API data array; CSV uses the template. Maximum 5 MB / 5,000 positions.')
            if st.button('Compare uploaded snapshot', disabled=uploaded is None, key='flight_compare_upload'):
                cache.pop(cache_key, None)
                try:
                    records = parse_upload(uploaded.getvalue(), uploaded.name)
                    cache[cache_key] = compare(candidate, records, radius, tolerance, provider='User-uploaded snapshot (provenance unverified)')
                except FlightDataError as exc:
                    st.error(str(exc))
    report = cache.get(cache_key)
    if report and (report['radius_m'] != radius or report['tolerance_s'] != tolerance):
        st.info('Matching settings changed. Run the comparison again to apply them.')
        report = None
    if report:
        st.write('**Flight comparison:**', report['status'])
        st.write('**Transponder assessment:**', report['transponder_assessment'])
        st.caption(f"{report['provider']} · {report['records_checked']} positions checked · capture {report['capture_time']}")
        if report['incomplete']:
            st.warning('The response reached its result cap; more flights may exist. This comparison is incomplete.')
        if report['match_count'] > 1:
            st.warning('Several flight records fit. No single flight identity is assigned.')
        for match in report['matches']:
            with st.expander(match['callsign'] or match['flight'] or match['fr24_id'], expanded=report['match_count'] == 1):
                st.write('**Reported flight / callsign:**', (match['flight'] or 'Unknown') + ' / ' + (match['callsign'] or 'Unknown'))
                st.write('**Reported registration:**', match['reg'] or 'Unknown')
                st.write('**ICAO hex / squawk:**', (match['hex'] or 'Unknown') + ' / ' + (match['squawk'] or 'Unknown'))
                st.write('**Position source:**', match['source'] or 'Unknown')
                st.write('**Record time:**', match['timestamp'])
                st.write('**Separation:**', f"{match['distance_m']:.1f} m · {match['time_offset_s']:.1f} seconds")
                st.write('**Reported altitude (ft) / ground speed (kt) / track (°):**',
                         ' / '.join(str(match[k]) if match[k] is not None else 'Unknown' for k in ['alt', 'gspeed', 'track']))
    else:
        st.caption('Not compared with these settings. Transponder state: UNDETERMINED.')
    st.caption(DISCLAIMER)
    st.caption('Flight results stay in this browser session for up to 30 minutes; they are not written to the evidence database or included in map exports.')
    if cache and st.button('Clear flight comparison results', key='flight_compare_clear'):
        cache.clear()
        st.rerun(scope='fragment')

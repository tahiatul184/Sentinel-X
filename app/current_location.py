"""Use a browser-authorized location fix for a fixed satellite collection area."""
import math
from pathlib import Path
import time
import uuid

import streamlit as st
import streamlit.components.v1 as components

from collection_store import atomic_json, load_config, validate_config
from satellite_downloader import area_bbox

_device_location = components.declare_component(
    'satellite_device_location', path=str(Path(__file__).parent/'location_component'))


def apply_device_location(root, store, result, *, now=None):
    now = time.time() if now is None else now
    if not isinstance(result, dict) or result.get('status') != 'ok':
        raise ValueError('No successful browser location fix was received.')
    lat, lon, accuracy, captured = (float(result[k]) for k in
                                    ('latitude', 'longitude', 'accuracy_m', 'timestamp_ms'))
    captured /= 1000
    if not all(math.isfinite(x) for x in [lat, lon, accuracy, captured]) or accuracy < 0:
        raise ValueError('Your browser returned an invalid location. Please try again.')
    if not -60 <= now-captured <= 300:
        raise ValueError('The location fix is out of date. Press Use my current location again.')
    cfg = dict(load_config(root), auto_latitude=lat, auto_longitude=lon,
               auto_area_name='Current device location', auto_location_ready=True,
               auto_location_source='browser', auto_location_accuracy_m=accuracy,
               auto_location_captured_at=captured, auto_collect_enabled=True,
               ops_automation_enabled=True, ops_weather_enabled=True, ops_auto_report_enabled=True,
               ops_weather_interval_minutes=5, ops_report_interval_minutes=5)
    validate_config(cfg)
    area_bbox(cfg)
    atomic_json(root/'collection_config.json', cfg)
    store.setting('multisatellite_request', uuid.uuid4().hex)
    store.setting('ops_request', uuid.uuid4().hex)
    store.setting('paused', False)
    return cfg


def location_picker(store, root):
    cfg = load_config(root)
    st.subheader('Your collection location')
    if not cfg['auto_location_ready']:
        st.info('First launch will request browser/device location permission. No coordinate is guessed or preloaded.')
    else:
        st.caption(f"Saved area: {cfg['auto_area_name']} · {cfg['auto_latitude']:.5f}, {cfg['auto_longitude']:.5f}")
        if cfg.get('auto_location_source') == 'browser':
            accuracy = cfg.get('auto_location_accuracy_m')
            if accuracy is not None:
                st.caption(f'Browser-reported location accuracy: ±{accuracy:,.0f} m. Check the area on the map in Collect images.')
                if accuracy > float(cfg['auto_radius_km'])*1000:
                    st.warning('Location uncertainty is wider than your selected area. Check or adjust the coordinates in Collect images.')
    result = _device_location(key='satellite_device_location', default=None, auto_request=not cfg['auto_location_ready'])
    if isinstance(result, dict) and result.get('event_id') != st.session_state.get('location_event_handled'):
        st.session_state['location_event_handled'] = result.get('event_id')
        if result.get('status') == 'error':
            st.warning(result.get('message', 'Location lookup failed. Enter coordinates in Collect images.'))
        else:
            try:
                apply_device_location(root, store, result)
                st.rerun()
            except (KeyError, TypeError, ValueError, OSError) as exc:
                st.error(str(exc))
    st.caption('The saved area stays fixed if you move. Use the button again to update it. Precise coordinates are stored locally, but are sent to the configured weather and satellite providers when those services are queried. '
               'You can also enter latitude and longitude in Collect images.')

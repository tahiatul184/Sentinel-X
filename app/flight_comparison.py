"""Time/location correlation with flight records, never an off-transponder verdict."""
import csv
from datetime import datetime, timezone
import io
import json
import math
import time

API_URL = 'https://fr24api.flightradar24.com/api/historic/flight-positions/full'
CSV_TEMPLATE = 'fr24_id,timestamp,lat,lon,callsign,flight,reg,hex,source,alt,gspeed,track,squawk\n'
LIMIT = 100
MAX_BYTES = 5_000_000
DISCLAIMER = ('A nearby flight record is only a possible association, not confirmed aircraft identity. '
              'No matching record does not prove a transponder is off: coverage gaps, filtering, '
              'ground operations, timestamp errors and image geolocation errors can cause a mismatch.')


class FlightDataError(ValueError):
    pass


def timestamp(value):
    try:
        if isinstance(value, bool):
            raise ValueError()
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
            result = datetime.fromtimestamp(float(value), timezone.utc)
        else:
            result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if result.tzinfo is None:
            raise ValueError()
        return result.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError, OSError):
        raise FlightDataError('Flight timestamps must be UNIX seconds or ISO-8601 with a timezone.') from None


def number(value, low, high, field):
    try:
        result = float(value)
        if not math.isfinite(result) or not low <= result <= high:
            raise ValueError()
        return result
    except (TypeError, ValueError, OverflowError):
        raise FlightDataError(f'Invalid {field}.') from None


def normalize_records(payload):
    raw = payload.get('data') if isinstance(payload, dict) else payload
    if not isinstance(raw, list) or len(raw) > 5000:
        raise FlightDataError('Supply a JSON data array or CSV with at most 5,000 positions.')
    rows = []
    for row in raw:
        if not isinstance(row, dict):
            raise FlightDataError('Each flight position must be an object.')
        ident = str(row.get('fr24_id') or row.get('flight_id') or '').strip()
        if not ident or len(ident) > 120:
            raise FlightDataError('Each flight position needs a fr24_id (or flight_id).')
        normalized = dict(fr24_id=ident, timestamp=timestamp(row.get('timestamp')).isoformat(),
            latitude=number(row.get('lat', row.get('latitude')), -90, 90, 'latitude'),
            longitude=number(row.get('lon', row.get('longitude')), -180, 180, 'longitude'))
        for key in ['callsign', 'flight', 'reg', 'hex', 'source', 'squawk', 'type']:
            normalized[key] = str(row.get(key) or '').strip()[:120]
        for key, bounds in [('alt', (-2000, 100000)), ('gspeed', (0, 2500)), ('track', (0, 360))]:
            value = row.get(key)
            normalized[key] = None if value in (None, '') else number(value, *bounds, key)
        rows.append(normalized)
    return rows


def parse_upload(content, name):
    if len(content) > MAX_BYTES:
        raise FlightDataError('Flight snapshot must be at most 5 MB.')
    try:
        text = content.decode('utf-8-sig')
        if name.lower().endswith('.json'):
            payload = json.loads(text)
        else:
            reader = csv.DictReader(io.StringIO(text))
            fields = set(reader.fieldnames or [])
            if not ('timestamp' in fields and fields & {'fr24_id', 'flight_id'} and fields & {'lat', 'latitude'} and fields & {'lon', 'longitude'}):
                raise FlightDataError('CSV needs fr24_id, timestamp, lat and lon columns (or documented aliases).')
            payload = list(reader)
        return normalize_records(payload)
    except (UnicodeError, json.JSONDecodeError, csv.Error):
        raise FlightDataError('Could not read the flight JSON/CSV snapshot.') from None


def distance_m(a, b):
    lat1, lat2 = math.radians(a['latitude']), math.radians(b['latitude'])
    dlon = math.radians(b['longitude'] - a['longitude'])
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 12742000 * math.asin(min(1, math.sqrt(max(0, h))))


def compare(candidate, records, radius_m=500, tolerance_s=15, *, provider='Uploaded snapshot', incomplete=False):
    radius_m = number(radius_m, 50, 5000, 'matching radius')
    tolerance_s = number(tolerance_s, 1, 120, 'time tolerance')
    acquired = timestamp(candidate['acquired_at'])
    number(candidate['latitude'], -90, 90, 'candidate latitude')
    number(candidate['longitude'], -180, 180, 'candidate longitude')
    nearby = {}
    aligned = 0
    for record in records:
        dt = abs((timestamp(record['timestamp']) - acquired).total_seconds())
        if dt > tolerance_s:
            continue
        aligned += 1
        distance = distance_m(candidate, record)
        if distance > radius_m:
            continue
        match = dict(record, distance_m=round(distance, 1), time_offset_s=round(dt, 1))
        old = nearby.get(record['fr24_id'])
        if old is None or (dt, distance) < (old['time_offset_s'], old['distance_m']):
            nearby[record['fr24_id']] = match
    matches = sorted(nearby.values(), key=lambda m: (m['time_offset_s'], m['distance_m'], m['fr24_id']))
    transmission = 'UNDETERMINED'
    if not records:
        status = 'NO RECORDS RETURNED'
    elif not aligned:
        status = 'NO TIME-ALIGNED RECORDS'
    elif not matches:
        status = 'NO MATCHING FLIGHT RECORD'
    elif len(matches) > 1:
        status = 'AMBIGUOUS MATCH'
    else:
        source = matches[0]['source'].upper()
        if source in {'ADSB', 'ADS-B', 'MLAT'}:
            status = 'POSSIBLE BROADCAST MATCH'
            transmission = 'Broadcast-derived record nearby; aircraft association unconfirmed'
        elif source in {'ESTIMATION', 'ESTIMATED'}:
            status = 'POSSIBLE ESTIMATED MATCH'
        else:
            status = 'POSSIBLE FLIGHT MATCH'
    return dict(status=status, transponder_assessment=transmission, candidate_id=candidate['candidate_id'],
                capture_time=acquired.isoformat(), provider=provider, matches=matches[:20],
                match_count=len(matches), records_checked=len(records), time_aligned_records=aligned,
                radius_m=radius_m, tolerance_s=tolerance_s, incomplete=bool(incomplete),
                compared_at=time.time(), disclaimer=DISCLAIMER)


def query_bounds(candidate, radius_m):
    lat = number(candidate['latitude'], -85, 85, 'query latitude (must be within ±85°)')
    lon = number(candidate['longitude'], -180, 180, 'query longitude')
    angle = number(radius_m, 50, 5000, 'matching radius') * 1.02 / 6371000
    dlat = math.degrees(angle)
    dlon = math.degrees(math.asin(math.sin(angle) / math.cos(math.radians(lat))))
    if lon - dlon < -180 or lon + dlon > 180:
        raise FlightDataError('This query crosses the date line. Use an uploaded flight snapshot instead.')
    return f'{lat+dlat:.7f},{lat-dlat:.7f},{lon-dlon:.7f},{lon+dlon:.7f}'


def fetch_historical(candidate, token, radius_m=500, get=None):
    if not token or not token.strip():
        raise FlightDataError('Set FR24_API_TOKEN in the server environment and restart the app.')
    import requests
    get = get or requests.get
    params = dict(timestamp=int(timestamp(candidate['acquired_at']).timestamp()),
                  bounds=query_bounds(candidate, radius_m), limit=LIMIT)
    try:
        response = get(API_URL, params=params, headers={'Accept': 'application/json',
            'Accept-Version': 'v1', 'Authorization': 'Bearer ' + token.strip()},
            timeout=(5, 25), allow_redirects=False)
    except requests.RequestException:
        raise FlightDataError('FlightRadar24 request failed. Check connectivity and try again.') from None
    try:
        if response.status_code != 200:
            messages = {401: 'Token is invalid.', 402: 'API credits are unavailable.',
                        403: 'The token or subscription cannot access this historical query.',
                        429: 'API rate limit reached; wait before retrying.'}
            raise FlightDataError('FlightRadar24 HTTP ' + str(response.status_code) + ': '
                                  + messages.get(response.status_code, 'Historical query was not completed.'))
        try:
            payload = response.json()
        except ValueError:
            raise FlightDataError('FlightRadar24 returned invalid JSON.') from None
        if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
            raise FlightDataError('FlightRadar24 returned an unexpected response schema.')
        records = normalize_records(payload)
        return dict(records=records, incomplete=len(payload['data']) >= LIMIT,
                    provider='FlightRadar24 historical API', queried_at=candidate['acquired_at'])
    finally:
        response.close()

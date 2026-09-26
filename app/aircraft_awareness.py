"""Satellite-only aircraft candidate fusion for human review.

Inputs are outputs from separately validated, georeferenced satellite models.
Broad scene statistics are never treated as aircraft detections.
"""
from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from datetime import datetime, timezone

MODALITIES = {'optical', 'sar', 'thermal', 'foundation'}
FIELDS = ('site_id', 'scene_id', 'acquired_at', 'modality', 'latitude', 'longitude',
          'confidence', 'gsd_m', 'object_length_m', 'registration_error_m',
          'model_id', 'quality', 'cloud_fraction', 'native_gsd_m', 'enhancement', 'validation_id')
TEMPLATE = ','.join(FIELDS) + '\n'


def _number(value, name, lo, hi, optional=False):
    if value is None or str(value).strip() == '':
        if optional:
            return None
        raise ValueError(f'{name} is required')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not math.isfinite(result) or not lo <= result <= hi:
        raise ValueError(f'{name} must be finite and between {lo} and {hi}')
    return result


def parse_detections(csv_text):
    """Validate detector outputs and reject non-satellite modalities."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames or not set(FIELDS[:9]).issubset(reader.fieldnames):
        raise ValueError('CSV requires site_id, scene_id, acquired_at, modality, latitude, longitude, confidence, gsd_m and object_length_m')
    rows = []
    for line, raw in enumerate(reader, 2):
        site = str(raw.get('site_id') or '').strip()
        scene = str(raw.get('scene_id') or '').strip()
        model = str(raw.get('model_id') or '').strip()
        modality = str(raw.get('modality') or '').strip().lower()
        if not site or not scene or not model:
            raise ValueError(f'Line {line}: site_id, scene_id and model_id are required')
        if modality not in MODALITIES:
            raise ValueError(f'Line {line}: unsupported satellite modality {modality!r}')
        try:
            timestamp = datetime.fromisoformat(str(raw.get('acquired_at') or '').replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError(f'Line {line}: invalid acquisition time') from exc
        if timestamp.tzinfo is None:
            raise ValueError(f'Line {line}: acquisition time requires timezone')
        rows.append(dict(site_id=site, scene_id=scene, acquired_at=timestamp.astimezone(timezone.utc).isoformat(),
            modality=modality, model_id=model,
            native_gsd_m=_number(raw.get('native_gsd_m'), 'native_gsd_m', 0.01, 1000, True),
            enhancement=str(raw.get('enhancement') or 'unknown').strip(),
            validation_id=str(raw.get('validation_id') or '').strip(),
            latitude=_number(raw.get('latitude'), 'latitude', -90, 90),
            longitude=_number(raw.get('longitude'), 'longitude', -180, 180),
            confidence=_number(raw.get('confidence'), 'confidence', 0, 1),
            gsd_m=_number(raw.get('gsd_m'), 'gsd_m', 0.01, 1000),
            object_length_m=_number(raw.get('object_length_m'), 'object_length_m', 1, 150),
            registration_error_m=_number(raw.get('registration_error_m'), 'registration_error_m', 0, 10000, True),
            quality=_number(raw.get('quality'), 'quality', 0, 1, True),
            cloud_fraction=_number(raw.get('cloud_fraction'), 'cloud_fraction', 0, 1, True)))
        if len(rows) > 10000:
            raise ValueError('Maximum 10,000 detections per upload')
    if not rows:
        raise ValueError('No detections supplied; empty input is unknown, not absence')
    return rows


def _distance(a, b):
    lat1, lat2 = math.radians(a['latitude']), math.radians(b['latitude'])
    dlat, dlon = lat2-lat1, math.radians(b['longitude']-a['longitude'])
    h = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 12742000 * math.asin(min(1, math.sqrt(max(0, h))))


def analyze(rows, max_gap_hours=72):
    """Fuse coincident candidates and screen sparse displacement between acquisitions.

    Modalities from the same scene may be correlated. Their confidence is not
    multiplied or interpreted as a calibrated probability.
    """
    if not rows:
        return {'state': 'NO OBSERVATIONS', 'observations': [], 'movement_candidates': [], 'anomalies': []}
    if not math.isfinite(max_gap_hours) or max_gap_hours <= 0:
        raise ValueError('max_gap_hours must be positive and finite')
    normalized = []
    for row in rows:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerow(row)
        normalized.extend(parse_detections(output.getvalue()))
    # Exact duplicate imports never add corroboration.
    normalized = list({tuple(r.get(k) for k in FIELDS): r for r in normalized}.values())
    groups = defaultdict(list)
    for row in normalized:
        groups[(row['site_id'], row['scene_id'], row['acquired_at'])].append(row)
    observations = []
    for (site, scene, stamp), candidates in sorted(groups.items(), key=lambda pair: pair[0][2]):
        clusters = []
        for row in sorted(candidates, key=lambda r: r['confidence'], reverse=True):
            match = next((c for c in clusters if _distance(row, c[0]) <=
                         min(10, max(2, 2 * min(row['gsd_m'], c[0]['gsd_m'])))
                         and all(r['model_id'] != row['model_id'] for r in c)), None)
            if match is None:
                clusters.append([row])
            else:
                match.append(row)
        for cluster in clusters:
            direct = [r for r in cluster if r['modality'] in {'optical', 'sar'}]
            def evidence_score(r):
                quality = r['quality'] if r['quality'] is not None else 0.5
                cloud = (r['cloud_fraction'] if r['cloud_fraction'] is not None else 0.5) if r['modality'] == 'optical' else 0
                return r['confidence'] * quality * (1 - cloud)
            resolved = [r for r in direct if r['object_length_m'] / max(r['gsd_m'], r['native_gsd_m'] or r['gsd_m']) >= 3]
            best = max(resolved or direct or cluster, key=evidence_score)
            effective_gsd = max(best['gsd_m'], best['native_gsd_m'] or best['gsd_m'])
            pixels = best['object_length_m'] / effective_gsd
            modalities = sorted({r['modality'] for r in cluster})
            score = evidence_score(best) if direct else 0.0
            observations.append(dict(site_id=best['site_id'], scene_id=scene, acquired_at=stamp,
                latitude=best['latitude'], longitude=best['longitude'],
                presence='CANDIDATE' if pixels >= 3 and score >= 0.5 and bool(set(modalities) & {'optical', 'sar'}) else 'INSUFFICIENT EVIDENCE',
                screening_score=round(score, 3), modalities=modalities,
                resolution_status='NATIVE DOCUMENTED' if best['native_gsd_m'] is not None else 'NATIVE RESOLUTION UNVERIFIED',
                native_gsd_m=best['native_gsd_m'], enhancement=best['enhancement'],
                validation_status='REFERENCE PROVIDED / VERIFY SCOPE' if best['validation_id'] else 'LOCAL VALIDATION MISSING',
                validation_id=best['validation_id'],
                model_ids=sorted({r['model_id'] for r in cluster}),
                pixel_span=round(pixels, 2), gsd_m=best['gsd_m'],
                registration_error_m=best['registration_error_m'], evidence_count=len(cluster),
                note='Model candidate for human review; score is not a calibrated probability.'))
    movement, anomalies = [], []
    for current in observations:
        if current['presence'] != 'CANDIDATE':
            continue
        t = datetime.fromisoformat(current['acquired_at'])
        history = [o for o in observations if o['site_id'] == current['site_id']
                   and datetime.fromisoformat(o['acquired_at']) < t]
        if not history:
            continue
        previous_time = max(o['acquired_at'] for o in history)
        elapsed = (t - datetime.fromisoformat(previous_time)).total_seconds()
        if elapsed > max_gap_hours * 3600:
            continue
        previous = sorted([(i, o, _distance(o, current)) for i, o in enumerate(observations)
                           if o['site_id'] == current['site_id'] and o['acquired_at'] == previous_time
                           and o['presence'] == 'CANDIDATE'], key=lambda x: x[2])
        if not previous or previous[0][2] > 2000:
            continue
        _, prior, distance = previous[0]
        peers = [o for o in observations if o['site_id'] == current['site_id']
                 and o['acquired_at'] == current['acquired_at'] and o['presence'] == 'CANDIDATE']
        reverse = sorted((_distance(prior, o), j) for j, o in enumerate(peers))
        margin = max(10, 2 * max(prior['gsd_m'], current['gsd_m']))
        ambiguous = (len(previous) > 1 and previous[1][2] - distance < margin) or (
            len(reverse) > 1 and reverse[1][0] - reverse[0][0] < margin)
        if peers[reverse[0][1]] is not current:
            continue
        errors = [prior['registration_error_m'], current['registration_error_m']]
        error = sum(errors) + prior['gsd_m'] + current['gsd_m'] if all(e is not None for e in errors) else None
        lower_bound = max(0.0, distance - error) if error is not None else None
        state = 'AMBIGUOUS ASSOCIATION' if ambiguous else ('REGISTRATION UNKNOWN' if error is None else
                ('POSSIBLE MOVEMENT' if lower_bound > 0 else 'UNRESOLVED'))
        item = dict(site_id=current['site_id'], from_scene_id=prior['scene_id'], to_scene_id=current['scene_id'],
            from_acquired_at=prior['acquired_at'], to_acquired_at=current['acquired_at'],
            displacement_m=round(distance, 1), minimum_displacement_m=round(lower_bound, 1) if lower_bound is not None else None,
            elapsed_hours=round(elapsed/3600, 3), state=state,
            note='Sparse revisits cannot establish identity, route, velocity, or continuous tracking. Error bounds are supplied, not statistical confidence intervals.')
        movement.append(item)
        if state == 'POSSIBLE MOVEMENT':
            anomalies.append(dict(kind='POSITION CHANGE CANDIDATE', **item))
    return {'state': 'REVIEW CANDIDATES' if any(o['presence'] == 'CANDIDATE' for o in observations) else 'INSUFFICIENT EVIDENCE',
            'observations': observations, 'movement_candidates': movement, 'anomalies': anomalies,
            'limitations': 'Detector scores require independent calibration. Thermal surface products and medium-resolution scenes cannot confirm small aircraft. No detection does not establish absence.'}

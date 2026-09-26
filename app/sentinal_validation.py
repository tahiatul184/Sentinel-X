"""Auditable offline detection evaluation. Scores remain uncalibrated model scores.

Boxes use pixel xyxy in each original scene's common grid. Fully reviewed scenes
must include empty ground-truth lists; unreviewed regions are not negatives.
"""
import math
from collections import defaultdict
import numpy as np


def finite(value, name, low=0, high=float('inf')):
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} outside valid range')
    return value


def check_manifest(scenes):
    if not isinstance(scenes, list) or not scenes:
        raise ValueError('scenes must be a nonempty list')
    ids, groups, source_ids = set(), defaultdict(set), defaultdict(set)
    for s in scenes:
        for field in ('scene_id', 'source_scene_id', 'spatial_group', 'country', 'sensor', 'season', 'split'):
            if not str(s.get(field, '')).strip():
                raise ValueError(f'Missing {field}')
        if s['scene_id'] in ids:
            raise ValueError('Duplicate scene_id')
        ids.add(s['scene_id'])
        if s['split'] not in {'train', 'calibration', 'test'}:
            raise ValueError('split must be train, calibration or test')
        groups[s['spatial_group']].add(s['split'])
        source_ids[s['source_scene_id']].add(s['split'])
        finite(s['area_km2'], 'area_km2', 1e-10)
        finite(s['native_gsd_m'], 'native_gsd_m', .01, 1000)
        if not isinstance(s.get('ground_truth'), list):
            raise ValueError('ground_truth must be a list (including reviewed empty scenes)')
        for box in s['ground_truth']:
            check_box(box)
    if any(len(v) > 1 for v in groups.values()):
        raise ValueError('Spatial leakage: spatial_group crosses dataset splits')
    if any(len(v) > 1 for v in source_ids.values()):
        raise ValueError('Source leakage: source_scene_id crosses dataset splits')
    return ids


def check_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError('Each box must be pixel xyxy with four values')
    x1, y1, x2, y2 = [finite(x, 'box coordinate') for x in box]
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Box must have positive width and height')
    return x1, y1, x2, y2


def iou(a, b):
    x = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    y = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = x*y
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def _metrics(scenes, predictions, threshold, overlap):
    by_id = {s['scene_id']: s for s in scenes}
    predictions = sorted([p for p in predictions if p['scene_id'] in by_id], key=lambda p: -p['score'])
    matched = defaultdict(set)
    labels, scores = [], []
    for p in predictions:
        sid = p['scene_id']
        # Match highest IoU unmatched truth, not an already claimed truth.
        choices = [(iou(p['box'], b), j) for j, b in enumerate(by_id[sid]['ground_truth']) if j not in matched[sid]]
        best = max(choices, default=(0, -1))
        hit = best[0] >= overlap
        if hit:
            matched[sid].add(best[1])
        labels.append(int(hit)); scores.append(p['score'])
    labels = np.asarray(labels, dtype=float); scores = np.asarray(scores, dtype=float)
    total = sum(len(s['ground_truth']) for s in scenes)
    chosen = scores >= threshold
    tp = int(labels[chosen].sum()); fp = int(chosen.sum()) - tp; fn = total - tp
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/total if total else None
    ap = None
    if total:
        cumulative = np.cumsum(labels)
        r = np.r_[0, cumulative/total, 1]
        p = np.r_[0, cumulative/np.arange(1, len(labels)+1), 0]
        p = np.maximum.accumulate(p[::-1])[::-1]
        ap = float(np.sum((r[1:]-r[:-1])*p[1:]))
    ece = brier = None
    if len(scores):
        brier = float(np.mean((scores-labels)**2)); ece = 0.0
        for k in range(10):
            take = (scores >= k/10) & ((scores < (k+1)/10) if k < 9 else scores <= 1)
            if take.any():
                ece += float(take.mean()*abs(scores[take].mean()-labels[take].mean()))
    return dict(scenes=len(scenes), ground_truth=total, predictions=len(scores), tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                ap=ap, fp_per_km2=fp/sum(s['area_km2'] for s in scenes),
                candidate_brier=brier, candidate_ece=ece)


def evaluate(document, threshold=.5, overlap=.5):
    """Evaluate every method on the SAME fully annotated test scenes, including misses."""
    threshold=finite(threshold, 'threshold', 0, 1); overlap=finite(overlap, 'overlap', .01, 1)
    scenes = document['scenes']; ids = check_manifest(scenes)
    test = [s for s in scenes if s['split'] == 'test']
    if not test or any(s.get('fully_reviewed') is not True for s in test):
        raise ValueError('Every test scene must be fully_reviewed; unreviewed scenes cannot establish false positives')
    methods = document['methods']
    if not isinstance(methods, dict) or not methods:
        raise ValueError('methods must map method names to prediction lists')
    result = {}
    for name, predictions in methods.items():
        if not isinstance(predictions, list):
            raise ValueError('Method predictions must be lists')
        clean = []
        for p in predictions:
            if p['scene_id'] not in ids:
                raise ValueError('Prediction references an unknown scene')
            clean.append(dict(scene_id=p['scene_id'], box=check_box(p['box']), score=finite(p['score'], 'score', 0, 1)))
        strata = {}
        for field in ('country', 'sensor', 'season'):
            for value in sorted({s[field] for s in test}):
                subset = [s for s in test if s[field] == value]
                strata[f'{field}:{value}'] = _metrics(subset, clean, threshold, overlap)
        result[name] = dict(overall=_metrics(test, clean, threshold, overlap), strata=strata)
    return dict(methods=result, threshold=threshold, iou_threshold=overlap,
                notes=['Boxes must share each scene original pixel grid. Area must match fully reviewed footprint.',
                       'AP is all-points interpolated AP at the selected IoU; not COCO mAP.',
                       'Candidate Brier/ECE measure matched proposal correctness, not empty-sky probability.',
                       'Spatial groups are user-declared: inspect geographic separation and label quality independently.',
                       'Choose threshold on calibration split, never by optimizing test performance.',
                       'No confidence intervals or statistical significance are claimed.'])


def representation_change(a, b, max_gap_hours=72):
    """Within-checkpoint cosine change only; never blend incompatible latent spaces."""
    from datetime import datetime
    max_gap_hours = finite(max_gap_hours, 'max_gap_hours', .001)
    for key in ('model_id', 'checkpoint_sha256', 'band_schema', 'site_id', 'footprint_id'):
        if not a.get(key) or a.get(key) != b.get(key):
            raise ValueError(f'Paired features require matching {key}')
    times = [datetime.fromisoformat(r['acquired_at'].replace('Z', '+00:00')) for r in (a,b)]
    if any(t.tzinfo is None for t in times):
        raise ValueError('Timezone required')
    gap = (times[1]-times[0]).total_seconds()/3600
    if not 0 < gap <= max_gap_hours:
        raise ValueError('Temporal gap outside configured window')
    for r in (a,b):
        if r.get('alignment_verified') is not True or finite(r.get('valid_fraction', 0), 'valid_fraction', 0, 1) < .8:
            raise ValueError('Verified alignment and >=80% valid footprint required')
    x, y = (np.asarray(r['embedding'], dtype=float) for r in (a,b))
    if x.ndim != 1 or x.shape != y.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Compatible finite one-dimensional embeddings required')
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        raise ValueError('Zero-norm embedding')
    return dict(cosine_change=float(1-np.clip(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)), -1, 1)),
                elapsed_hours=gap, state='REPRESENTATION CHANGE ONLY',
                note='Contextual feature change is not aircraft presence, identity, or an attributed event.')

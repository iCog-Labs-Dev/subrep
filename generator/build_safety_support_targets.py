"""Build empirical, capped-simplex teacher regions from saved three-objective outcomes.

This is an offline grid teacher, not a statistical safety guarantee. Each region
has one non-idle witness policy; never merge regions certified for different
policies. Learned predictions must be certified again at benchmark time.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from certification.pds_test import PDSGate
from utils.weight_set_store import WeightSetStore


def region_grid(resolution):
    caps = np.array(list(itertools.product(np.linspace(0, 1, resolution + 1), repeat=3)))
    caps = caps[caps.sum(axis=1) > 1 + 1e-8]
    # Relative area inside the simplex, by inclusion-exclusion.
    area = np.zeros(len(caps))
    for mask in itertools.product((0, 1), repeat=3):
        area += (-1) ** sum(mask) * np.maximum(1 - caps @ np.array(mask), 0) ** 2
    return caps[area > 1e-8], area[area > 1e-8]


def choose_region(delta, caps, areas, epsilon):
    """Largest grid region with a positive-task, non-idle witness (idle removed by caller)."""
    best = None
    for index, dn in enumerate(delta):
        if dn[1] <= 1e-6:
            continue
        remaining = np.ones(len(caps))
        worst = np.zeros(len(caps))
        for axis in np.argsort(dn):
            mass = np.minimum(remaining, caps[:, axis])
            worst += mass * dn[axis]
            remaining -= mass
        valid = np.flatnonzero(dn[1] + worst >= -epsilon + 1e-6)
        if not len(valid):
            continue
        j = valid[np.argmax(areas[valid])]
        rank = (float(areas[j]), float(dn[1]))
        if best is None or rank > best[0]:
            best = (rank, index, caps[j])
    return best


def vertices(caps):
    result = []
    for order in itertools.permutations(range(3)):
        w = np.zeros(3)
        remaining = 1.
        for axis in order:
            w[axis] = min(remaining, caps[axis])
            remaining -= w[axis]
        result.append(w)
    return np.unique(np.asarray(result, dtype=np.float32), axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--scales-file', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epsilon', type=float, default=.2)
    p.add_argument('--resolution', type=int, default=10)
    args = p.parse_args()
    if not np.isfinite(args.epsilon) or args.epsilon < 0 or not 2 <= args.resolution <= 30:
        p.error('Require finite epsilon >= 0 and resolution between 2 and 30')
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    scales_path = Path(args.scales_file)
    scales = np.asarray(json.loads(scales_path.read_text())['divisors'], dtype=float)
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Expected three finite positive scales')
    caps, areas = region_grid(args.resolution)
    store = WeightSetStore(3)
    seen = set()
    records = []
    for path in sorted(Path(args.data_dir).glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            if not bool(data['values_are_raw']):
                raise ValueError('Expected raw outcomes')
            names = data['candidate_skill_ids'].astype(str)
            values = np.asarray(data['candidate_motives'], dtype=float)
            context = data['context']
            if values.shape != (len(names), 3) or not np.isfinite(values).all():
                raise ValueError('Expected finite three-objective outcomes')
            if np.count_nonzero(names == 'zero_action') != 1:
                raise ValueError('Expected exactly one idle baseline')
            key = store.context_key(context)
            if key in seen:
                raise ValueError('Duplicate context: refusing to merge witness regions')
            seen.add(key)
            delta = (values - values[names == 'zero_action'][0]) / scales
            active = names != 'zero_action'
            choice = choose_region(delta[active], caps, areas, args.epsilon)
            record = dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          context_seed=int(data['context_seed']), admitted=choice is not None)
            if choice is not None:
                rank, index, support = choice
                dn = delta[active][index]
                points = vertices(support)
                target = points.max(axis=0)
                if not PDSGate(args.epsilon).admit(float(dn[1]), dn, support_values=target):
                    raise ValueError('Stored region failed exact gate recheck')
                for w in points:
                    store.observe_certified_weight(context, w)
                record.update(witness=str(names[active][index]), delta_n=dn.tolist(),
                              delta_r=float(dn[1]), support=target.tolist(), relative_area=rank[0])
            records.append(record)
    targets = np.asarray([s for _, s in store.get_all_support_targets()])
    if len(targets) < 3:
        raise ValueError('Fewer than three admissible contexts; cannot split support data')
    if np.allclose(targets, targets[0]):
        raise ValueError('All targets identical: refusing a misleading context-dependent fit')
    root.mkdir(parents=True)
    store.save(root / 'weights.json')
    manifest = dict(contexts=len(records), admitted_contexts=len(targets), epsilon=args.epsilon,
                    resolution=args.resolution, scales=scales.tolist(),
                    scales_sha256=hashlib.sha256(scales_path.read_bytes()).hexdigest(),
                    semantics='Largest-area grid capped simplex with one positive-task non-idle PDS witness. Empirical outcomes only; not a population guarantee.',
                    evaluation_scope='If base MDN used these contexts, support split is head-only validation, not independent end-to-end evaluation.',
                    records=records)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Support targets: {len(targets)}/{len(records)} contexts; saved to {root}')


if __name__ == '__main__':
    main()

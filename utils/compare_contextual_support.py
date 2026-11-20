"""Compare support regions on saved evidence and paired re-execution; no simulation."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from utils.mdn_checkpoint_loader import load_mdn_checkpoint


def coverage(caps):
    """Exact uniform-simplex volume fraction by inclusion-exclusion."""
    caps = np.asarray(caps, dtype=float)
    if caps.shape != (3,) or not np.isfinite(caps).all() or (caps < 0).any() or (caps > 1).any() or caps.sum() < 1 - 1e-6:
        raise ValueError('Expected feasible three-objective caps')
    value = sum((-1)**len(s) * max(0., 1 - caps[list(s)].sum())**2
                for n in range(4) for s in itertools.combinations(range(3), n))
    return float(np.clip(value, 0, 1))


def evaluate_context(record, execution, scales, caps, epsilon):
    baseline = np.asarray(record['outcomes']['zero_action']['motives'])
    admitted = []
    cds = 0
    total = len(record['outcomes']) - 1
    for name, outcome in record['outcomes'].items():
        if name == 'zero_action':
            continue
        dn = (np.asarray(outcome['motives']) - baseline) / scales
        dr = float(dn[1])
        first = CDSGate().admit(dr, dn, support_values=caps)
        if first or PDSGate(epsilon).admit(dr, dn, support_values=caps):
            admitted.append((name, dr, dn))
            cds += int(first)
    selections = []
    for weight in ([1/3]*3, [.8,.1,.1], [.1,.8,.1], [.1,.1,.8]):
        candidates = admitted if np.all(np.asarray(weight) <= caps + 1e-9) else []
        ranked = sorted(candidates, key=lambda item: (-(item[1] + np.dot(weight, item[2])), item[0]))
        name = ranked[0][0] if ranked else 'zero_action'
        actual = execution['outcomes'][name]
        raw = actual.get('undiscounted_motives')
        selections.append(dict(policy=name, weight=weight, fallback=not bool(ranked),
            discounted_cost=-actual['motives'][0], task_return=actual['motives'][1],
            effort=-actual['motives'][2], goal_success=float(actual['goals'] > 0),
            raw_cost=-raw[0] if raw is not None else None))
    return dict(total=total, passed=len(admitted), cds=cds, pds_only=len(admitted)-cds,
                rejected=total-len(admitted), coverage=coverage(caps), caps=caps.tolist(), selections=selections)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-dir', required=True)
    p.add_argument('--fixed-caps', type=float, nargs=3, default=[.8,.8,.8],
                   help='Predeclared baseline; do not tune on evaluation contexts')
    args = p.parse_args()
    fixed = np.asarray(args.fixed_caps)
    coverage(fixed)
    root = Path(args.benchmark_dir)
    manifest = json.loads((root/'manifest.json').read_text())
    checkpoint = manifest['mdn_checkpoint']
    import hashlib
    if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != manifest['mdn_sha256']:
        raise ValueError('MDN checkpoint changed since benchmark')
    model = load_mdn_checkpoint(checkpoint, map_location='cpu')
    model.eval()
    rows = {name: [] for name in ('full_simplex', 'fixed_region', 'learned_region')}
    for seed in manifest['config']['training_seeds']:
        folder = root/f'seed_{seed}'
        scales = np.asarray(json.loads((folder/'scales.json').read_text()))
        for record in json.loads((folder/'evidence.json').read_text()):
            execution = json.loads((folder/str(record['context_seed'])/'execution.json').read_text())
            if execution['context_seed'] != record['context_seed'] or not np.array_equal(execution['initial_observation'], record['initial_observation']):
                raise ValueError('Execution context mismatch')
            with torch.no_grad():
                caps = model.forward_inference(torch.tensor(record['initial_observation'], dtype=torch.float32))[1].numpy().reshape(-1)
            for name, values in zip(rows, (np.ones(3), fixed, caps)):
                row = evaluate_context(record, execution, scales, values, manifest['config']['epsilon'])
                rows[name].append(dict(training_seed=seed, context_seed=record['context_seed'], **row))
    summary = {}
    for name, records in rows.items():
        selected = [s for r in records for s in r['selections']]
        summary[name] = {k: sum(r[k] for r in records) for k in ('total','passed','rejected','cds','pds_only')}
        summary[name]['mean_region_coverage'] = float(np.mean([r['coverage'] for r in records]))
        for key in ('fallback','discounted_cost','task_return','effort','goal_success','raw_cost'):
            values = [s[key] for s in selected if s[key] is not None]
            summary[name]['mean_'+key] = float(np.mean(values)) if len(values) == len(selected) and values else None
    report = dict(scope='Independent region ablations, without full-simplex-first fallback. Same-start re-execution, not transfer certification. Raw cost is descriptive, not a separate safety certificate.',
                  fixed_caps=fixed.tolist(), summary=summary, contexts=rows)
    (root/'support_comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = ['# Support-region comparison', '', report['scope'], '',
             '| Region | Passed | Rejected | Coverage | Fallback | Goal success | Raw cost |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, s in summary.items():
        lines.append(f"| {name} | {s['passed']} | {s['rejected']} | {s['mean_region_coverage']:.3f} | {s['mean_fallback']:.3f} | {s['mean_goal_success']:.3f} | {s['mean_raw_cost']} |")
    (root/'support_comparison.md').write_text('\n'.join(lines)+'\n')
    print('Saved:', root/'support_comparison.md')


if __name__ == '__main__':
    main()

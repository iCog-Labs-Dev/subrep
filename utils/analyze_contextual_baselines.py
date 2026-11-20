"""Offline end-to-end selector comparison on saved contextual benchmark episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def score(outcome, baseline, scales, weight):
    delta = (np.asarray(outcome['motives']) - np.asarray(baseline['motives'])) / scales
    return float(delta[1] + np.dot(weight, delta))


def analyze_query(evidence, execution, library, selection, scales):
    """Select using evidence only; measure using the paired execution only."""
    weight = np.asarray(selection['weight'])
    outcomes = evidence['outcomes']
    names = sorted(set(outcomes) - {'zero_action'})
    if set(outcomes) != set(execution['outcomes']) or not {'ppo','ppo_lagrangian','zero_action'} <= set(outcomes):
        raise ValueError('Missing or inconsistent candidate outcomes')
    ids, eligible = {}, []
    for skill_id, entry in library['skills'].items():
        name = skill_id.split(':')[0]
        cert = entry['certificate']
        ids[name] = skill_id
        caps = cert.get('wx_support_values')
        if cert['weight_region_type'] == 'FULL_SIMPLEX' or (caps is not None and np.all(weight <= np.asarray(caps) + 1e-9)):
            eligible.append(name)
    def best(pool):
        return min(pool, key=lambda n: (-score(outcomes[n], outcomes['zero_action'], scales, weight), ids.get(n,n))) if pool else 'zero_action'
    chosen = best(eligible)
    if chosen != selection['policy'] or bool(selection['fallback']) != (not eligible):
        raise ValueError('Reconstructed SubRep selection differs from saved selection')
    unrestricted = best(names)
    pools = {'subrep': [chosen], 'zero_action': ['zero_action'], 'always_ppo': ['ppo'],
             'always_ppo_lagrangian': ['ppo_lagrangian'], 'random_candidate': names,
             'random_certified': eligible or ['zero_action'], 'without_certification': [unrestricted]}
    rows = []
    for method, pool in pools.items():
        measurements = []
        for name in pool:
            actual = execution['outcomes'][name]
            raw = actual['undiscounted_motives']
            actual_score = score(actual, execution['outcomes']['zero_action'], scales, weight)
            measurements.append(dict(score=actual_score, task_return=actual['motives'][1],
                safety_cost=-raw[0], discounted_safety_cost=-actual['motives'][0],
                effort=-actual['motives'][2], goal_success=float(actual['goals'] > 0),
                worse_than_idle=float(actual_score < -1e-9), idle_execution=float(name == 'zero_action')))
        row = {key: float(np.mean([m[key] for m in measurements])) for key in measurements[0]}
        row.update(method=method, scenario=selection['scenario'],
                   fallback=float(method in ('subrep','random_certified') and not eligible))
        rows.append(row)
    ablation = dict(scenario=selection['scenario'],
        blocked_selection=float(unrestricted not in eligible),
        rejected_selection=float(unrestricted not in ids),
        outside_region_selection=float(unrestricted in ids and unrestricted not in eligible))
    return rows, ablation


def run(root, bootstrap=2000, seed=42):
    root = Path(root)
    sources = {}
    def read(path):
        data = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    manifest = read(root/'manifest.json')
    rows, ablations = [], []
    for training_seed in manifest['config']['training_seeds']:
        folder = root/f'seed_{training_seed}'
        scales = np.asarray(read(folder/'scales.json'))
        if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Invalid scales')
        evidence = read(folder/'evidence.json')
        seen = set()
        for record in evidence:
            context = record['context_seed']
            if context in seen:
                raise ValueError('Duplicate context')
            seen.add(context)
            local = folder/str(context)
            execution = read(local/'execution.json')
            if execution['context_seed'] != context or not np.array_equal(execution['initial_observation'],record['initial_observation']):
                raise ValueError('Execution start mismatch')
            library = read(local/'library.json')
            for selection in read(local/'selection.json'):
                r, a = analyze_query(record, execution, library, selection, scales)
                identity = dict(training_seed=training_seed, context_seed=context)
                rows.extend(dict(**identity, **x) for x in r)
                ablations.append(dict(**identity, **a))
    metrics = ['score','task_return','safety_cost','discounted_safety_cost','effort','goal_success','fallback','idle_execution','worse_than_idle']
    summary, comparisons = [], []
    rng = np.random.default_rng(seed)
    for scenario in sorted({r['scenario'] for r in rows}):
        group = [r for r in rows if r['scenario'] == scenario]
        methods = sorted({r['method'] for r in group})
        ordered = {m: sorted([r for r in group if r['method'] == m], key=lambda r:(r['context_seed'],r['training_seed'])) for m in methods}
        for method, entries in ordered.items():
            summary.append(dict(scenario=scenario,method=method,queries=len(entries),**{k:float(np.mean([r[k] for r in entries])) for k in metrics}))
        # Resample context seeds, keeping paired methods and training seeds together.
        context_ids = sorted({r['context_seed'] for r in group})
        indices = rng.integers(0,len(context_ids),(bootstrap,len(context_ids)))
        for method in methods:
            if method == 'subrep': continue
            for metric in ('score','safety_cost','goal_success'):
                diffs = np.array([np.mean([a[metric]-b[metric] for a,b in zip(ordered['subrep'],ordered[method]) if a['context_seed']==c]) for c in context_ids])
                ci = np.quantile(diffs[indices].mean(1),[.025,.975])
                comparisons.append(dict(scenario=scenario,baseline=method,metric=metric,
                    subrep_minus_baseline=float(diffs.mean()),ci95=ci.tolist()))
    ablation_summary = []
    for scenario in sorted({a['scenario'] for a in ablations}):
        group = [a for a in ablations if a['scenario']==scenario]
        ablation_summary.append(dict(scenario=scenario,**{k:float(np.mean([a[k] for a in group])) for k in ('blocked_selection','rejected_selection','outside_region_selection')}))
    scope = ('Offline paired replay of saved executions. Selection uses evidence only. Random selectors report exact uniform expectations over saved outcomes, not new random rollouts. '
             'Task/effort are discounted; safety_cost is undiscounted. Score = normalized task delta + weighted normalized objective deltas. '
             'No absolute safety guarantee. Context bootstrap intervals do not quantify training-seed uncertainty. Removing certification removes admission and region restrictions together.')
    report = dict(scope=scope,bootstrap_samples=bootstrap,seed=seed,summary=summary,paired_comparisons=comparisons,
                  ablation=ablation_summary,rows=rows,source_sha256=sources)
    (root/'baseline_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines = ['# End-to-end SubRep baseline comparison','',scope,'',
        '| Priority | Method | Score | Task | Raw cost | Effort | Goal success | Fallback |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['scenario']} | {r['method']} | {r['score']:.3f} | {r['task_return']:.3f} | {r['safety_cost']:.3f} | {r['effort']:.3f} | {r['goal_success']:.1%} | {r['fallback']:.1%} |")
    lines += ['', '## Certification ablation','', '| Priority | With gates raw cost | Without gates raw cost | Blocked selection | Rejected | Outside region |','|---|---:|---:|---:|---:|---:|']
    for a in ablation_summary:
        lookup = {r['method']:r for r in summary if r['scenario']==a['scenario']}
        lines.append(f"| {a['scenario']} | {lookup['subrep']['safety_cost']:.3f} | {lookup['without_certification']['safety_cost']:.3f} | {a['blocked_selection']:.1%} | {a['rejected_selection']:.1%} | {a['outside_region_selection']:.1%} |")
    lines += ['', '## Paired differences: SubRep minus baseline','', '| Priority | Baseline | Metric | Difference | Context-bootstrap 95% interval |','|---|---|---|---:|---|']
    for r in comparisons:
        lines.append(f"| {r['scenario']} | {r['baseline']} | {r['metric']} | {r['subrep_minus_baseline']:.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] |")
    (root/'baseline_comparison.md').write_text('\n'.join(lines)+'\n')
    print('Saved:',root/'baseline_comparison.md')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark-dir', required=True)
    p.add_argument('--bootstrap-samples',type=int,default=2000)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.bootstrap_samples < 1 or args.seed < 0: p.error('Invalid bootstrap settings')
    run(args.benchmark_dir,args.bootstrap_samples,args.seed)


if __name__ == '__main__': main()

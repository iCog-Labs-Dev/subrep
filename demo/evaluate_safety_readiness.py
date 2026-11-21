"""Paired four-objective development comparison and exact-context gate screening.

No MDN, no transfer claim, and no statistical safety guarantee.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from pilot.safety_gymnasium_ppo import SafetyPPOPilot
from utils.contextual_safety_benchmark import certify
from utils.safety_reuse_benchmark import Collector, Config, fit_scales, write_json


def summarize(records, names, scales, epsilon, cost_limit):
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Require four finite positive scales')
    if not records or not names:
        raise ValueError('Require contexts and policies')
    if len({r['context_seed'] for r in records}) != len(records):
        raise ValueError('Duplicate context seeds')
    rows, audits = [], []
    for name in names:
        totals, success, admitted, joint = [], [], [], []
        for record in records:
            outcome = record['outcomes'][name]
            raw = np.asarray(outcome['undiscounted_motives'], dtype=float)
            dn = (np.asarray(outcome['motives']) - np.asarray(record['outcomes']['zero_action']['motives'])) / scales
            if raw.shape != (4,) or dn.shape != (4,) or not np.isfinite(raw).all() or not np.isfinite(dn).all():
                raise ValueError('Require finite four-objective outcomes')
            decision, attempts = certify(float(dn[1]), dn, epsilon)
            goal = outcome['goals'] > 0
            passed = decision is not None
            totals.append(raw * [-1, 1, -1, -1])
            success.append(goal); admitted.append(passed)
            joint.append(passed and goal and -raw[0] <= cost_limit)
            audits.append(dict(policy=name, context_seed=record['context_seed'], admitted=passed,
                               goal_reached=bool(goal), episode_cost=float(-raw[0]),
                               delta_r=float(dn[1]), delta_n=dn.tolist(), attempts=attempts))
        mean = np.mean(totals, axis=0)
        rows.append(dict(policy=name, contexts=len(records), goal_rate=float(np.mean(success)),
                         mean_cost=float(mean[0]), mean_return=float(mean[1]), mean_effort=float(mean[2]),
                         mean_action_change=float(mean[3]), admission_rate=float(np.mean(admitted)),
                         admitted_goal_and_episode_budget_rate=float(np.mean(joint))))
    coverage = np.mean([any(a['admitted'] for a in audits if a['context_seed'] == r['context_seed']) for r in records])
    return dict(policies=rows, policy_context_admission_rate=float(np.mean([a['admitted'] for a in audits])),
                context_coverage=float(coverage), audits=audits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=610000)
    p.add_argument('--contexts', type=int, default=100)
    p.add_argument('--development-contexts', type=int, default=30)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--epsilon', type=float, default=.2)
    p.add_argument('--cost-limit', type=float, default=1.)
    args = p.parse_args()
    config = Config(output=args.output, seed=args.seed, contexts=args.contexts,
                    development_contexts=args.development_contexts, max_steps=args.max_steps,
                    epsilon=args.epsilon, objectives=4, training_seeds=(42,))
    config.validate()
    if not np.isfinite(args.cost_limit) or args.cost_limit < 0:
        p.error('Require a finite nonnegative cost limit')
    root = Path(args.output)
    if root.exists():
        p.error('Use a new output directory')
    models, provenance = {}, []
    for value in args.checkpoint:
        path = Path(value).resolve()
        if path.stem in models:
            p.error('Checkpoint names must be unique')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload.get('metadata', {}).get('environment') != config.env_id:
            p.error(f'Environment metadata mismatch: {path}')
        model = SafetyPPOPilot.load(path, map_location='cpu'); model.eval()
        models[path.stem] = model
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    collector = Collector(config, models)
    try:
        # Only the requested trained policies count towards the acceptance denominator.
        collector.policies = {k: v for k, v in collector.policies.items() if k == 'zero_action' or k in models}
        root.mkdir(parents=True)
        dev_seeds = list(range(args.seed + 1, args.seed + 1 + args.development_contexts))
        seeds = list(range(dev_seeds[-1] + 1, dev_seeds[-1] + 1 + args.contexts))
        write_json(root/'manifest.json', dict(settings=vars(args), checkpoints=provenance,
                   scale_context_seeds=dev_seeds, comparison_context_seeds=seeds,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   scope='Development only; deterministic policies; exact-start empirical certification against zero action. No MDN.'))
        dev = collector.collect(dev_seeds)
        write_json(root/'development.json', dev)
        scales = fit_scales([dev])
        write_json(root/'scales.json', dict(objective_names=['Safety','Task','ControlEfficiency','ActionSmoothness'], divisors=scales.tolist()))
        records = collector.collect(seeds)
        write_json(root/'evidence.json', records)
    finally:
        collector.close()
    result = summarize(records, list(models), scales, args.epsilon, args.cost_limit)
    write_json(root/'summary.json', result)
    lines = ['# Four-objective policy readiness (development)', '',
             'Outcome metrics are undiscounted episode totals. Certification uses discounted, development-scaled improvements against zero action.',
             'Full-simplex CDS/PDS only; MDN-region acceptance is not measured. Gates use these exact-start empirical outcomes, not independent safety confidence bounds.',
             'Task return also appears as payoff in the existing certification formula. Cost limit is a separate diagnostic, not a new gate.', '',
             f"Policy–context admission: {result['policy_context_admission_rate']:.1%}; contexts with at least one admission: {result['context_coverage']:.1%}.", '',
             '| Policy | Goal success | Mean cost | Mean return | Effort | Action change | Admitted | Admitted + goal + episode cost within limit |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['policies']:
        lines.append(f"| {r['policy']} | {r['goal_rate']:.1%} | {r['mean_cost']:.3f} | {r['mean_return']:.3f} | {r['mean_effort']:.3f} | {r['mean_action_change']:.3f} | {r['admission_rate']:.1%} | {r['admitted_goal_and_episode_budget_rate']:.1%} |")
    lines += ['', 'A 60% admission rate alone is not proof of task competence or safety. The joint column is an episode-level diagnostic, stricter than a mean-cost budget.',
              'Freeze settings before running; do not adjust epsilon/scales or discard difficult contexts to reach 60%. Reserve unused contexts for final benchmarking.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('Report:', root/'report.md')


if __name__ == '__main__':
    main()

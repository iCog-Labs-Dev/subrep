"""Offline filter evaluation on COMPLETE held-out raw_mixed contexts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from generator.multi_policy_data import load_records, load_or_create_manifest, partition, group_contexts
from generator.multi_policy_pipeline import load_model
from generator.candidate_filter import select_top_k
from generator.policy_registry import POLICY_NAMES


def assess(groups, selections, baseline_stats):
    """Decisions use recorded outcomes, never predictions."""
    calculator = ImprovementCalculator(baseline_stats)
    cds, pds = CDSGate(), PDSGate()
    all_count = selected_count = all_cert = selected_cert = cds_admit = pds_admit = 0
    payoff = []
    motives = []
    delta_payoff = []
    delta_motives = []
    for context, rows in groups.items():
        by_name = {r['policy']: r for r in rows}
        chosen = selections[context]
        if len(set(chosen)) != len(chosen) or not set(chosen) <= set(by_name):
            raise ValueError('Invalid candidate selection')
        for r in rows:
            delta_r, delta_n = calculator.compute_improvements(r['payoff'], r['motives'])
            decision = {'delta_r': delta_r, 'delta_n': np.asarray(delta_n),
                        'cds_admit': bool(cds.admit(delta_r, delta_n)),
                        'pds_admit': bool(pds.admit(delta_r, delta_n))}
            certifiable = decision['cds_admit'] and decision['pds_admit']
            all_count += 1
            all_cert += int(certifiable)
            if r['policy'] in chosen:
                selected_count += 1
                selected_cert += int(certifiable)
                cds_admit += int(decision['cds_admit'])
                pds_admit += int(decision['pds_admit'])
                payoff.append(r['payoff'])
                motives.append(r['motives'])
                delta_payoff.append(float(decision['delta_r']))
                delta_motives.append(np.asarray(decision['delta_n']).tolist())
    return dict(total_candidates=all_count, executed=selected_count,
                avoided=all_count-selected_count, execution_savings_fraction=(all_count-selected_count)/all_count,
                genuinely_certifiable=all_cert, certifiable_found=selected_cert,
                certifiable_missed=all_cert-selected_cert,
                missed_certifiable_fraction=(all_cert-selected_cert)/all_cert if all_cert else None,
                cds_admission_rate=cds_admit/selected_count,
                pds_admission_rate=pds_admit/selected_count,
                cds_rejection_rate=1-cds_admit/selected_count,
                pds_rejection_rate=1-pds_admit/selected_count,
                measured_payoff_mean=float(np.mean(payoff)),
                measured_motives_mean=np.mean(motives, axis=0).tolist(),
                measured_payoff_improvement_mean=float(np.mean(delta_payoff)),
                measured_motive_improvement_mean=np.mean(delta_motives, axis=0).tolist(),
                )

def evaluate_filter(groups, model, baseline_stats, k=3, seed=42):
    import random
    rng = random.Random(seed)
    neural = {}
    random_selection = {}
    fixed = {}
    all_candidates = {}
    fixed_order = ['ppo_deterministic'] + [p for p in POLICY_NAMES if p != 'ppo_deterministic']
    for context, rows in groups.items():
        names = [r['policy'] for r in rows]
        neural[context] = select_top_k(model, rows[0]['obs'], names, k)
        random_selection[context] = rng.sample(sorted(names), k)
        fixed[context] = [name for name in fixed_order if name in names][:k]
        all_candidates[context] = names
    return {'neural_top_k': assess(groups, neural, baseline_stats),
            'random_same_budget': assess(groups, random_selection, baseline_stats),
            'fixed_order_same_budget': assess(groups, fixed, baseline_stats),
            'all_candidates_full_budget': assess(groups, all_candidates, baseline_stats)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='data/raw_mixed')
    parser.add_argument('--split-manifest', default='data/mixed_split_manifest.json')
    parser.add_argument('--model-path', default='models/multi_policy_generator.pt')
    parser.add_argument('--baseline-episodes', type=int, default=20)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default='reports/multi_policy')
    args = parser.parse_args(argv)
    rows = load_records(args.data_dir)
    manifest = load_or_create_manifest(rows, args.split_manifest, seed=args.seed)
    test = partition(rows, manifest)['test']
    groups = group_contexts(test, complete=True)
    model, conditioned = load_model(args.model_path)
    if not conditioned:
        raise ValueError('Filter requires policy-conditioned model')
    from generator.evaluate_generator_report import compute_idle_baseline_stats
    baseline_stats = compute_idle_baseline_stats(episodes=args.baseline_episodes, seed=args.seed)
    result = evaluate_filter(groups, model, baseline_stats, k=args.k, seed=args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/'filter_metrics.json').write_text(json.dumps(result, indent=2))
    lines = ['# Multi-policy candidate filter — offline held-out evaluation', '',
             'All outcomes below were measured during complete candidate-set collection.',
             'Execution savings are **simulated**, not savings achieved while collecting the benchmark.',
             'CDS/PDS decisions use recorded outcomes and unchanged gates.',
             '| Method | Executed | Savings | Certifiable found | Missed rate | CDS admit | PDS admit |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, r in result.items():
        miss = 'N/A' if r['missed_certifiable_fraction'] is None else f"{r['missed_certifiable_fraction']:.1%}"
        lines.append(f"| {name} | {r['executed']} | {r['execution_savings_fraction']:.1%} | {r['certifiable_found']} | {miss} | {r['cds_admission_rate']:.1%} | {r['pds_admission_rate']:.1%} |")
    (out/'results_summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

"""Collect paired four-objective data and train MDN plus optional margin support.

Training/development only. Run certification on separate contexts afterwards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from pilot.safety_gymnasium_ppo import SafetyPPOPilot
from utils.safety_reuse_benchmark import Collector, Config, fit_scales, write_json

OBJECTIVES = ['Safety', 'Task', 'ControlEfficiency', 'ActionSmoothness']


def export_training(records, destination, scales):
    """Keep raw values for support training; attach scales for the MDN adapter."""
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Require four positive finite scales')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    names_reference = None
    for record in records:
        names = list(record['outcomes'])
        if names_reference is None:
            names_reference = names
        if names != names_reference or names.count('zero_action') != 1:
            raise ValueError('Inconsistent candidates or missing baseline')
        values = np.asarray([record['outcomes'][n]['motives'] for n in names], dtype=np.float32)
        if values.shape != (len(names), 4) or not np.isfinite(values).all():
            raise ValueError('Require finite four-objective outcomes')
        path = destination / f"context_{record['context_seed']}.npz"
        if path.exists():
            raise ValueError('Duplicate context seed')
        np.savez_compressed(path, context=np.asarray(record['initial_observation'], dtype=np.float32),
            candidate_skill_ids=np.asarray(names), candidate_motives=values, candidate_payoffs=values[:, 1],
            context_seed=record['context_seed'], objective_names=np.asarray(OBJECTIVES),
            objective_scales=scales, values_are_raw=True, env_id='SafetyPointGoal1-v0', gamma=.99,
            scaling_source='separate_development_contexts', schema_version=3)


def training_commands(root, epsilon, regularization, epochs, seed):
    root = Path(root)
    scales = json.loads((root/'scales.json').read_text())
    if scales['objective_names'] != OBJECTIVES:
        raise ValueError('Objective order mismatch')
    divisors = np.asarray(scales['divisors'], dtype=float)
    if divisors.shape != (4,) or not np.isfinite(divisors).all() or np.any(divisors <= 0):
        raise ValueError('Invalid scales')
    base = root/'mdn'
    return [
        [sys.executable, '-m', 'generator.train_mdn_candidate_sets', '--data-dir', str(root/'candidate_sets'),
         '--baseline-candidate-id', 'zero_action', '--seed', str(seed), '--device', 'cpu',
         '--policy-checkpoint', str(base/'policy.pth'), '--auxiliary-checkpoint', str(base/'auxiliary.pth')],
        [sys.executable, '-m', 'generator.train_mdn_margin_support', '--data-dir', str(root/'candidate_sets'),
         '--scales-file', str(root/'scales.json'), '--payoff-scale', str(divisors[1]),
         '--baseline-candidate-id', 'zero_action', '--base-checkpoint', str(base/'auxiliary.pth'),
         '--output-dir', str(root/'support'), '--eligibility', 'feasible', '--epsilon', str(epsilon),
         '--regularization', str(regularization), '--epochs', str(epochs), '--seed', str(seed)]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', action='append')
    p.add_argument('--stage', choices=['collect','train','all'], default='all')
    p.add_argument('--contexts', type=int, default=300)
    p.add_argument('--scale-contexts', type=int, default=30)
    p.add_argument('--context-seed', type=int, default=620000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--epsilon', type=float, default=.2)
    p.add_argument('--regularization', type=float, default=.1)
    p.add_argument('--support-epochs', type=int, default=500)
    args = p.parse_args()
    if min(args.contexts, args.scale_contexts) < 10 or args.max_steps < 1 or args.support_epochs < 1:
        p.error('Require >=10 contexts per collection and positive steps/epochs')
    if min(args.seed, args.context_seed) < 0 or not np.isfinite([args.epsilon,args.regularization]).all() or min(args.epsilon,args.regularization) < 0:
        p.error('Invalid seed, epsilon or regularization')
    root = Path(args.output).resolve()
    if args.stage in ('collect','all'):
        if not args.checkpoint or root.exists():
            p.error('Collection requires --checkpoint and a new output directory')
        models, provenance = {}, []
        for value in args.checkpoint:
            path = Path(value).resolve()
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if path.stem in models or payload.get('metadata', {}).get('environment') != 'SafetyPointGoal1-v0':
                p.error(f'Duplicate name or environment mismatch: {path}')
            model = SafetyPPOPilot.load(path, map_location='cpu'); model.eval()
            models[path.stem] = model
            provenance.append(dict(name=path.stem, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        config = Config(str(root), objectives=4, max_steps=args.max_steps)
        collector = Collector(config, models)
        try:
            collector.policies = {k:v for k,v in collector.policies.items() if k=='zero_action' or k in models}
            root.mkdir(parents=True)
            scale_seeds = list(range(args.context_seed+1, args.context_seed+1+args.scale_contexts))
            train_seeds = list(range(scale_seeds[-1]+1, scale_seeds[-1]+1+args.contexts))
            write_json(root/'manifest.json', dict(settings=vars(args), checkpoints=provenance,
                scale_context_seeds=scale_seeds, training_context_seeds=train_seeds,
                role='Training/development only; trainers split context groups internally. No independent final test.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            development = collector.collect(scale_seeds)
            write_json(root/'scale_evidence.json', development)
            scales = fit_scales([development])
            write_json(root/'scales.json', dict(objective_names=OBJECTIVES, divisors=scales.tolist()))
            records = collector.collect(train_seeds)
            write_json(root/'training_evidence.json', records)
            export_training(records, root/'candidate_sets', scales)
            write_json(root/'benchmark_contract.json', dict(environment=config.env_id, observation_schema='native_flat_v1',
                objectives=['safety','task','control_efficiency','action_smoothness']))
        finally:
            collector.close()
    if args.stage in ('train','all'):
        if not (root/'manifest.json').is_file() or not (root/'candidate_sets').is_dir():
            p.error('Missing collection; run --stage collect first')
        if (root/'mdn').exists() or (root/'support').exists():
            p.error('Training output already exists; refusing to overwrite checkpoints')
        commands = training_commands(root, args.epsilon, args.regularization, args.support_epochs, args.seed)
        write_json(root/'training_commands.json', commands)
        (root/'mdn').mkdir()
        for command in commands:
            subprocess.run(command, check=True)
        print('Support checkpoint:', root/'support/support.pth')
    print('Run directory:', root)
    print('Next: certification comparison on unused contexts; regional admission is not guaranteed.')


if __name__ == '__main__':
    main()

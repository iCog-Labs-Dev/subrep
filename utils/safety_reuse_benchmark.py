"""Paired, held-out three-objective certification and fixed-policy reuse benchmark.

Certificates describe empirical policy means over an evidence distribution, not
state-conditioned guarantees. Evaluation replays fixed policies on paired seeds;
priority changes occur between episodes. No learned MDN is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np
import torch

from certification.certificate_schema import Certificate
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from library.skill_library import SkillLibrary
from library.skill_selector import select_best_skill_entry
from pilot.safety_gymnasium_ppo import SafetyPPOPilot

METHODS = ('subrep', 'always_ppo', 'always_ppo_lagrangian', 'unrestricted',
           'random_candidate', 'random_certified', 'idle')
CERTIFIED = ('subrep', 'random_certified')
OBJECTIVES = ('safety', 'task', 'control_efficiency')


@dataclass
class Config:
    output: str
    training_seeds: tuple[int, ...] = (42, 43, 44)
    ppo_template: str = 'models/safety_ppo_fixed_seed{seed}_updates50.pt'
    lagrangian_template: str = 'models/safety_ppo_lagrangian_fixed_seed{seed}_updates50.pt'
    env_id: str = 'SafetyPointGoal1-v0'
    seed: int = 60000
    development_contexts: int = 30
    contexts: int = 30
    batches: int = 5
    max_steps: int = 200
    epsilon: float = 0.2
    gamma: float = 0.99
    bootstrap_samples: int = 2000

    def validate(self):
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError('Training seeds must be nonempty and unique')
        if any(s < 0 for s in self.training_seeds) or self.seed < 0:
            raise ValueError('Seeds must be nonnegative')
        if min(self.development_contexts, self.contexts) < 2:
            raise ValueError('At least two development and evaluation contexts required')
        if min(self.batches, self.max_steps, self.bootstrap_samples) < 1:
            raise ValueError('Batches, max steps and bootstrap samples must be positive')
        if not np.isfinite(self.epsilon) or self.epsilon < 0 or not 0 <= self.gamma <= 1:
            raise ValueError('Require finite epsilon >=0 and gamma in [0,1]')


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def seed_plan(config):
    """Same environment seeds across training seeds; disjoint stage ranges."""
    start = config.seed + 1
    development = list(range(start, start + config.development_contexts))
    start += config.development_contexts
    batches = []
    for _ in range(config.batches):
        batches.append({'evidence': list(range(start, start + config.contexts)),
                        'evaluation': list(range(start + config.contexts, start + 2 * config.contexts))})
        start += 2 * config.contexts
    return {'development': development, 'batches': batches}


def priorities(count):
    safety, task, effort = np.array([[.8,.1,.1], [.1,.8,.1], [.1,.1,.8]])
    return {'balanced': np.tile([1/3]*3, (count, 1)),
            'safety': np.tile(safety, (count, 1)), 'task': np.tile(task, (count, 1)),
            'effort': np.tile(effort, (count, 1)),
            'gradual': np.array([(1-t)*safety+t*task for t in np.linspace(0, 1, count)]),
            'abrupt': np.array([safety if i < count//2 else task for i in range(count)]),
            'empty_library_control': np.tile([1/3]*3, (count, 1))}


def control_effort(action, low, high):
    action = np.asarray(action, dtype=float)
    if (action.shape != low.shape or not np.isfinite(action).all()
            or np.any(action < low) or np.any(action > high)):
        raise ValueError('Policy produced an invalid action')
    if np.any(low >= 0) or np.any(high <= 0):
        raise ValueError('Action bounds must straddle zero')
    return float(np.mean((action / np.where(action >= 0, high, -low))**2))


def load_models(config):
    """Reject mislabeled/duplicate seeds and save exact checkpoint provenance."""
    groups, provenance, seen = {}, [], {'ppo': set(), 'ppo_lagrangian': set()}
    for seed in config.training_seeds:
        group = {}
        for name, template in [('ppo', config.ppo_template), ('ppo_lagrangian', config.lagrangian_template)]:
            path = Path(template.format(seed=seed))
            payload = torch.load(path, map_location='cpu', weights_only=False)
            meta = payload.get('metadata', {})
            settings = meta.get('config', {})
            if settings.get('seed') != seed or bool(settings.get('use_lagrangian')) != (name == 'ppo_lagrangian'):
                raise ValueError(f'Checkpoint seed/type metadata mismatch: {path}')
            if meta.get('environment') != config.env_id:
                raise ValueError(f'Checkpoint environment mismatch: {path}')
            digest = hashlib.sha256()
            for key, tensor in sorted(payload['state_dict'].items()):
                digest.update(key.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            state_hash = digest.hexdigest()
            if state_hash in seen[name]:
                raise ValueError(f'Duplicate trained parameters across seeds: {path}')
            seen[name].add(state_hash)
            group[name] = SafetyPPOPilot.load(path, map_location='cpu')
            provenance.append({'training_seed': seed, 'policy': name, 'path': str(path.resolve()),
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'parameter_sha256': state_hash, 'training_config': settings})
        groups[seed] = group
    return groups, provenance


class Collector:
    """Native Safety-Gymnasium rollouts, including goal events and three motives."""
    def __init__(self, config, models):
        import safety_gymnasium
        self.env = safety_gymnasium.make(config.env_id)
        self.config = config
        self.low = np.asarray(self.env.action_space.low)
        self.high = np.asarray(self.env.action_space.high)
        for model in models.values():
            if (model.observation_dim != int(np.prod(self.env.observation_space.shape))
                    or model.action_dim != self.low.size
                    or not np.allclose(model.action_low.cpu().numpy(), self.low)
                    or not np.allclose(model.action_high.cpu().numpy(), self.high)):
                self.env.close()
                raise ValueError('Checkpoint observation/action contract mismatch')
        self.rng = np.random.default_rng(0)
        self.policies = {'zero_action': lambda obs: np.zeros_like(self.low),
                         'small_random': lambda obs: self.rng.uniform(self.low, self.high).astype(np.float32)*.25,
                         'random': lambda obs: self.rng.uniform(self.low, self.high).astype(np.float32)}
        for axis in range(2):
            for sign, label in [(1, 'positive'), (-1, 'negative')]:
                action = np.zeros_like(self.low)
                action.flat[axis % action.size] = sign*.35
                self.policies[f'axis{axis}_{label}'] = lambda obs, a=action: np.clip(a, self.low, self.high)
        for name, model in models.items():
            self.policies[name] = lambda obs, m=model: m.predict(obs, deterministic=True)

    def collect(self, seeds):
        records = []
        for seed in seeds:
            outcomes, reference = {}, None
            for index, (name, policy) in enumerate(self.policies.items()):
                obs, _ = self.env.reset(seed=seed)
                if reference is None:
                    reference = np.array(obs, copy=True)
                elif not np.allclose(obs, reference):
                    raise RuntimeError('Environment seed did not reproduce initial observation')
                self.rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
                motives, totals = np.zeros(3), np.zeros(3)
                goals = 0
                for step in range(self.config.max_steps):
                    action = np.asarray(policy(obs), dtype=np.float32)
                    effort = control_effort(action, self.low, self.high)
                    obs, reward, cost, terminated, truncated, info = self.env.step(action)
                    vector = np.array([-float(cost), float(reward), -effort])
                    if not np.isfinite(vector).all():
                        raise ValueError('Nonfinite environment measurement')
                    motives += self.config.gamma**step * vector
                    totals += vector
                    goals += int(bool(info.get('goal_met', False)))
                    if terminated or truncated:
                        break
                outcomes[name] = {'motives': motives.tolist(), 'undiscounted_motives': totals.tolist(),
                                  'goals': goals, 'steps': step+1,
                                  'stop_reason': 'terminated' if terminated else 'truncated' if truncated else 'max_steps'}
            records.append({'context_seed': int(seed), 'outcomes': outcomes})
            if len(records) % 5 == 0 or len(records) == len(seeds):
                print(f'Collected {len(records)}/{len(seeds)} contexts; last environment seed {seed}', flush=True)
        return records

    def close(self):
        self.env.close()


def fit_scales(development_groups):
    deltas = []
    for records in development_groups:
        for r in records:
            base = np.array(r['outcomes']['zero_action']['motives'])
            deltas.extend(np.array(v['motives'])-base for k,v in r['outcomes'].items() if k != 'zero_action')
    scales = np.mean(np.abs(deltas), axis=0)
    return np.where(scales > 1e-12, scales, 1.)


def make_plan(records, scales, policies, config, seed, folder):
    names = list(policies)
    if any(set(r['outcomes']) != set(names) for r in records):
        raise ValueError('Evidence policy pool mismatch')
    means = {name: np.mean([r['outcomes'][name]['motives'] for r in records], axis=0) for name in names}
    deltas = {name: (means[name]-means['zero_action'])/scales for name in names}
    library = SkillLibrary()
    audit = []
    for name in names:
        if name == 'zero_action':
            continue
        dn = deltas[name]
        dr = float(dn[1])
        cds, pds = CDSGate(), PDSGate(epsilon=config.epsilon)
        cds_pass, pds_pass = cds.admit(dr, dn), pds.admit(dr, dn)
        gate = 'CDS' if cds_pass else 'PDS' if pds_pass else None
        margin = float(dr + min(dn))
        audit.append({'skill_id': name, 'gate': gate, 'cds_pass': bool(cds_pass), 'pds_pass': bool(pds_pass),
                      'delta_r': dr, 'delta_n': dn.tolist(), 'worst_case_score': margin,
                      'epsilon': config.epsilon, 'admitted': gate is not None})
        if gate:
            epsilon = 0. if gate == 'CDS' else config.epsilon
            cert = Certificate(name, gate, dr, tuple(dn), max(0., margin+epsilon), epsilon,
                               datetime.now(timezone.utc).isoformat(), int(seed), config.gamma,
                               'zero_action', config.env_id, config.max_steps, 'safety-reuse-v1')
            if not library.add_skill(name, cert, policies[name]):
                raise RuntimeError(f'Library rejected gate-admitted certificate: {name}')
    library.save(str(Path(folder)/'library.json'))
    write_json(Path(folder)/'certificates.json', [e.certificate.to_dict() for e in library.get_admitted_skills()])
    rng = np.random.default_rng(seed)
    candidates = [n for n in names if n != 'zero_action']
    rows = []
    for scenario, weights in priorities(config.contexts).items():
        for step, w in enumerate(weights):
            entries = [] if scenario == 'empty_library_control' else library.query_admissible(w)
            eligible = [e.skill_id for e in entries]
            scores = {name: float(dn[1] + w@dn) for name, dn in deltas.items()}
            unrestricted = min(candidates, key=lambda n: (-scores[n], n))
            choices = {'subrep': select_best_skill_entry(entries, w)[0] if entries else 'zero_action',
                       'always_ppo': 'ppo', 'always_ppo_lagrangian': 'ppo_lagrangian',
                       'unrestricted': unrestricted, 'random_candidate': str(rng.choice(candidates)),
                       'random_certified': str(rng.choice(eligible)) if eligible else 'zero_action', 'idle': 'zero_action'}
            for method, name in choices.items():
                entry = library.get_skill(name)
                threshold = -entry.epsilon if entry is not None else None
                rows.append({'scenario': scenario, 'step': step, 'method': method, 'selected_skill': name,
                             'weights': w.tolist(), 'evidence_score': scores[name],
                             'certificate_threshold': threshold,
                             'certified_execution': method in CERTIFIED and bool(entries),
                             'abstained': method in CERTIFIED and not entries,
                             'idle_execution': name == 'zero_action', 'eligible_count': len(entries)})
    # Bind the verified library's executable policies into paired evaluation.
    for entry in library.get_admitted_skills():
        policies[entry.skill_id] = entry.policy
    return {'audit': audit, 'rows': rows}


def evaluate(plan, records, scales, config):
    rows = []
    for selection in plan['rows']:
        record = records[selection['step']]
        result = record['outcomes'][selection['selected_skill']]
        dn = (np.array(result['motives'])-record['outcomes']['zero_action']['motives'])/scales
        score = float(dn[1] + np.dot(selection['weights'], dn))
        threshold = selection['certificate_threshold']
        rows.append({**selection, 'context_seed': record['context_seed'], 'score': score,
                     'task_return': result['motives'][1], 'safety_cost': -result['motives'][0],
                     'control_effort': -result['motives'][2], 'goals': result['goals'],
                     'goal_success': result['goals'] > 0,
                     'undiscounted_safety_cost': -result['undiscounted_motives'][0],
                     'negative_transfer': score < -1e-9,
                     'budget_violation': score < -config.epsilon-1e-9,
                     'certificate_violation': score < threshold-1e-9 if selection['certified_execution'] else None})
    return rows


def crossed_interval(cube, samples, rng):
    """Resample training runs and paired environment batches/contexts separately.

    Environment draws are shared across sampled training runs. Methods have
    already been differenced on each episode, preserving paired comparisons.
    """
    nt, nb, nc = cube.shape
    draws = []
    for _ in range(samples):
        ts = rng.integers(nt, size=nt)
        bs = rng.integers(nb, size=nb)
        values = [cube[ts[:, None], b, rng.integers(nc, size=nc)[None, :]].mean() for b in bs]
        draws.append(float(np.mean(values)))
    return np.quantile(draws, [.025, .975]).tolist()


def summarize(rows, config):
    output = []
    rng = np.random.default_rng(config.seed)
    for scenario in priorities(config.contexts):
        for method in METHODS:
            selected = [r for r in rows if r['scenario'] == scenario and r['method'] == method]
            reference = {(r['training_seed'], r['batch'], r['step']): r['score'] for r in rows
                         if r['scenario'] == scenario and r['method'] == 'subrep'}
            cube = np.zeros((len(config.training_seeds), config.batches, config.contexts))
            differences = cube.copy()
            for r in selected:
                index = (config.training_seeds.index(r['training_seed']), r['batch'], r['step'])
                cube[index] = r['score']
                differences[index] = r['score']-reference[(r['training_seed'], r['batch'], r['step'])]
            reused = [r for r in selected if r['certified_execution']]
            metrics = {key: float(np.mean([r[key] for r in selected])) for key in
                       ('score', 'task_return', 'safety_cost', 'control_effort', 'goals', 'goal_success',
                        'undiscounted_safety_cost', 'abstained', 'idle_execution', 'negative_transfer', 'budget_violation')}
            conditional = {key: float(np.mean([r[key] for r in reused])) if reused else None for key in
                           ('score', 'task_return', 'safety_cost', 'control_effort', 'goal_success', 'certificate_violation')}
            output.append({'scenario': scenario, 'method': method, 'episodes': len(selected), **metrics,
                           'score_ci95': crossed_interval(cube, config.bootstrap_samples, rng),
                           'difference_vs_subrep': float(differences.mean()),
                           'difference_vs_subrep_ci95': crossed_interval(differences, config.bootstrap_samples, rng),
                           'training_seed_scores': dict(zip(map(str, config.training_seeds), cube.mean(axis=(1,2)).tolist())),
                           'training_seed_score_sd': float(np.std(cube.mean(axis=(1,2)), ddof=1)) if len(cube)>1 else None,
                           'certified_executions': len(reused), 'certified_only': conditional})
    return output


def run(config, *, collector_class=Collector, models=None, provenance=None):
    config.validate()
    if Path(config.output).exists():
        raise FileExistsError(f'Use a new output directory: {config.output}')
    if models is None:
        models, provenance = load_models(config)
    root = Path(config.output)
    root.mkdir(parents=True)
    splits = seed_plan(config)
    versions = {}
    for package in ('numpy', 'torch', 'safety-gymnasium', 'gymnasium', 'mujoco'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    source_paths = [Path(__file__), Path(__file__).parents[1]/'demo/run_safety_reuse_benchmark.py',
                    Path(__file__).parents[1]/'library/skill_library.py',
                    Path(__file__).parents[1]/'certification/cds_test.py',
                    Path(__file__).parents[1]/'certification/pds_test.py',
                    Path(__file__).parents[1]/'pilot/safety_gymnasium_ppo.py']
    write_json(root/'manifest.json', {'config': asdict(config), 'splits': splits, 'checkpoints': provenance,
        'versions': versions, 'commit': commit, 'dirty': dirty,
        'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'collector': collector_class.__name__, 'objectives': OBJECTIVES,
        'score': 'normalized task payoff + weighted normalized motives (task contributes twice)',
        'scope': 'Empirical mean, FULL_SIMPLEX certificates; fixed episode policies; shifts between episodes; no MDN',
        'ci': '95% crossed bootstrap over training seeds and shared environment batches/contexts; three training seeds give limited precision'})
    development = []
    for training_seed in config.training_seeds:
        collector = collector_class(config, models[training_seed])
        try:
            records = collector.collect(splits['development'])
            write_json(root/f'seed_{training_seed}/development.json', records)
            development.append(records)
        finally:
            collector.close()
        print(f'Development collected: training seed {training_seed}', flush=True)
    scales = fit_scales(development)
    write_json(root/'frozen_scales.json', {'divisors': scales.tolist(), 'objectives': OBJECTIVES,
                                        'source': 'all development policies and training seeds; mean absolute baseline deltas'})
    all_rows, admissions = [], []
    for training_seed in config.training_seeds:
        collector = collector_class(config, models[training_seed])
        try:
            for batch, split in enumerate(splits['batches']):
                folder = root/f'seed_{training_seed}/batch_{batch}'
                folder.mkdir(parents=True)
                evidence = collector.collect(split['evidence'])
                write_json(folder/'evidence.json', evidence)
                plan = make_plan(evidence, scales, collector.policies, config,
                                 int(np.random.SeedSequence([config.seed, training_seed, batch]).generate_state(1)[0]), folder)
                write_json(folder/'selection_plan.json', plan)
                # All methods' selections are on disk BEFORE any evaluation rollout.
                evaluation = collector.collect(split['evaluation'])
                write_json(folder/'evaluation.json', evaluation)
                rows = [{**r, 'training_seed': training_seed, 'batch': batch}
                        for r in evaluate(plan, evaluation, scales, config)]
                write_json(folder/'results.json', rows)
                all_rows.extend(rows)
                cds = sum(a['gate']=='CDS' for a in plan['audit'])
                pds = sum(a['gate']=='PDS' for a in plan['audit'])
                count = len(plan['audit'])
                admissions.append({'training_seed': training_seed, 'batch': batch, 'attempted': count,
                                   'cds': cds, 'pds': pds, 'admitted': cds+pds, 'rejected': count-cds-pds,
                                   'admission_rate': (cds+pds)/count, 'rejection_rate': 1-(cds+pds)/count})
                print(f'Completed seed {training_seed}, batch {batch}: {cds} CDS, {pds} PDS / {count}', flush=True)
        finally:
            collector.close()
    summary = summarize(all_rows, config)
    write_json(root/'admissions.json', admissions)
    write_json(root/'summary.json', summary)
    lines = ['# Held-out three-objective SubRep reuse benchmark', '',
             'Scores are improvements over idle; higher is better. Costs and effort are discounted totals.',
             'Success means at least one actual goal_met event, not positive task reward.', '',
             '| Scenario | Method | Score [95% CI] | Task return | Goal success | Safety cost | Effort | Fallback | Certified violations |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lo, hi = r['score_ci95']
        cv = r['certified_only']['certificate_violation']
        violation = 'n/a' if cv is None else f'{cv:.1%}'
        lines.append(f"| {r['scenario']} | {r['method']} | {r['score']:.3f} [{lo:.3f}, {hi:.3f}] | {r['task_return']:.3f} | {r['goal_success']:.1%} | {r['safety_cost']:.3f} | {r['control_effort']:.3f} | {r['abstained']:.1%} | {violation} |")
    lines += ['', '## Paired comparisons', '',
              'Difference is method minus SubRep. Negative favors SubRep; intervals spanning zero are inconclusive.', '',
              '| Scenario | Method | Difference [95% CI] |', '|---|---|---:|']
    for r in summary:
        if r['method'] != 'subrep':
            lo, hi = r['difference_vs_subrep_ci95']
            lines.append(f"| {r['scenario']} | {r['method']} | {r['difference_vs_subrep']:.3f} [{lo:.3f}, {hi:.3f}] |")
    lines += ['', '## Admission counts', '', '| Training seed | Batch | CDS | PDS | Rejected |', '|---|---|---:|---:|---:|']
    lines += [f"| {a['training_seed']} | {a['batch']} | {a['cds']} | {a['pds']} | {a['rejected']} |" for a in admissions]
    lines += ['', '## Interpretation', '',
        '- summary.json includes paired method-minus-SubRep score confidence intervals, training-seed means, and certified-only outcomes.',
        '- Negative paired differences favor SubRep. An interval spanning zero is inconclusive.',
        '- Empirical mean certificates do not guarantee new-episode safety or individual-objective improvement.',
        '- All candidate policies are executed on paired seeds; precommitted selections index these outcomes. No evaluation outcomes influence selection.',
        '- FULL_SIMPLEX weights change between reset episodes. This does not test state-conditioned retrieval, within-episode switching, learned MDN geometry, or MetaMo.',
        '- Idle fallback is separate from successful reuse. The empty-library control is excluded from primary interpretation.',
        '- Three independent training runs give limited uncertainty precision. Smoke runs establish plumbing only.',
        '- Task payoff also appears in the motive vector; normalization and epsilon determine the allowed trade-offs.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Complete: {root / "report.md"}', flush=True)
    return summary

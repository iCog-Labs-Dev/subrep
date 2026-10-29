"""Exact-start contextual certificates; no claim of transfer to unseen starts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from certification.certificate_schema import Certificate
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from library.skill_library import SkillLibrary
from library.skill_selector import select_best_skill_entry
from utils.mdn_checkpoint_loader import load_mdn_checkpoint
from utils.safety_reuse_benchmark import Config, Collector, load_models, fit_scales, write_json


def context_key(record, config):
    obs = np.asarray(record['initial_observation'], dtype=float)
    if obs.ndim != 1 or not np.isfinite(obs).all():
        raise ValueError('Expected a finite flat initial observation')
    payload = [config.env_id, config.gamma, config.max_steps, record['context_seed'],
               record['observation_schema'], obs.tolist()]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def certify(dr, dn, epsilon, support=None):
    """Ordered first-pass decision, preserving every attempted gate."""
    attempts = []
    for region, values in [('FULL_SIMPLEX', None), ('MDN_WX', support)]:
        if region == 'MDN_WX' and values is None:
            attempts.extend({'region': region, 'gate': g, 'status': 'unavailable'} for g in ('CDS', 'PDS'))
            break
        for gate in (CDSGate(), PDSGate(epsilon)):
            passed = gate.admit(dr, dn, support_values=values)
            margin = gate.get_admission_margin(dr, dn, support_values=values)
            attempts.append({'region': region, 'gate': gate.get_gate_type(),
                             'status': 'passed' if passed else 'failed', 'margin': float(margin)})
            if passed:
                return attempts[-1], attempts
    return None, attempts


def eligible(library, stored_key, current_key, weight):
    # Context filtering applies even to FULL_SIMPLEX certificates.
    if stored_key != current_key:
        return []
    weight = np.asarray(weight, dtype=float)
    if weight.shape != (3,) or not np.isfinite(weight).all() or np.any(weight < 0) or not np.isclose(weight.sum(), 1.):
        raise ValueError('Expected three finite simplex weights')
    result = []
    for entry in library.get_admitted_skills():
        caps = entry.wx_support_values
        if caps is not None and np.any(np.asarray(weight) > np.asarray(caps) + 1e-9):
            continue
        result.append(entry)
    return result


def run_contextual(config, mdn_checkpoint=None, mdn_contract=None, scales_file=None, resume=False):
    config.validate()
    root = Path(config.output)
    if root.exists() and not resume:
        raise FileExistsError(f'Use a new output directory: {root}')
    model = None
    if mdn_checkpoint:
        if not mdn_contract:
            raise ValueError('Supply --mdn-contract describing the model input and objective order')
        contract = json.loads(Path(mdn_contract).read_text())
        expected = {'environment': config.env_id, 'observation_schema': 'native_flat_v1',
                    'objectives': ['safety', 'task', 'control_efficiency']}
        if any(contract.get(k) != v for k, v in expected.items()):
            raise ValueError('MDN contract does not match this environment/observation/objective schema')
        model = load_mdn_checkpoint(mdn_checkpoint)
        if model.num_objectives != 3:
            raise ValueError('MDN must have three objectives')
    models, provenance = load_models(config)
    manifest = {'config': vars(config), 'checkpoints': provenance,
        'scope': 'Exact reset seed and observation only; same-start re-execution, not unseen-context transfer',
        'score': 'normalized task payoff plus weighted motives; task contributes twice',
        'scales_file': scales_file, 'mdn_checkpoint': mdn_checkpoint, 'mdn_contract': mdn_contract,
        'mdn_sha256': hashlib.sha256(Path(mdn_checkpoint).read_bytes()).hexdigest() if model else None,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if resume:
        previous = json.loads((root/'manifest.json').read_text())
        normalized = json.loads(json.dumps(manifest))
        for field in ('config', 'checkpoints', 'mdn_sha256', 'mdn_contract', 'scales_file'):
            if previous[field] != normalized[field]:
                raise ValueError(f'Resume provenance mismatch: {field}')
        write_json(root/'resume_manifest.json', manifest)
    else:
        root.mkdir(parents=True)
        write_json(root/'manifest.json', manifest)
    dev_seeds = list(range(config.seed + 1, config.seed + 1 + config.development_contexts))
    evidence_seeds = list(range(dev_seeds[-1] + 1, dev_seeds[-1] + 1 + config.contexts))
    audits, results = [], []
    for seed in config.training_seeds:
        collector = Collector(config, models[seed])
        try:
            folder = root/f'seed_{seed}'
            dev_path = folder/'development.json'
            dev = json.loads(dev_path.read_text()) if resume and dev_path.exists() else collector.collect(dev_seeds)
            if scales_file:
                from utils.safety_objectives import FrozenObjectiveScales
                scales = np.asarray(FrozenObjectiveScales.load(scales_file).divisors)
            else:
                scales = fit_scales([dev])
            if resume and (folder/'scales.json').exists():
                if not np.array_equal(scales, np.asarray(json.loads((folder/'scales.json').read_text()))):
                    raise ValueError('Resume scales changed')
            write_json(folder/'development.json', dev)
            write_json(folder/'scales.json', scales.tolist())
            evidence_path = folder/'evidence.json'
            evidence = json.loads(evidence_path.read_text()) if resume and evidence_path.exists() else collector.collect(evidence_seeds)
            if [r['context_seed'] for r in evidence] != evidence_seeds:
                raise ValueError('Evidence context seeds do not match configuration')
            write_json(folder/'evidence.json', evidence)
            for record in evidence:
                key = context_key(record, config)
                context = record['initial_observation']
                alpha = support = None
                if model:
                    if len(context) != model.trunk[0].in_features:
                        raise ValueError('MDN input dimension does not match native observation')
                    with torch.no_grad():
                        a, s = model.forward_inference(torch.tensor(context, dtype=torch.float32))
                    alpha, support = a.numpy().reshape(-1), s.numpy().reshape(-1)
                library = SkillLibrary()
                baseline = np.asarray(record['outcomes']['zero_action']['motives'])
                policy_ids = {p['policy']: p['parameter_sha256'] for p in provenance if p['training_seed'] == seed}
                names = {}
                for name, outcome in record['outcomes'].items():
                    if name == 'zero_action':
                        continue
                    dn = (np.asarray(outcome['motives']) - baseline) / scales
                    dr = float(dn[1])
                    decision, attempts = certify(dr, dn, config.epsilon, support)
                    policy_id = name + ':' + policy_ids.get(name, 'fixed_v1')
                    skill_id = policy_id + ':' + key
                    names[skill_id] = name
                    audits.append({'training_seed': seed, 'context_key': key, 'context_seed': record['context_seed'],
                        'policy_id': policy_id, 'skill_id': skill_id, 'delta_r': dr, 'delta_n': dn.tolist(),
                        'attempts': attempts, 'admitted': decision is not None,
                        'evidence': f'seed_{seed}/evidence.json'})
                    if decision:
                        meta = {}
                        if decision['region'] == 'MDN_WX':
                            meta = dict(certification_context=tuple(context), mdn_alpha=tuple(alpha),
                                wx_support_directions=tuple(map(tuple, np.eye(3))), wx_support_values=tuple(support))
                        cert = Certificate(skill_id, decision['gate'], dr, tuple(dn), decision['margin'],
                            0. if decision['gate'] == 'CDS' else config.epsilon,
                            datetime.now(timezone.utc).isoformat(), seed, config.gamma, 'zero_action',
                            config.env_id, config.max_steps, 'exact-context-v1',
                            weight_region_type=decision['region'], **meta)
                        if not library.add_skill(skill_id, cert, collector.policies[name],
                                                  weight_region_type=decision['region'], **meta):
                            raise RuntimeError('Library rejected certificate')
                local = folder/str(record['context_seed'])
                local.mkdir(parents=True, exist_ok=True)
                library.save(str(local/'library.json'))
                selections = []
                # Check the actual reset observation before selecting, not future outcomes.
                obs, _ = collector.env.reset(seed=record['context_seed'])
                current = {**record, 'initial_observation': np.asarray(obs).tolist()}
                for label, w in [('balanced', [1/3]*3), ('safety', [.8,.1,.1]),
                                 ('task', [.1,.8,.1]), ('effort', [.1,.1,.8])]:
                    entries = eligible(library, key, context_key(current, config), w)
                    chosen = select_best_skill_entry(entries, np.asarray(w))[0] if entries else None
                    selections.append({'scenario': label, 'weight': w, 'skill_id': chosen,
                                       'policy': names[chosen] if chosen else 'zero_action', 'fallback': chosen is None})
                write_json(local/'selection.json', selections)
                fresh = collector.collect([record['context_seed']], repeat=1)[0]
                if context_key(fresh, config) != key:
                    raise RuntimeError('Evaluation start differs from selected context')
                write_json(local/'execution.json', fresh)
                for selection in selections:
                    actual = fresh['outcomes'][selection['policy']]
                    results.append({'training_seed': seed, 'context_seed': record['context_seed'], **selection,
                                    'actual_motives': actual['motives'], 'goals': actual['goals']})
        finally:
            collector.close()
    write_json(root/'admissions.json', audits)
    write_json(root/'results.json', results)
    lines = ['# Context-specific certification', '',
        'Exact-start evidence and same-start re-execution only. Deterministic policies may repeat identical outcomes.',
        'MDN checks: ' + ('enabled' if model else 'unavailable (no checkpoint supplied)'), '',
        '| Seed | Context | Policy | Result |', '|---|---|---|---|']
    for row in audits:
        passed = next((a for a in row['attempts'] if a['status'] == 'passed'), None)
        status = passed['region']+'/'+passed['gate'] if passed else 'Rejected'
        lines.append(f"| {row['training_seed']} | {row['context_seed']} | {row['policy_id'].split(':')[0]} | {status} |")
    lines += ['', '## Re-execution outcomes', '',
              '| Seed | Context | Priority | Policy | Fallback | Safety cost | Task return | Effort | Goals |',
              '|---|---|---|---|---|---|---|---|---|']
    for row in results:
        safety, task, effort = row['actual_motives']
        lines.append(f"| {row['training_seed']} | {row['context_seed']} | {row['scenario']} | {row['policy']} | {row['fallback']} | {-safety:.4f} | {task:.4f} | {-effort:.4f} | {row['goals']} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Results: {root}/report.md')

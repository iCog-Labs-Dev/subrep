"""Paired full-simplex/fixed/MDN certification on unused reset contexts.

Exact-start empirical evidence and same-start re-execution, not transfer or a
statistical safety guarantee. Full-simplex admissions are retained in each method.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from certification.certificate_schema import Certificate
from library.skill_library import SkillLibrary
from library.skill_selector import select_best_skill_entry
from pilot.safety_gymnasium_ppo import SafetyPPOPilot
from utils.contextual_safety_benchmark import certify, context_key, eligible, priority_queries
from utils.mdn_checkpoint_loader import load_mdn_checkpoint
from utils.safety_reuse_benchmark import Collector, Config, write_json
from demo.run_four_objective_mdn import OBJECTIVES

METHODS = ('full_simplex', 'full_plus_fixed', 'full_plus_mdn')


def region_coverage(caps):
    """Uniform volume fraction of the 4D weight simplex inside coordinate caps."""
    caps = np.asarray(caps, dtype=float)
    if caps.shape != (4,) or not np.isfinite(caps).all() or np.any(caps < 0) or np.any(caps > 1) or caps.sum() < 1-1e-6:
        raise ValueError('Expected feasible four-objective caps')
    volume = sum((-1)**len(s) * max(0., 1-caps[list(s)].sum())**3
                 for n in range(5) for s in itertools.combinations(range(4), n))
    return float(np.clip(volume, 0, 1))


def check_seeds(manifest, seeds):
    if len(set(seeds)) != len(seeds) or not seeds or min(seeds) < 0:
        raise ValueError('Require unique nonnegative evaluation seeds')
    used = set(manifest['scale_context_seeds']) | set(manifest['training_context_seeds'])
    if used.intersection(seeds):
        raise ValueError('Evaluation contexts overlap training/scale contexts')


def evaluate_evidence(record, scales, alpha, caps, fixed, config, policies):
    """Make all gate/selection decisions before viewing re-execution outcomes."""
    region_coverage(caps); region_coverage(fixed)
    alpha = np.asarray(alpha, dtype=float)
    if alpha.shape != (4,) or not np.isfinite(alpha).all() or np.any(alpha <= 0):
        raise ValueError('Expected four finite positive alpha values')
    scales = np.asarray(scales, dtype=float)
    if scales.shape != (4,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Invalid scales')
    key = context_key(record, config)
    base = np.asarray(record['outcomes']['zero_action']['motives'])
    audits = {m: [] for m in METHODS}
    libraries = {m: SkillLibrary() for m in ('full_simplex','full_plus_mdn')}
    candidates = {m: [] for m in METHODS}
    for name, outcome in record['outcomes'].items():
        if name == 'zero_action': continue
        dn = (np.asarray(outcome['motives'])-base)/scales
        dr = float(dn[1])
        if dn.shape != (4,) or not np.isfinite(dn).all():
            raise ValueError('Invalid four-objective motives')
        skill_id = name+':'+key
        for method, support in zip(METHODS, (None, fixed, caps)):
            decision, attempts = certify(dr, dn, config.epsilon, support)
            region = decision['region'] if decision else None
            if method == 'full_plus_fixed':
                # Fixed geometry must never be mislabelled as an MDN prediction.
                attempts = [{**a, 'region': 'FIXED_REGION' if a['region']=='MDN_WX' else a['region']} for a in attempts]
                if region == 'MDN_WX': region = 'FIXED_REGION'
            audits[method].append(dict(policy=name, context_seed=record['context_seed'], admitted=decision is not None,
                region=region, gate=decision['gate'] if decision else None, attempts=attempts,
                delta_r=dr, delta_n=dn.tolist()))
            if decision is None: continue
            limits = None if region == 'FULL_SIMPLEX' else np.asarray(support)
            candidates[method].append((skill_id,name,dr,dn,limits))
            if method not in libraries: continue
            meta = {}
            if region == 'MDN_WX':
                meta = dict(certification_context=tuple(float(v) for v in record['initial_observation']),
                    mdn_alpha=tuple(float(v) for v in alpha), wx_support_directions=tuple(map(tuple,np.eye(4))),
                    wx_support_values=tuple(float(v) for v in caps))
            cert = Certificate(skill_id=skill_id, gate_type=decision['gate'], delta_r=dr, delta_n=tuple(dn),
                admission_margin=decision['margin'], epsilon=0. if decision['gate']=='CDS' else config.epsilon,
                timestamp=datetime.now(timezone.utc).isoformat(), seed=record['context_seed'], gamma=config.gamma,
                baseline_id='zero_action', environment=config.env_id, episode_length=config.max_steps,
                version='four-objective-exact-context-v1', weight_region_type=region, **meta)
            if not libraries[method].add_skill(skill_id,cert,policies[name],weight_region_type=region,**meta):
                raise RuntimeError('Library rejected certificate')
    selections = {m: [] for m in METHODS}
    for method in METHODS:
        name_by_id = {item[0]:item[1] for item in candidates[method]}
        for label, weight in priority_queries(4):
            if method in libraries:
                entries = eligible(libraries[method],key,key,weight)
                chosen = select_best_skill_entry(entries,np.asarray(weight))[0] if entries else None
            else:
                options = [c for c in candidates[method] if c[4] is None or np.all(np.asarray(weight)<=c[4]+1e-9)]
                options.sort(key=lambda c: (-(c[2]+np.dot(weight,c[3])),c[0]))
                chosen = options[0][0] if options else None
            selections[method].append(dict(priority=label, weight=weight, skill_id=chosen,
                policy=name_by_id[chosen] if chosen else 'zero_action', fallback=chosen is None))
    return audits, selections, libraries


def summarize(contexts, cost_limit):
    result = {}
    for method in METHODS:
        audits = [a for c in contexts for a in c['audits'][method]]
        selected = [s for c in contexts for s in c['selections'][method]]
        passed = sum(a['admitted'] for a in audits)
        full = sum(a['region']=='FULL_SIMPLEX' for a in audits)
        result[method] = dict(total=len(audits), admitted=passed, rejected=len(audits)-passed,
            admission_rate=passed/len(audits), full_simplex=full, additional_regional=passed-full,
            cds=sum(a['gate']=='CDS' for a in audits), pds=sum(a['gate']=='PDS' for a in audits),
            context_coverage=float(np.mean([any(a['admitted'] for a in c['audits'][method]) for c in contexts])),
            mean_proposed_region_volume=float(np.mean([c['coverage'][method] for c in contexts])),
            fallback_rate=float(np.mean([s['fallback'] for s in selected])),
            goal_rate=float(np.mean([s['goals']>0 for s in selected])),
            mean_cost=float(np.mean([s['cost'] for s in selected])),
            mean_return=float(np.mean([s['task_return'] for s in selected])),
            mean_effort=float(np.mean([s['effort'] for s in selected])),
            mean_action_change=float(np.mean([s['action_change'] for s in selected])),
            admitted_goal_budget_rate=float(np.mean([not s['fallback'] and s['goals']>0 and s['cost']<=cost_limit for s in selected])))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--support-checkpoint', help='Optional newly trained support checkpoint; keep original run data and policies')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--contexts',type=int,default=100)
    p.add_argument('--seed',type=int,default=630000)
    p.add_argument('--fixed-caps',type=float,nargs=4,default=[.8]*4)
    p.add_argument('--cost-limit',type=float,default=1.)
    args=p.parse_args()
    if args.contexts<2 or args.seed<0 or not np.isfinite(args.cost_limit) or args.cost_limit<0:
        p.error('Invalid count, seed or cost limit')
    root=Path(args.output).resolve(); source=Path(args.run_dir).resolve()
    if root.exists(): p.error('Use a new output directory')
    fixed=np.asarray(args.fixed_caps);region_coverage(fixed)
    manifest=json.loads((source/'manifest.json').read_text())
    contract=json.loads((source/'benchmark_contract.json').read_text())
    expected=dict(environment='SafetyPointGoal1-v0',observation_schema='native_flat_v1',
                  objectives=['safety','task','control_efficiency','action_smoothness'])
    if contract!=expected: raise ValueError('MDN contract mismatch')
    scale_data=json.loads((source/'scales.json').read_text())
    scales=np.asarray(scale_data['divisors'])
    if scale_data['objective_names']!=OBJECTIVES or scales.shape!=(4,) or not np.isfinite(scales).all() or np.any(scales<=0):
        raise ValueError('Invalid scale schema')
    checkpoint=Path(args.support_checkpoint).resolve() if args.support_checkpoint else source/'support/support.pth'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    training=payload['support_training']
    if training.get('scales_sha256') != hashlib.sha256((source/'scales.json').read_bytes()).hexdigest():
        raise ValueError('Scales changed since support training')
    if json.loads(json.dumps(training.get('dataset_contract'))) != ['SafetyPointGoal1-v0', .99, OBJECTIVES]:
        raise ValueError('Support training environment/objective contract mismatch')
    settings=training['settings']
    if settings['baseline_candidate_id']!='zero_action' or not np.isclose(settings['payoff_scale'],scales[1]):
        raise ValueError('Support training baseline/payoff scale mismatch')
    config=Config(str(root),objectives=4,contexts=args.contexts,max_steps=manifest['settings']['max_steps'],
                  epsilon=float(settings['epsilon']),seed=args.seed)
    config.validate()
    seeds=list(range(args.seed+1,args.seed+1+args.contexts));check_seeds(manifest,seeds)
    mdn=load_mdn_checkpoint(checkpoint,map_location='cpu');mdn.eval()
    if mdn.num_objectives!=4: raise ValueError('Expected four-objective MDN')
    models={}
    for info in manifest['checkpoints']:
        path=Path(info['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:
            raise ValueError(f'Policy changed since training: {path}')
        if info['name'] in models: raise ValueError('Duplicate policy name')
        models[info['name']]=SafetyPPOPilot.load(path,map_location='cpu');models[info['name']].eval()
    collector=Collector(config,models)
    contexts=[]
    try:
        collector.policies={k:v for k,v in collector.policies.items() if k=='zero_action' or k in models}
        root.mkdir(parents=True)
        write_json(root/'manifest.json',dict(settings=vars(args),config=vars(config),evaluation_seeds=seeds,
            training_manifest=manifest, checkpoints=manifest['checkpoints'],
            mdn_checkpoint=str(checkpoint),mdn_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            scales=scale_data, fixed_caps=fixed.tolist(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Exact-start empirical certification; same-start deterministic re-execution, not independent safety confidence or unseen-state transfer.'))
        for index,seed in enumerate(seeds):
            local=root/str(seed)
            record=collector.collect([seed])[0]
            write_json(local/'evidence.json',record)
            if len(record['initial_observation'])!=mdn.input_dim: raise ValueError('MDN input dimension mismatch')
            with torch.no_grad():
                a,s=mdn.forward_inference(torch.tensor(record['initial_observation'],dtype=torch.float32))
            alpha=a.cpu().numpy().reshape(-1);caps=s.cpu().numpy().reshape(-1)
            audits,selections,libraries=evaluate_evidence(record,scales,alpha,caps,fixed,config,collector.policies)
            write_json(local/'admissions.json',audits)
            write_json(local/'selection.json',selections)
            for method,library in libraries.items():library.save(str(local/f'{method}_library.json'))
            execution=collector.collect([seed],repeat=1)[0]
            if context_key(record,config)!=context_key(execution,config):raise RuntimeError('Execution context mismatch')
            write_json(local/'execution.json',execution)
            for method in METHODS:
                for selection in selections[method]:
                    outcome=execution['outcomes'][selection['policy']];raw=outcome['undiscounted_motives']
                    selection.update(goals=outcome['goals'],cost=-raw[0],task_return=raw[1],effort=-raw[2],action_change=-raw[3])
            contexts.append(dict(context_seed=seed,caps=caps.tolist(),alpha=alpha.tolist(),audits=audits,selections=selections,
                coverage=dict(full_simplex=1.,full_plus_fixed=region_coverage(fixed),full_plus_mdn=region_coverage(caps))))
            print(f'Completed comparison {index+1}/{len(seeds)}',flush=True)
    finally:collector.close()
    summary=summarize(contexts,args.cost_limit)
    write_json(root/'support_comparison.json',dict(summary=summary,contexts=contexts))
    lines=['# Four-objective support comparison','',
        'Full-simplex passes are retained in every method; regional counts are additional admissions.',
        'Exact-start empirical evidence and deterministic same-start re-execution. No guarantee of absolute safety or transfer.',
        'Outcomes are undiscounted episode totals, averaged equally over contexts and five predeclared priority profiles, including idle fallback.',
        'Proposed-region volume is uniform-simplex volume, not the fraction of states that are safe.', '',
        '| Method | Admitted / total | Extra regional | CDS / PDS | Context coverage | Region volume | Fallback | Goal success | Mean cost | Effort | Action change | Admitted + goal + episode budget |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in summary.items():
        lines.append(f"| {name} | {r['admitted']}/{r['total']} ({r['admission_rate']:.1%}) | {r['additional_regional']} | {r['cds']}/{r['pds']} | {r['context_coverage']:.1%} | {r['mean_proposed_region_volume']:.1%} | {r['fallback_rate']:.1%} | {r['goal_rate']:.1%} | {r['mean_cost']:.3f} | {r['mean_effort']:.3f} | {r['mean_action_change']:.3f} | {r['admitted_goal_budget_rate']:.1%} |")
    lines+=['','Regional certificates are stored in each context’s `full_plus_mdn_library.json` as `MDN_WX`.',
        'Fixed-region decisions are baseline audit records, not falsely labelled MDN certificates.',
        'Full-simplex certificates remain eligible outside the learned region. Regional certificates require matching context and in-region priorities.',
        'The cost limit is a separate episode-level outcome diagnostic; certification does not enforce it as an absolute safety constraint.',
        'Do not count five priority queries as five independent environment contexts. Further tuning requires fresh final contexts.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('Report:',root/'report.md')


if __name__=='__main__':main()

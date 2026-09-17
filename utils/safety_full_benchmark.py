"""Held-out three-objective policy benchmark; empirical evidence, not a safety guarantee."""
import json
import hashlib
import platform
from importlib.metadata import version, PackageNotFoundError
import subprocess
from pathlib import Path
import numpy as np
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from utils.safety_objectives import FrozenObjectiveScales

METHODS = ('best_admissible', 'random_candidate', 'random_admissible', 'baseline', 'best_unrestricted')


def priorities(count):
    focus = np.array([[.8,.1,.1], [.1,.8,.1], [.1,.1,.8]])
    return {
        'balanced': np.tile([1/3]*3, (count,1)),
        **{name: np.tile(w,(count,1)) for name,w in zip(('safety','task','effort'),focus)},
        'gradual': np.array([(1-t)*focus[0]+t*focus[1] for t in np.linspace(0,1,count)]),
        'abrupt': np.array([focus[0] if i < count//2 else focus[1] for i in range(count)]),
        'empty_library_control': np.tile([1/3]*3,(count,1)),
    }


def make_plan(records, scales, count, seed, epsilon):
    """Freeze all selections before accessing held-out outcomes."""
    if not records or count < 2 or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('Nonempty evidence, count >=2 and nonnegative epsilon required')
    ids = list(records[0]['candidate_skill_ids'])
    if any(np.asarray(r['candidate_motives']).shape != (len(ids),3) or not np.all(np.isfinite(r['candidate_motives'])) for r in records):
        raise ValueError('Finite three-objective evidence required')
    if len(set(ids)) != len(ids) or 'zero_action' not in ids:
        raise ValueError('Unique policy IDs including zero_action required')
    if any(list(r['candidate_skill_ids']) != ids for r in records):
        raise ValueError('Candidate pools differ')
    baseline = ids.index('zero_action')
    means = np.mean([r['candidate_motives'] for r in records],axis=0)
    dn = (means-means[baseline])/scales.divisors
    dr = dn[:,1]  # Separate payoff is the task return, deliberately counted twice.
    candidates = [i for i in range(len(ids)) if i != baseline]
    cds, pds = CDSGate(), PDSGate(epsilon=epsilon)
    audit = []
    admitted = []
    for i in candidates:
        gate = 'CDS' if cds.admit(dr[i],dn[i]) else 'PDS' if pds.admit(dr[i],dn[i]) else None
        if gate:
            admitted.append(i)
        audit.append(dict(skill_id=str(ids[i]), gate=gate, admitted=bool(gate),
            delta_r=float(dr[i]),delta_n=dn[i].tolist(),weight_region_type='FULL_SIMPLEX',
            inequality_value=float(dr[i]+min(dn[i])),epsilon=0. if gate=='CDS' else epsilon,
            reason=None if gate else 'CDS and PDS inequalities failed'))
    rows = []
    rng = np.random.default_rng(seed)
    for scenario, weights in priorities(count).items():
        eligible = [] if scenario == 'empty_library_control' else admitted
        for step,w in enumerate(weights):
            scores = dr+dn@w
            best = lambda pool: max(pool,key=lambda i:(scores[i],str(ids[i]))) if pool else baseline
            chosen = [best(eligible),int(rng.choice(candidates)) if candidates else baseline,
                      int(rng.choice(eligible)) if eligible else baseline,baseline,best(candidates)]
            for method,i in zip(METHODS,chosen):
                threshold = 0. if cds.admit(dr[i],dn[i]) else -epsilon
                rows.append(dict(scenario=scenario,step=step,method=method,selected_skill_id=str(ids[i]),
                    weights=w.tolist(),evidence_score=float(scores[i]),threshold=float(threshold),
                    selection_valid=bool(scores[i]>=threshold-1e-9),
                    selected_admitted=i in eligible,eligible_count=len(eligible),
                    stored_count=len(admitted),candidate_count=len(candidates),
                    reuse_eligibility=len(eligible)/len(admitted) if admitted else None,
                    abstained=method in ('best_admissible','random_admissible') and not eligible,
                    fallback=i==baseline))
    return dict(audit=audit,rows=rows,policy_ids=list(map(str,ids)))


def evaluate(plan, records, scales):
    rows=[]
    for selection in plan['rows']:
        record=records[selection['step']]
        ids=list(record['candidate_skill_ids'])
        if list(map(str,ids)) != plan['policy_ids']:
            raise ValueError('Evaluation candidate pool differs from evidence')
        raw=np.asarray(record['candidate_motives'])
        i,b=ids.index(selection['selected_skill_id']),ids.index('zero_action')
        delta=(raw[i]-raw[b])/scales.divisors
        score=float(delta[1]+np.dot(selection['weights'],delta))
        rows.append({**selection,'context_seed':int(record['context_seed']),
                     'raw_objective_returns':raw[i].tolist(),'objective_improvements':delta.tolist(),
                     'execution_score':score,'execution_threshold_violation':score < selection['threshold']-1e-9})
    return rows


def summarize(rows):
    output=[]
    for scenario in priorities(2):
        for method in METHODS:
            group=[r for r in rows if r['scenario']==scenario and r['method']==method]
            by_seed=[]
            for seed in sorted(set(r['batch_seed'] for r in group)):
                sample=[r for r in group if r['batch_seed']==seed]
                reference={r['step']:r for r in rows if r['batch_seed']==seed and r['scenario']==scenario and r['method']=='best_admissible'}
                by_seed.append(dict(seed=seed,mean_score=float(np.mean([r['execution_score'] for r in sample])),
                    paired_difference_vs_subrep=float(np.mean([r['execution_score']-reference[r['step']]['execution_score'] for r in sample])),
                    mean_raw_objectives=np.mean([r['raw_objective_returns'] for r in sample],axis=0).tolist()))
            output.append(dict(scenario=scenario,method=method,seed_results=by_seed,
                mean_score=float(np.mean([s['mean_score'] for s in by_seed])),
                seed_score_std=float(np.std([s['mean_score'] for s in by_seed],ddof=1)) if len(by_seed)>1 else None,
                abstention_rate=float(np.mean([r['abstained'] for r in group])),
                selection_validity=float(np.mean([r['selection_valid'] for r in group])),
                execution_violation_rate=float(np.mean([r['execution_threshold_violation'] for r in group]))))
    return output


def run_benchmark(output, *, contexts=20, batches=5, max_steps=200, seed=2000,
                  epsilon=0., env_id='SafetyPointGoal1-v0', collector_class=None,
                  ppo_checkpoint=None, ppo_lagrangian_checkpoint=None):
    if contexts<2 or batches<1 or max_steps<1 or not np.isfinite(epsilon) or epsilon<0:
        raise ValueError('Require contexts >=2, batches >=1, max_steps >=1, epsilon >=0')
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    if collector_class is None:
        from data_collector.collect_safety_gymnasium_rollouts import SafetyGymnasiumRolloutCollector
        collector_class=SafetyGymnasiumRolloutCollector
    checkpoints={}
    for name,path in [('ppo_checkpoint',ppo_checkpoint),('ppo_lagrangian_checkpoint',ppo_lagrangian_checkpoint)]:
        if path:
            checkpoints[name]={'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    except subprocess.CalledProcessError:
        commit,dirty=None,None
    dependencies={}
    for package in ('safety-gymnasium','gymnasium','mujoco','numpy','torch'):
        try:
            dependencies[package]=version(package)
        except PackageNotFoundError:
            dependencies[package]=None
    manifest=dict(dependencies=dependencies,contexts=contexts,batches=batches,max_steps=max_steps,seed=seed,epsilon=epsilon,
        env_id=env_id,checkpoints=checkpoints,commit=commit,dirty=dirty,python=platform.python_version(),
        platform=platform.platform(),numpy=np.__version__,gamma=.99,
        evidence='Real environment only when using the real collector; tests use a fake simulator.',
        limitations=['Certification uses empirical mean evidence, not a statistical guarantee on new episodes.',
                     'Priority changes occur between independent episodes, not within an episode.',
                     'Empty-library scenario deliberately disables all stored skills; it is a control.',
                     'Candidate policies are fixed; batches vary environment seeds, not training seeds.'])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def collect(folder,start,scales_file=None):
        collector=collector_class(objectives=3,save_dir=str(root/folder),seed=start,max_steps=max_steps,
            scales_file=scales_file,env_id=env_id,ppo_checkpoint=ppo_checkpoint,
            ppo_lagrangian_checkpoint=ppo_lagrangian_checkpoint)
        try:
            return collector.collect(contexts)
        finally:
            collector.close()
    development=collect('development',seed)
    delta=[]
    for r in development:
        ids=list(r['candidate_skill_ids']);b=ids.index('zero_action')
        delta.extend(r['candidate_motives'][i]-r['candidate_motives'][b] for i in range(len(ids)) if i!=b)
    scales=FrozenObjectiveScales.fit_development(delta,source=f'development seeds {seed+1}-{seed+contexts}')
    scales_path=root/'frozen_scales.json';scales.save(scales_path)
    all_rows=[]
    admissions=[]
    for batch in range(batches):
        evidence_seed=seed+(2*batch+1)*(contexts+1)
        eval_seed=seed+(2*batch+2)*(contexts+1)
        evidence=collect(f'batch_{batch}/evidence',evidence_seed,str(scales_path))
        plan=make_plan(evidence,scales,contexts,evidence_seed,epsilon)
        admitted=sum(a['admitted'] for a in plan['audit'])
        admissions.append(dict(batch_seed=evidence_seed,admitted=admitted,rejected=len(plan['audit'])-admitted,
            admission_rate=admitted/len(plan['audit']) if plan['audit'] else None))
        # Persist decisions before collecting any held-out outcomes.
        (root/f'batch_{batch}/selection_plan.json').write_text(json.dumps(plan,indent=2))
        evaluation=collect(f'batch_{batch}/evaluation',eval_seed,str(scales_path))
        rows=[{**r,'batch_seed':evidence_seed} for r in evaluate(plan,evaluation,scales)]
        (root/f'batch_{batch}/results.json').write_text(json.dumps(rows,indent=2))
        all_rows.extend(rows)
    summary=summarize(all_rows)
    (root/'admission_summary.json').write_text(json.dumps(admissions,indent=2))
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Three-objective held-out benchmark','',
           'Scores retain task payoff plus weighted objectives. Execution violations are measured, not hidden.',
           'Raw objective vectors and paired seed differences are in JSON; admission audits are in each selection plan.','',
           '| Scenario | Method | Mean score | Seed SD | Abstention | Execution violation |',
           '|---|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['scenario']} | {r['method']} | {r['mean_score']:.4f} | {r['seed_score_std']} | {r['abstention_rate']:.3f} | {r['execution_violation_rate']:.3f} |")
    lines+=['','## Limitations','']+['- '+x for x in manifest['limitations']]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return summary

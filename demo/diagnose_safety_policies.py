"""Read-only outcome audit, or explicitly requested fresh development executions."""
import argparse
import json
from pathlib import Path
import numpy as np
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate


def validate_task_payoff(payoffs, motives, *, source="record"):
    # Scalar totals accumulate in Python float; motive totals in float32.
    # Cancellation near zero makes the default absolute tolerance too strict.
    payoff = np.asarray(payoffs, dtype=float)
    task = np.asarray(motives, dtype=float)[:, 1]
    if payoff.shape != task.shape or not np.isfinite(payoff).all() or not np.isfinite(task).all():
        raise ValueError(f"{source}: invalid payoff/task measurements")
    error = float(np.max(np.abs(payoff - task)))
    if not np.allclose(payoff, task, rtol=1e-5, atol=2e-6):
        raise ValueError(f"{source}: payoff differs from task motive; max absolute difference={error:.9g}")
    return error


def audit_arrays(names, values, scales, epsilon):
    values, scales = np.asarray(values, float), np.asarray(scales, float)
    if values.ndim != 3 or values.shape[1:] != (len(names), 3) or not np.isfinite(values).all():
        raise ValueError('Expected finite context x policy x three-objective outcomes')
    if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Invalid frozen scales')
    if len(set(names)) != len(names) or names.count('zero_action') != 1:
        raise ValueError('Unique policies and exactly one zero_action baseline required')
    delta = (values - values[:, names.index('zero_action'), None, :]) / scales
    rows, details = [], []
    for j, name in enumerate(names):
        cds, pds = [], []
        for i, dn in enumerate(delta[:, j]):
            dr = float(dn[1])
            c = CDSGate().admit(dr, dn)
            p = PDSGate(epsilon).admit(dr, dn)
            cds.append(c); pds.append(p)
            details.append(dict(context_index=i, policy=name, delta_r=dr, delta_n=dn.tolist(),
                cds_margin=float(CDSGate().get_admission_margin(dr, dn)),
                pds_margin=float(PDSGate(epsilon).get_admission_margin(dr, dn)),
                result='CDS' if c else 'PDS' if p else 'rejected',
                worst_objective=['safety','task','efficiency'][int(np.argmin(dn))]))
        rows.append(dict(policy=name, contexts=len(values), mean_task=float(values[:,j,1].mean()),
            mean_cost=float(-values[:,j,0].mean()), mean_effort=float(-values[:,j,2].mean()),
            task_positive_rate=float(np.mean(values[:,j,1]>0)),
            cds_passes=sum(map(int,cds)), pds_only_passes=sum(int(p and not c) for p,c in zip(pds,cds)),
            rejected=sum(int(not p and not c) for p,c in zip(pds,cds))))
    return rows, details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', action='append', default=[])
    p.add_argument('--scales-file', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--fresh', action='store_true')
    p.add_argument('--seed', type=int, default=160000)
    p.add_argument('--contexts', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--epsilon', type=float, default=.2)
    args=p.parse_args()
    if not args.data_dir and not args.fresh:
        p.error('Supply --data-dir or --fresh')
    if min(args.contexts,args.max_steps,args.repeats)<1 or not np.isfinite(args.epsilon) or args.epsilon<0:
        p.error('Invalid counts or epsilon')
    root=Path(args.output)
    if root.exists(): raise FileExistsError('Use a new output directory')
    scales=np.asarray(json.loads(Path(args.scales_file).read_text())['divisors'])
    report={'scope':'Development diagnostics; no MDN support evaluation or training.',
            'scales':scales.tolist(),'epsilon':args.epsilon,'datasets':[], 'seed_overlaps':[]}
    seed_sets=[]
    all_details=[]
    for folder in args.data_dir:
        files=sorted(Path(folder).glob('*.npz'))
        if not files: raise ValueError(f'No NPZ files in {folder}')
        values=[]; seeds=[]; steps=[]; names=None; mismatches=0; payoff_error=0.0
        for file in files:
            with np.load(file,allow_pickle=False) as d:
                current=d['candidate_skill_ids'].astype(str).tolist()
                if names is not None and names!=current: raise ValueError('Policy order mismatch')
                names=current
                if not bool(d['values_are_raw']): raise ValueError('Expected raw measurements')
                payoff_error=max(payoff_error, validate_task_payoff(
                    d['candidate_payoffs'],d['candidate_motives'],source=str(file)))
                mismatches+=int(not np.allclose(d['objective_scales'],scales))
                values.append(d['candidate_motives']); seeds.append(int(d['context_seed'])); steps.extend(d['step_counts'].tolist())
        rows,details=audit_arrays(names,values,scales,args.epsilon)
        for detail in details: detail.update(dataset=folder,context_seed=seeds[detail['context_index']])
        all_details.extend(details)
        report['datasets'].append(dict(source=folder,contexts=len(files),duplicate_seed_records=len(seeds)-len(set(seeds)),
            payoff_task_max_absolute_difference=payoff_error,
            payoff_task_tolerance=dict(rtol=1e-5,atol=2e-6),scale_metadata_mismatches=mismatches,steps_min=min(steps),steps_max=max(steps),policies=rows,
            goals='Not recorded in these NPZ files; cannot infer success from returns.'))
        seed_sets.append((folder,set(seeds)))
    for i,(left,seeds) in enumerate(seed_sets):
        for right,other in seed_sets[i+1:]:
            report['seed_overlaps'].append(dict(left=left,right=right,count=len(seeds & other)))
    if args.fresh:
        from utils.safety_reuse_benchmark import Config, Collector, load_models
        config=Config(output=str(root),training_seeds=(42,),max_steps=args.max_steps)
        config.validate()
        models,provenance=load_models(config)
        fresh_seeds=list(range(args.seed+1,args.seed+args.contexts+1))
        if any(set(fresh_seeds)&seeds for _,seeds in seed_sets): raise ValueError('Fresh seeds overlap supplied data')
        collector=Collector(config,models[42])
        try:
            records=[r for repeat in range(args.repeats) for r in collector.collect(fresh_seeds,repeat=repeat)]
        finally: collector.close()
        names=list(records[0]['outcomes'])
        rows,details=audit_arrays(names,[[r['outcomes'][n]['motives'] for n in names] for r in records],scales,args.epsilon)
        for row in rows:
            outcomes=[r['outcomes'][row['policy']] for r in records]
            row.update(goal_reached_rate=float(np.mean([o['goals']>0 for o in outcomes])),
                total_goals=sum(o['goals'] for o in outcomes),mean_steps=float(np.mean([o['steps'] for o in outcomes])),
                stop_reasons={reason:sum(o['stop_reason']==reason for o in outcomes) for reason in sorted({o['stop_reason'] for o in outcomes})})
        report['fresh']=dict(checkpoints=provenance,contexts=args.contexts,repeats=args.repeats,
            max_steps=args.max_steps,policies=rows, note='Repeated deterministic executions are not independent samples.')
        for detail in details:
            record=records[detail['context_index']]
            detail.update(dataset='fresh',context_seed=record['context_seed'],repeat=record['repeat'])
        all_details.extend(details)
    root.mkdir(parents=True)
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    (root/'certification_details.json').write_text(json.dumps(all_details,indent=2))
    if args.fresh: (root/'executions.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(report,indent=2))
    print('Saved reports:',root)


if __name__=='__main__': main()

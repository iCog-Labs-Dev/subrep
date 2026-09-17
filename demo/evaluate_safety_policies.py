"""Compare checkpoint policies on fresh development contexts, without admission filtering."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def summarize(records):
    ids=list(records[0]['candidate_skill_ids'])
    if any(list(r['candidate_skill_ids']) != ids for r in records):
        raise ValueError('Policy order changed')
    values=np.asarray([r['candidate_motives'] for r in records],dtype=float)
    if values.shape != (len(records),len(ids),3) or not np.isfinite(values).all():
        raise ValueError('Finite three-objective outcomes required')
    baseline=ids.index('zero_action')
    output=[]
    for i,name in enumerate(ids):
        # Positive costs/effort are easier to read. All returns are discounted.
        measurements=values[:,i]*[-1,1,-1]
        delta=values[:,i]-values[:,baseline]
        output.append(dict(policy=str(name),contexts=len(records),
            mean_safety_cost=float(measurements[:,0].mean()),
            mean_task_return=float(measurements[:,1].mean()),
            mean_control_effort=float(measurements[:,2].mean()),
            std_safety_task_effort=np.std(measurements,axis=0,ddof=1).tolist() if len(records)>1 else None,
            mean_improvement_safety_task_efficiency=delta.mean(axis=0).tolist(),
            task_better_than_idle_rate=float(np.mean(delta[:,1]>0)),
            positive_safety_cost_rate=float(np.mean(measurements[:,0]>0))))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoint',action='append',help='Repeat for a chosen subset; otherwise all models/safety_ppo*.pt')
    p.add_argument('--contexts',type=int,default=30)
    p.add_argument('--seed',type=int,default=30000)
    p.add_argument('--max-steps',type=int,default=200)
    args=p.parse_args()
    if args.contexts<2 or args.max_steps<1:
        p.error('Require at least two contexts and positive max-steps')
    paths=[Path(x).resolve() for x in args.checkpoint] if args.checkpoint else sorted(Path('models').glob('safety_ppo*.pt'))
    if not paths or any(not x.is_file() for x in paths):
        p.error('No checkpoints found or a requested file is missing')
    if len({x.stem for x in paths})!=len(paths):
        p.error('Checkpoint filenames must have unique stems')
    from data_collector.collect_safety_gymnasium_rollouts import SafetyGymnasiumRolloutCollector, SafetyCandidatePolicy
    from pilot.safety_gymnasium_ppo import SafetyPPOPilot
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    collector=SafetyGymnasiumRolloutCollector(objectives=3,save_dir=str(root/'rollouts'),
        seed=args.seed,max_steps=args.max_steps)
    try:
        policies=list(collector.candidate_policies); provenance=[]
        for path in paths:
            model=SafetyPPOPilot.load(path,map_location='cpu')
            if model.observation_dim != int(np.prod(collector.env.observation_space.shape)):
                raise ValueError(f'Incompatible observation dimensions: {path}')
            if model.action_dim != int(np.prod(collector.env.action_space.shape)):
                raise ValueError(f'Incompatible action dimensions: {path}')
            if not np.allclose(model.action_low.detach().cpu().numpy(),collector.env.action_space.low) or not np.allclose(model.action_high.detach().cpu().numpy(),collector.env.action_space.high):
                raise ValueError(f'Incompatible action bounds: {path}')
            policies.append(SafetyCandidatePolicy(path.stem,lambda obs,m=model:m.predict(obs,deterministic=True)))
            provenance.append(dict(policy=path.stem,path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        collector.candidate_policies=tuple(policies)
        manifest=dict(purpose='development policy comparison; not final evaluation',seed=args.seed,
            context_seeds=list(range(args.seed+1,args.seed+args.contexts+1)),max_steps=args.max_steps,
            environment=collector.env_id,gamma=collector.gamma,checkpoints=provenance,
            measurement='Discounted cumulative safety cost, task return, and normalized squared-action effort. No objective scaling or certification.')
        (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
        rows=summarize(collector.collect(args.contexts))
    finally:
        collector.close()
    (root/'summary.json').write_text(json.dumps(rows,indent=2))
    lines=['# Development policy evaluation','',
        'Same context seeds for all policies. Higher task return is better; lower cost and effort are better.',
        'All measurements are discounted cumulative returns. No admission filtering or composite ranking.',
        'Use these results for development decisions only; reserve fresh seeds for final evaluation.','',
        '| Policy | Task return | Safety cost | Control effort | Task better than idle |',
        '|---|---:|---:|---:|---:|']
    lines += [f"| {r['policy']} | {r['mean_task_return']:.4f} | {r['mean_safety_cost']:.4f} | {r['mean_control_effort']:.4f} | {r['task_better_than_idle_rate']:.1%} |" for r in rows]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('Complete:',root/'report.md')


if __name__=='__main__':
    main()

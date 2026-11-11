"""Collect four-objective candidate NPZs with frozen existing policies.

Also export matched three-objective projections for controlled dimension scaling.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from utils.safety_reuse_benchmark import Config, Collector, load_models, fit_scales, write_json


def export(records, out, gamma):
    names4=['Safety','Task','ControlEfficiency','ActionSmoothness']
    definitions=['negative environment cost','environment reward','negative mean squared normalized action',
                 'negative mean squared half-difference of normalized consecutive applied actions; first step zero']
    for dim in (3,4):
        folder=out/f'objectives{dim}';folder.mkdir(parents=True,exist_ok=False)
        for record in records:
            names=list(record['outcomes'])
            values=np.array([record['outcomes'][n]['motives'][:dim] for n in names])
            np.savez_compressed(folder/f"context_{record['context_seed']}.npz",context=np.asarray(record['initial_observation'],dtype=np.float32),
                candidate_skill_ids=np.asarray(names),candidate_motives=values,candidate_payoffs=values[:,1],
                context_seed=record['context_seed'],gamma=gamma,env_id='SafetyPointGoal1-v0',
                objective_names=np.asarray(names4[:dim]),measurement_definitions=np.asarray(definitions[:dim]),
                values_are_raw=True,objective_scales=np.ones(dim),scaling_source='raw',schema_version=2 if dim==3 else 3)
        scales=fit_scales([records])[:dim]
        write_json(out/f'scales{dim}.json',dict(schema_version=1,objective_names=names4[:dim],divisors=scales.tolist(),
            development_source=str(folder.resolve()),estimator='mean_absolute_development_delta_return_zero_fallback_1'))
    write_json(out/'benchmark_contract4.json',dict(environment='SafetyPointGoal1-v0',observation_schema='native_flat_v1',
        objectives=['safety','task','control_efficiency','action_smoothness']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--contexts',type=int,default=300)
    p.add_argument('--seed',type=int,default=410000);p.add_argument('--training-seed',type=int,default=42)
    p.add_argument('--max-steps',type=int,default=200)
    p.add_argument('--ppo-checkpoint',required=True);p.add_argument('--lagrangian-checkpoint',required=True)
    a=p.parse_args();out=Path(a.output)
    config=Config(str(out),training_seeds=(a.training_seed,),ppo_template=a.ppo_checkpoint,
        lagrangian_template=a.lagrangian_checkpoint,contexts=a.contexts,seed=a.seed,max_steps=a.max_steps,objectives=4)
    config.validate()
    if out.exists():raise FileExistsError(out)
    models,provenance=load_models(config)
    collector=Collector(config,models[a.training_seed])
    try:
        out.mkdir(parents=True)
        write_json(out/'manifest.json',dict(config=vars(config),checkpoints=provenance,role='training/development only; reserve different seeds for evaluation'))
        records=collector.collect(range(a.seed+1,a.seed+a.contexts+1))
        write_json(out/'evidence.json',records)
        export(records,out,config.gamma)
    finally:collector.close()
    print('Saved training/development data:',out)


if __name__=='__main__':main()

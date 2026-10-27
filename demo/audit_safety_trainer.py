"""Inspect checkpoints on saved starting observations; no training or environment runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pilot.safety_gymnasium_ppo import SafetyPPOPilot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--checkpoint',action='append',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): raise FileExistsError('Use a new report path')
    files=sorted(Path(args.data_dir).glob('*.npz'))
    if not files: raise ValueError('No saved contexts found')
    observations=[]; environments=set()
    for path in files:
        with np.load(path,allow_pickle=False) as data:
            observations.append(data['context'])
            environments.add(str(data['env_id'].item()))
    x=torch.tensor(np.stack(observations),dtype=torch.float32)
    results=[]
    for filename in args.checkpoint:
        payload=torch.load(filename,map_location='cpu',weights_only=False)
        model=SafetyPPOPilot.load(filename)
        metadata=payload.get('metadata',{})
        if environments != {metadata.get('environment')}:
            raise ValueError(f'Checkpoint/data environment mismatch: {filename}')
        if x.shape[1]!=model.observation_dim: raise ValueError('Observation dimension mismatch')
        if not all(torch.isfinite(t).all() for t in model.state_dict().values()):
            raise ValueError('Nonfinite checkpoint parameters')
        with torch.no_grad():
            mean,std,value=model(x)
            span=model.action_high-model.action_low
            near=((mean-model.action_low)/span < .01) | ((model.action_high-mean)/span < .01)
            distribution,_=model.distribution(x)
            clipping=distribution.cdf(model.action_low)+1-distribution.cdf(model.action_high)
        results.append(dict(checkpoint=filename,sha256=hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
            metadata=metadata,contexts=len(files),
            mean_action_by_dimension=mean.mean(0).tolist(),
            action_std_across_contexts=mean.std(0,unbiased=False).tolist(),
            near_bound_fraction_by_dimension=near.float().mean(0).tolist(),
            exploration_std_by_dimension=std.mean(0).tolist(),
            expected_sampling_clip_fraction_by_dimension=clipping.mean(0).tolist(),
            critic_value_min=float(value.min()),critic_value_max=float(value.max())))
    report=dict(scope='Checkpoint audit at saved initial observations only. No trajectories, training curves, or proof of policy competence.',
        observation='Raw observation path; dimensions cannot verify historical observation semantics.',results=results)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    print('Saved:',output)


if __name__=='__main__': main()

"""Collect development/evidence/held-out rollouts and compare five selection methods."""
import argparse
from pathlib import Path
from utils.safety_full_benchmark import run_benchmark


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,help='New directory; existing paths are rejected')
    p.add_argument('--contexts',type=int,default=20)
    p.add_argument('--batches',type=int,default=5)
    p.add_argument('--max-steps',type=int,default=200)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--preset',choices=['simple','ppo-pds'],default='simple')
    p.add_argument('--epsilon',type=float,default=None)
    p.add_argument('--env-id',default='SafetyPointGoal1-v0')
    p.add_argument('--ppo-checkpoint')
    p.add_argument('--ppo-lagrangian-checkpoint')
    args=vars(p.parse_args(argv))
    preset=args.pop('preset')
    if args['epsilon'] is None:
        args['epsilon']=.2 if preset=='ppo-pds' else 0.
    if preset=='ppo-pds' and args['ppo_checkpoint'] is None:
        args['ppo_checkpoint']=str(Path(__file__).resolve().parents[1]/'models/safety_ppo_point_goal_seed42_updates50.pt')
    for key in ('ppo_checkpoint','ppo_lagrangian_checkpoint'):
        if args[key] and not Path(args[key]).is_file():
            p.error(f'Checkpoint does not exist: {args[key]}')
    return args


def main():
    args=parse_args()
    print('PPO checkpoint:',args['ppo_checkpoint'], 'PDS epsilon:',args['epsilon'],flush=True)
    run_benchmark(**args)
    print('Complete. Review report.md, summary.json, manifest.json and batch results in',args['output'])


if __name__=='__main__':
    main()

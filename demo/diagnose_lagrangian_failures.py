"""Replay costly development episodes; diagnose behavior without retraining."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def summarize_steps(steps, gamma):
    costs = np.array([s['cost'] for s in steps], dtype=float)
    longest = streak = 0
    for cost in costs:
        streak = streak + 1 if cost > 0 else 0
        longest = max(longest, streak)
    positive = np.flatnonzero(costs > 0)
    return dict(steps=len(steps), total_cost=float(costs.sum()),
        discounted_cost=float(np.dot(costs, gamma ** np.arange(len(costs)))),
        total_return=float(sum(s['reward'] for s in steps)),
        goals=sum(s['goal_met'] for s in steps),
        cost_steps=int(len(positive)), longest_cost_streak=longest,
        first_cost_step=int(positive[0]+1) if len(positive) else None,
        second_half_cost=float(costs[len(costs)//2:].sum()))


def select_episodes(records, controls):
    seeds = [int(r['seed']) for r in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError('Episode seeds must be unique')
    return ([r for r in records if r['cost'] > 0] +
            [r for r in records if r['cost'] == 0][:controls])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--controls', type=int, default=5)
    args = parser.parse_args()
    if args.controls < 0:
        parser.error('--controls must be nonnegative')
    import torch
    import safety_gymnasium
    from pilot.safety_gymnasium_ppo import SafetyPPOPilot

    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location='cpu')
    metadata = payload['metadata']
    if metadata.get('trainer_version') not in ('separate_reward_cost_critics_v2', 'finite_episode_cost_v3'):
        raise ValueError('This objective audit requires a versioned separate-critic checkpoint')
    config = metadata['config']
    horizon, gamma = int(config['max_episode_steps']), float(config['gamma'])
    records = json.loads(Path(args.episodes_file).read_text())
    selected = select_episodes(records, args.controls)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    model = SafetyPPOPilot.load(checkpoint, map_location='cpu')
    env = safety_gymnasium.make(metadata['environment'])
    results = []
    try:
        if model.observation_dim != int(np.prod(env.observation_space.shape)):
            raise ValueError('Observation dimensions differ')
        if not (np.allclose(model.action_low.numpy(), env.action_space.low.reshape(-1)) and
                np.allclose(model.action_high.numpy(), env.action_space.high.reshape(-1))):
            raise ValueError('Action bounds differ')
        for record in selected:
            seed = int(record['seed'])
            obs, _ = env.reset(seed=seed)
            steps = []
            for step in range(horizon):
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                with torch.no_grad():
                    distribution, value = model.distribution(torch.as_tensor(obs))
                    clip_probability = (distribution.cdf(model.action_low) +
                        1 - distribution.cdf(model.action_high)).numpy().tolist()
                action = model.predict(obs, deterministic=True)
                next_obs, reward, cost, terminated, truncated, info = env.step(action)
                steps.append(dict(step=step+1, observation=obs.tolist(), action=action.tolist(),
                    next_observation=np.asarray(next_obs).reshape(-1).tolist(),
                    reward=float(reward), cost=float(cost), goal_met=bool(info.get('goal_met', False)),
                    terminated=bool(terminated), truncated=bool(truncated),
                    critic_value=float(value), expected_sampling_clip_probability=clip_probability,
                    cost_components={k: float(v) for k,v in info.items()
                        if k.startswith('cost') and np.isscalar(v) and isinstance(v, (int,float,np.number))}))
                obs = next_obs
                if terminated or truncated:
                    break
            row = dict(seed=seed, **summarize_steps(steps, gamma))
            row['previous_cost'] = float(record['cost'])
            row['reproduces_previous_outcome'] = bool(
                np.isclose(row['total_cost'], record['cost']) and
                np.isclose(row['total_return'], record['return'], atol=1e-5) and
                row['goals'] == record['goals'])
            (root/f'episode_{seed}.json').write_text(json.dumps(steps, indent=2, allow_nan=False))
            results.append(row)
            report = dict(scope='Selected costly episodes plus zero-cost controls; not a population pass-rate estimate.',
                checkpoint=str(checkpoint), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                metadata=metadata, cost_objective_audit=dict(
                    training_cost_discount=1.0 if metadata['trainer_version']=='finite_episode_cost_v3' else gamma, evaluation_cost_discount=1.0,
                    evaluation_horizon=horizon,
                    training_time_limit_bootstrap=metadata['trainer_version']!='finite_episode_cost_v3',
                    note='v2 uses discounted continuing cost; v3 uses undiscounted episode cost with remaining time. The multiplier uses raw completed episode cost in both.'),
                episodes=results)
            (root/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps(row), flush=True)
    finally:
        env.close()
    print('Saved:', root/'summary.json')


if __name__ == '__main__':
    main()

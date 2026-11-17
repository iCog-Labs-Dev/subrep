"""Screen exact reset layouts, then verify with disjoint policy-noise seeds.

Empirical sampled-policy suitability only; not SubRep certification or transfer.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pilot.safety_gymnasium_ppo import SafetyPPOPilot
from utils.safety_objectives import action_change, control_effort


def summarize(episodes, cost_limit, goal_rate):
    costs = [e['cost'] for e in episodes]
    success = float(np.mean([e['goals'] > 0 for e in episodes]))
    cost = float(np.mean(costs))
    return dict(episodes=len(episodes), mean_cost=cost, goal_rate=success,
                mean_effort=float(np.mean([e['effort'] for e in episodes])),
                mean_action_change=float(np.mean([e['action_change'] for e in episodes])),
                empirical_pass=bool(cost <= cost_limit and success >= goal_rate))


def screen_then_verify(contexts, run, screen_episodes, verify_episodes, cost_limit, goal_rate):
    records = []
    # Complete screening before verification; verification never changes selection.
    for index, context in enumerate(contexts):
        episodes = run(context, index, 'screen', screen_episodes)
        records.append(dict(context_seed=context, screen= summarize(episodes, cost_limit, goal_rate),
                            screen_episodes=episodes, verification=None))
    for index, record in enumerate(records):
        if record['screen']['empirical_pass']:
            episodes = run(record['context_seed'], index, 'verify', verify_episodes)
            record['verification'] = summarize(episodes, cost_limit, goal_rate)
            record['verification_episodes'] = episodes
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--env-id', default='SafetyPointGoal1-v0')
    parser.add_argument('--context-seed', type=int, default=460000)
    parser.add_argument('--contexts', type=int, default=30)
    parser.add_argument('--screen-episodes', type=int, default=5)
    parser.add_argument('--verify-episodes', type=int, default=100)
    parser.add_argument('--noise-seed', type=int, default=730000)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--cost-limit', type=float, default=1.)
    parser.add_argument('--goal-rate', type=float, default=.9)
    args = parser.parse_args()
    if min(args.contexts, args.screen_episodes, args.verify_episodes, args.max_steps) < 1:
        parser.error('Counts must be positive')
    if not np.isfinite(args.cost_limit) or args.cost_limit < 0 or not 0 <= args.goal_rate <= 1:
        parser.error('Invalid cost limit or goal rate')
    if min(args.context_seed, args.noise_seed) < 0:
        parser.error('Seeds must be nonnegative')
    root = Path(args.output)
    if root.exists():
        raise FileExistsError('Use a new output directory')
    checkpoint = Path(args.checkpoint)
    payload = torch.load(checkpoint, map_location='cpu')
    if payload.get('metadata', {}).get('environment') != args.env_id:
        raise ValueError('Checkpoint environment missing or mismatched')
    model = SafetyPPOPilot.load(checkpoint, map_location='cpu')
    model.eval()
    import safety_gymnasium
    env = safety_gymnasium.make(args.env_id)
    root.mkdir(parents=True)
    manifest = dict(vars(args), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    scope='Exact reset layout; sampled policy; fresh action noise in verification. '
                          'Empirical thresholds, not confidence-certified safety or unseen-layout transfer.')
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    initial_observations = {}

    def run(context, index, phase, count):
        # Globally unique, disjoint noise seeds across contexts and phases.
        offset = index * (args.screen_episodes + args.verify_episodes)
        if phase == 'verify':
            offset += args.screen_episodes
        episodes = []
        for episode in range(count):
            noise_seed = args.noise_seed + offset + episode
            obs, _ = env.reset(seed=context)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            if context in initial_observations:
                np.testing.assert_array_equal(obs, initial_observations[context])
            else:
                initial_observations[context] = obs.copy()
            cost = effort = change = reward_total = 0.
            goals = 0
            previous = None
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.manual_seed(noise_seed)
                for step in range(args.max_steps):
                    action = model.predict(obs, deterministic=False)
                    effort += control_effort(action, env.action_space.low, env.action_space.high)
                    change += action_change(action, previous, env.action_space.low, env.action_space.high)
                    previous = np.array(action, copy=True)
                    obs, reward, c, terminated, truncated, info = env.step(action)
                    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                    cost += float(c)
                    reward_total += float(reward)
                    goals += int(bool(info.get('goal_met', False)))
                    if terminated or truncated:
                        break
            episodes.append(dict(noise_seed=noise_seed, cost=cost, goals=goals,
                                 effort=effort, action_change=change, reward=reward_total, steps=step+1))
        (root/f'{context}_{phase}.json').write_text(json.dumps(episodes, indent=2))
        print(context, phase, summarize(episodes, args.cost_limit, args.goal_rate), flush=True)
        return episodes

    try:
        records = screen_then_verify(list(range(args.context_seed, args.context_seed+args.contexts)), run,
                                    args.screen_episodes, args.verify_episodes, args.cost_limit, args.goal_rate)
    finally:
        env.close()
    selected = [r for r in records if r['verification'] is not None]
    passed = [r for r in selected if r['verification']['empirical_pass']]
    result = dict(manifest=manifest, screened=len(records), selected=len(selected),
                  verification_empirical_pass=len(passed), records=records,
                  initial_observations={str(k): v.tolist() for k,v in initial_observations.items()})
    (root/'summary.json').write_text(json.dumps(result, indent=2))
    lines = ['# Exact-context sampled-policy screening', '', manifest['scope'], '',
             f'Screened: {len(records)}; selected: {len(selected)}; verification empirical passes: {len(passed)}.', '',
             '| Context seed | Verification goal rate | Mean cost | Empirical thresholds met |',
             '|---|---:|---:|---|']
    for r in selected:
        v = r['verification']
        lines.append(f"| {r['context_seed']} | {v['goal_rate']:.1%} | {v['mean_cost']:.3f} | {v['empirical_pass']} |")
    lines += ['', 'A pass describes these finite samples only. It is not a statistical safety guarantee.',
              'Do not use verification episodes to tune thresholds or retrain. Further tuned claims need new evidence.',
              'These results do not authorize deterministic execution or bypass SubRep certification.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('Report:', root/'report.md')


if __name__ == '__main__':
    main()

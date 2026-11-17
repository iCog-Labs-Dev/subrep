"""Measure bounded-mean saturation and action/log-probability consistency.

Diagnostic only. Saturation is an optimization hypothesis, not proof of failure.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pilot.safety_gymnasium_ppo import SafetyPPOPilot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=470000)
    a = p.parse_args()
    if min(a.episodes, a.max_steps) < 1:
        p.error('Counts must be positive')
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Use a new output directory')
    path = Path(a.checkpoint)
    payload = torch.load(path, map_location='cpu')
    env_id = payload.get('metadata', {}).get('environment')
    if not env_id:
        raise ValueError('Checkpoint lacks environment metadata')
    model = SafetyPPOPilot.load(path, map_location='cpu')
    model.eval()
    import safety_gymnasium
    env = safety_gymnasium.make(env_id)
    derivatives, clipped_flags, errors, outcomes = [], [], [], []
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(a.seed)
            for episode in range(a.episodes):
                obs, _ = env.reset(seed=a.seed+episode)
                cost = goals = 0
                for step in range(a.max_steps):
                    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                    logits = model.actor_mean(model.backbone(torch.tensor(obs)))
                    derivatives.append(((1-torch.tanh(logits).square()) if model.actor_mean_type == "tanh" else torch.ones_like(logits)).numpy())
                    action, raw, old_logp, _ = model.act(obs, return_raw=True)
                    logp, _, _ = model.evaluate_actions(torch.tensor(obs), torch.tensor(raw))
                    errors.append(abs(logp.item()-old_logp))
                    clipped_flags.append(raw != action)
                    obs, _, c, terminated, truncated, info = env.step(action)
                    cost += float(c)
                    goals += int(bool(info.get('goal_met', False)))
                    if terminated or truncated:
                        break
                outcomes.append(dict(seed=a.seed+episode, cost=cost, goals=goals, steps=step+1))
    finally:
        env.close()
    d = np.asarray(derivatives)
    report = dict(config=vars(a), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        environment=env_id, actor_mean_type=model.actor_mean_type, steps=len(d),
        mean_unit_mean_derivative_by_action=d.mean(0).tolist(),
        fraction_derivative_below_001_by_action=(d < .01).mean(0).tolist(),
        fraction_actions_clipped_by_action=np.asarray(clipped_flags).mean(0).tolist(),
        max_raw_log_probability_error=max(errors), episodes=outcomes,
        interpretation='Derivative near zero attenuates changes to the action mean. This is not proof of a bug. '
                       'Raw-action probabilities should match collection. These episodes are diagnostic development data.',
        reference_comparison={
            'cost_surrogate': 'Same reward-minus-multiplier-times-cost sign and 1+multiplier divisor as OmniSafe PPOLag; local code additionally normalizes the combined advantage.',
            'actor_mean': 'Local tanh-bounded Gaussian mean; OmniSafe GaussianLearningActor uses an unbounded Gaussian mean.',
            'budget': 'Local cost critic uses gamma=1 and remaining episode time; reward uses gamma=.99 and time-limit bootstrap. This is not an identical reference configuration.',
            'multiplier': 'Local projected scalar update after policy update; reference updates multiplier before policy update.',
            'sources': [
                'https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag.py',
                'https://github.com/PKU-Alignment/omnisafe/blob/main/omnisafe/models/actor/gaussian_learning_actor.py']})
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','reference_comparison')}, indent=2))
    print('Saved:', out/'report.json')


if __name__ == '__main__':
    main()

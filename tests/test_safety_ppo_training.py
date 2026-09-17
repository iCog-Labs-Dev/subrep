"""Regression coverage for PPO returns across resets and rollout boundaries."""
import numpy as np
import pytest

from pilot.safety_gymnasium_ppo import SafetyPPOConfig
from pilot.train_safety_gymnasium_ppo import _collect_rollout


class ValueModel:
    def act(self, obs, **kwargs):
        return np.zeros(1), np.zeros(1), 0.0, float(obs[0])

    def distribution(self, obs):
        return None, obs[0]


class EpisodeEnv:
    def __init__(self, ending, reset_value):
        self.ending = ending
        self.reset_value = reset_value
        self.state = 0.0
        self.steps = 0
        self.resets = 0

    def step(self, action):
        self.state += 1
        self.steps += 1
        boundary = self.steps == 2
        return (np.array([self.state]), 1.0, 0.0,
                boundary and self.ending == 'terminated',
                boundary and self.ending == 'truncated', {})

    def reset(self):
        self.state = self.reset_value
        self.steps = 0
        self.resets += 1
        return np.array([self.state]), {}


def collect(ending, reset_value=100.0, rollout_steps=4):
    env = EpisodeEnv(ending, reset_value)
    result = _collect_rollout(
        env=env, model=ValueModel(), obs=np.array([0.0]),
        config=SafetyPPOConfig(gamma=0.5, gae_lambda=1.0,
                               rollout_steps=rollout_steps,
                               max_episode_steps=2 if ending == 'cutoff' else 100),
        device='cpu', episode_return=0.0, episode_cost=0.0,
        episode_length=0, completed_returns=[], completed_costs=[],
        current_cost_penalty=0.0,
    )
    return env, result


@pytest.mark.parametrize('ending', ['cutoff', 'truncated', 'terminated'])
@pytest.mark.parametrize('rollout_steps', [2, 4])
def test_returns_stop_at_reset_and_bootstrap_only_time_limits(ending, rollout_steps):
    env, result = collect(ending, rollout_steps=rollout_steps)
    _, changed_reset = collect(ending, reset_value=1000.0, rollout_steps=rollout_steps)
    # With gamma=.5, terminal returns are [1.5, 1]. Time limits
    # additionally bootstrap V(final_obs)=2, yielding [2, 2].
    expected = [1.5, 1.0] if ending == 'terminated' else [2.0, 2.0]
    np.testing.assert_allclose(result['returns'][:2], expected)
    np.testing.assert_allclose(changed_reset['returns'][:2], expected)
    assert env.resets == rollout_steps // 2
    assert result['episode_length'] == 0


def test_unfinished_rollout_bootstraps_without_reset():
    env, result = collect('cutoff', rollout_steps=1)
    np.testing.assert_allclose(result['returns'], [1.5])
    assert env.resets == 0
    assert result['episode_length'] == 1
    assert result['episode_return'] == 1.0

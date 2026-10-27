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


def test_multiplier_direction_bounds_and_no_completed_episode():
    from pilot.train_safety_gymnasium_ppo import _update_lagrange_multiplier
    c = SafetyPPOConfig(cost_limit=1., lambda_lr=.5, max_lagrange_multiplier=3.)
    assert _update_lagrange_multiplier(1., 3., c) == 2.
    assert _update_lagrange_multiplier(1., 0., c) == .5
    assert _update_lagrange_multiplier(1., None, c) == 1.
    assert _update_lagrange_multiplier(0., 0., c) == 0.
    assert _update_lagrange_multiplier(2., 100., c) == 3.
    _, rollout = collect('cutoff', rollout_steps=1)
    assert rollout['rollout_mean_episode_cost'] is None


def test_cost_penalty_uses_raw_cost_and_preserves_episode_accounting():
    class CostEnv:
        def step(self, action):
            return np.zeros(1), 3., 2., True, False, {}
        def reset(self):
            return np.zeros(1), {}
    returns, costs = [], []
    r = _collect_rollout(env=CostEnv(), model=ValueModel(), obs=np.zeros(1),
        config=SafetyPPOConfig(rollout_steps=1), device='cpu', episode_return=0.,
        episode_cost=0., episode_length=0, completed_returns=returns,
        completed_costs=costs, current_cost_penalty=4.)
    np.testing.assert_allclose(r['returns'], [-5.])
    assert returns == [3.] and costs == [2.]
    assert r['rollout_mean_episode_cost'] == 2.


def test_ppo_log_probability_matches_raw_action_and_checkpoint(tmp_path):
    import torch
    from pilot.safety_gymnasium_ppo import SafetyPPOPilot
    torch.manual_seed(7)
    model = SafetyPPOPilot(observation_dim=3, action_dim=2,
        action_low=[-1,-1], action_high=[1,1], hidden_sizes=(8,))
    with torch.no_grad(): model.log_std.fill_(2.)
    obs = np.array([.2,.3,.4],dtype=np.float32)
    clipped = 0
    for _ in range(20):
        action, raw, old_log, _ = model.act(obs,return_raw=True)
        new_log, _, _ = model.evaluate_actions(torch.tensor(obs),torch.tensor(raw))
        assert new_log.item() == pytest.approx(old_log,abs=1e-6)
        np.testing.assert_allclose(action,np.clip(raw,-1,1))
        clipped += int(np.any(action != raw))
    assert clipped > 0
    path = tmp_path/'policy.pt'
    model.save(path)
    restored = SafetyPPOPilot.load(path)
    np.testing.assert_allclose(restored.predict(obs),model.predict(obs))


def test_checkpoint_selection_respects_budget():
    from pilot.train_safety_gymnasium_ppo import _checkpoint_rank
    c=SafetyPPOConfig(use_lagrangian=True,cost_limit=1.)
    safe={'goal_reached_rate':.2,'mean_return':1.,'mean_cost':.5}
    unsafe={'goal_reached_rate':.9,'mean_return':10.,'mean_cost':2.}
    assert _checkpoint_rank(safe,c)>_checkpoint_rank(unsafe,c)
    c.use_lagrangian=False
    assert _checkpoint_rank(unsafe,c)>_checkpoint_rank(safe,c)


@pytest.mark.parametrize("lagrangian", [False, True])
def test_training_monitor_with_fake_environment(tmp_path, monkeypatch, lagrangian):
    import sys
    import json
    from types import SimpleNamespace
    from pilot.train_safety_gymnasium_ppo import train_safety_ppo
    class Env:
        observation_space=SimpleNamespace(shape=(1,))
        action_space=SimpleNamespace(shape=(1,),low=np.array([-1.]),high=np.array([1.]),seed=lambda s: None)
        def reset(self,seed=None): return np.zeros(1,dtype=np.float32),{}
        def step(self,action): return np.zeros(1,dtype=np.float32),1.,0.,True,False,{'goal_met':True}
        def close(self): pass
    monkeypatch.setitem(sys.modules,'safety_gymnasium',SimpleNamespace(make=lambda name:Env()))
    output=tmp_path/'policy.pt'
    result=train_safety_ppo(env_id='fake',output_path=output,
        config=SafetyPPOConfig(total_updates=2,rollout_steps=2,update_epochs=1,minibatch_size=2, use_lagrangian=lagrangian),
        eval_episodes=2,eval_every=1)
    assert result['evaluation']['goal_reached_rate']==1.
    assert output.exists()
    rows=[json.loads(x) for x in (tmp_path/'policy_run/metrics.jsonl').read_text().splitlines()]
    assert len(rows)==2 and rows[-1]['environment_steps']==4
    assert len(list((tmp_path/'policy_run').glob('update_*.pt')))==2
    with pytest.raises(FileExistsError):
        train_safety_ppo(env_id='fake',output_path=output,config=SafetyPPOConfig())


def test_lagrangian_advantage_penalizes_cost_without_changing_relative_scale():
    from pilot.train_safety_gymnasium_ppo import _lagrangian_advantages
    reward = np.array([1., 1., 1.], dtype=np.float32)
    cost = np.array([0., 1., 2.], dtype=np.float32)
    advantage = _lagrangian_advantages(reward, cost, 3.)
    assert advantage[0] > advantage[1] > advantage[2]
    np.testing.assert_allclose(advantage.mean(), 0., atol=1e-6)


def test_separate_critic_update_cannot_change_actor_and_roundtrips(tmp_path):
    import torch
    from pilot.safety_gymnasium_ppo import SafetyPPOPilot
    model = SafetyPPOPilot(observation_dim=3, action_dim=1,
        action_low=[-1], action_high=[1], hidden_sizes=(8,), separate_critic=True)
    obs = torch.randn(5, 3)
    before = model(obs)[0].detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    model(obs)[2].square().mean().backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    optimizer.step()
    torch.testing.assert_close(model(obs)[0], before, rtol=0, atol=0)
    model.save(tmp_path/'separate.pt')
    restored = SafetyPPOPilot.load(tmp_path/'separate.pt')
    assert restored.separate_critic
    torch.testing.assert_close(restored(obs)[0], model(obs)[0])


@pytest.mark.parametrize('ending', ['terminated', 'truncated'])
def test_lagrangian_reward_and_cost_returns_bootstrap_separately(ending):
    import torch
    class CostEnv(EpisodeEnv):
        def step(self, action):
            obs, reward, _, terminal, truncated, info = super().step(action)
            return obs, reward, 2., terminal, truncated, info
    class CostValue(torch.nn.Module):
        def forward(self, obs): return obs[:1] * 3.
    rollout = _collect_rollout(env=CostEnv(ending, 100.), model=ValueModel(),
        obs=np.zeros(1, dtype=np.float32), config=SafetyPPOConfig(use_lagrangian=True,
        rollout_steps=2, gamma=.5, gae_lambda=1., max_episode_steps=100),
        device='cpu', episode_return=0., episode_cost=0., episode_length=0,
        completed_returns=[], completed_costs=[], current_cost_penalty=20., cost_critic=CostValue())
    np.testing.assert_allclose(rollout['returns'], [1.5, 1.] if ending=='terminated' else [2., 2.])
    np.testing.assert_allclose(rollout['cost_returns'], [3., 2.] if ending=='terminated' else [4.5, 5.])

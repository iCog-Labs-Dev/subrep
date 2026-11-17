import pytest
import torch
from pilot.safety_gymnasium_ppo import SafetyPPOPilot
from pilot.train_safety_gymnasium_ppo import initialize_from_checkpoint


def model(separate=False,high=1.):
    return SafetyPPOPilot(observation_dim=3,action_dim=1,action_low=[-1],action_high=[high],separate_critic=separate)


@pytest.mark.parametrize('separate',[False,True])
def test_initialization_preserves_actor_and_reward_value(tmp_path,separate):
    source=model(separate);path=tmp_path/'source.pt';source.save(path,metadata={'environment':'test'})
    target=model(True);report=initialize_from_checkpoint(target,path,'test')
    obs=torch.randn(8,3)
    for a,b in zip(source(obs),target(obs)):torch.testing.assert_close(a,b)
    assert len(report['sha256'])==64
    assert target.backbone[0].weight.data_ptr()!=target.critic_backbone[0].weight.data_ptr()
    assert all(p.requires_grad for p in target.parameters())


def test_contract_rejections(tmp_path):
    path=tmp_path/'source.pt';model().save(path,metadata={'environment':'test'})
    with pytest.raises(ValueError,match='environment'):initialize_from_checkpoint(model(True),path,'wrong')
    with pytest.raises(ValueError,match='mismatch'):initialize_from_checkpoint(model(True,2.),path,'test')


@pytest.mark.parametrize("diagnostic", [False, True])
def test_cost_warmup_freezes_policy_and_multiplier(monkeypatch, tmp_path, diagnostic):
    import json
    import sys
    import types
    import numpy as np
    import pilot.train_safety_gymnasium_ppo as trainer
    from pilot.safety_gymnasium_ppo import SafetyPPOConfig

    class Space:
        shape = (3,)
        low = np.full(3, -1., dtype=np.float32)
        high = np.ones(3, dtype=np.float32)
        def seed(self, seed):
            pass

    class Env:
        observation_space = Space()
        action_space = Space()
        def reset(self, **kwargs):
            return np.zeros(3, dtype=np.float32), {}
        def step(self, action):
            return np.ones(3, dtype=np.float32), 1., 2., False, False, {}
        def close(self):
            pass

    monkeypatch.setitem(sys.modules, 'safety_gymnasium', types.SimpleNamespace(make=lambda _: Env()))
    monkeypatch.setattr(trainer, 'evaluate_safety_ppo', lambda *a, **kw: {
        'goal_reached_rate': 1., 'mean_return': 1., 'mean_cost': 2.,
        'mean_effort': 0., 'mean_action_change': 0.})
    captured = {}
    original_collect = trainer._collect_rollout
    original_cost_update = trainer._cost_only_update
    def collect(**kwargs):
        model = kwargs['model']
        if 'policy' not in captured:
            captured['policy'] = {k: v.clone() for k, v in model.state_dict().items()}
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, captured['policy'][k], rtol=0, atol=0)
        assert kwargs['current_cost_penalty'] == .01
        return original_collect(**kwargs)
    def cost_update(critic, *args):
        before = [p.detach().clone() for p in critic.parameters()]
        result = original_cost_update(critic, *args)
        assert any(not torch.equal(a, b) for a, b in zip(before, critic.parameters()))
        captured['warmup_calls'] = captured.get('warmup_calls', 0) + 1
        return result
    def policy_update(**kwargs):
        if diagnostic:
            rollout = kwargs['rollout']
            expected = trainer._lagrangian_advantages(np.zeros_like(rollout['cost_advantages']), rollout['cost_advantages'], 1.)
            np.testing.assert_allclose(rollout['advantages'], expected)
            assert rollout['cost_only_diagnostic']
        rates = [g['lr'] for g in kwargs['optimizer'].param_groups]
        assert rates == [1e-5, 3e-4]
        assert len(kwargs['optimizer'].state) == 0
        captured['policy_calls'] = captured.get('policy_calls', 0) + 1
        return {}
    monkeypatch.setattr(trainer, '_collect_rollout', collect)
    monkeypatch.setattr(trainer, '_cost_only_update', cost_update)
    monkeypatch.setattr(trainer, '_ppo_update', policy_update)
    trainer.train_safety_ppo(env_id='test', output_path=tmp_path/'test.pt',
        config=SafetyPPOConfig(use_lagrangian=True, total_updates=1, rollout_steps=4,
            max_episode_steps=2, update_epochs=1, minibatch_size=2,
            initial_lagrange_multiplier=.01),
        eval_episodes=1, cost_warmup_updates=2, actor_learning_rate=1e-5,
        cost_only_diagnostic=diagnostic)
    assert captured['warmup_calls'] == 2
    assert captured['policy_calls'] == 1
    rows = [json.loads(x) for x in (tmp_path/'test_run/metrics.jsonl').read_text().splitlines()]
    assert [r['phase'] for r in rows] == ['initial', 'cost_warmup', 'cost_warmup', 'policy']
    assert [r['environment_steps'] for r in rows] == [0, 4, 8, 12]
    if diagnostic:
        assert rows[-1]['lagrange_multiplier'] == .01
        report = json.loads((tmp_path/'test_run/cost_diagnostic.json').read_text())
        assert report['initial']['sampled_evaluation']['mean_cost'] == 2.
        assert report['final']['evaluation']['mean_cost'] == 2.
    else:
        assert rows[-1]['lagrange_multiplier'] > .01
    assert (tmp_path/'test_run/update_00000.pt').exists()
    assert (tmp_path/'test_run/update_00001.pt').exists()
    summary = json.loads((tmp_path/'test_run/summary.json').read_text())
    assert summary['best_metadata']['update'] == 0


def test_cost_only_update_changes_actor_but_freezes_reward_critic():
    import numpy as np
    from pilot.safety_gymnasium_ppo import SafetyPPOConfig
    from pilot.train_safety_gymnasium_ppo import _ppo_update
    torch.manual_seed(7)
    policy = model(True)
    obs = torch.randn(16, 3)
    with torch.no_grad():
        dist, _ = policy.distribution(obs)
        actions = dist.sample()
        logp, _, _ = policy.evaluate_actions(obs, actions)
    before = {k: v.clone() for k, v in policy.state_dict().items()}
    rollout = dict(observations=obs.numpy(), actions=actions.numpy(), old_log_probs=logp.numpy(),
        advantages=np.linspace(-1, 1, 16, dtype=np.float32),
        returns=np.full(16, 10000., dtype=np.float32), cost_only_diagnostic=True)
    _ppo_update(model=policy, optimizer=torch.optim.Adam(policy.parameters(), lr=.001),
        rollout=rollout, config=SafetyPPOConfig(update_epochs=1, minibatch_size=16,
            entropy_coef=100.), device='cpu')
    for k, v in policy.state_dict().items():
        if k.startswith(('critic.', 'critic_backbone.')):
            torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    assert any(not torch.equal(v, before[k]) for k, v in policy.state_dict().items()
               if k.startswith(('actor_mean.', 'backbone.')))


def test_sampled_evaluation_reproducible_and_preserves_torch_rng(monkeypatch):
    import sys
    import types
    import numpy as np
    from pilot.train_safety_gymnasium_ppo import evaluate_safety_ppo
    class Env:
        action_space = types.SimpleNamespace(low=np.array([-1.]), high=np.array([1.]))
        def reset(self, **kwargs):
            return np.zeros(3, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(3, dtype=np.float32), float(action[0]), float(action[0]**2), False, False, {}
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, 'safety_gymnasium', types.SimpleNamespace(make=lambda _: Env()))
    policy = model(True)
    state = torch.random.get_rng_state().clone()
    args = dict(env_id='test', seed=42, episodes=3, max_episode_steps=3, deterministic=False)
    first = evaluate_safety_ppo(policy, **args)
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
    assert evaluate_safety_ppo(policy, **args) == first

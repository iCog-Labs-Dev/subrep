"""Behavioral checks for cost-only PPO updates and finite episode costs."""
import numpy as np
import torch
from pilot.safety_gymnasium_ppo import SafetyPPOPilot, SafetyPPOConfig
from pilot.train_safety_gymnasium_ppo import _compute_gae, _lagrangian_advantages, _ppo_update


def test_cost_update_moves_probability_toward_lower_cost_actions():
    torch.manual_seed(4)
    model = SafetyPPOPilot(observation_dim=3, action_dim=1, action_low=[-1], action_high=[1], separate_critic=True)
    obs = torch.zeros(64, 3)
    actions = torch.cat([torch.full((32, 1), -.5), torch.full((32, 1), .5)])
    with torch.no_grad():
        before, _, _ = model.evaluate_actions(obs, actions)
    costs = np.r_[np.zeros(32), np.ones(32)].astype(np.float32)
    rollout = dict(observations=obs.numpy(), actions=actions.numpy(), old_log_probs=before.numpy(),
        advantages=_lagrangian_advantages(np.zeros(64, dtype=np.float32), costs, 1.),
        returns=np.zeros(64, dtype=np.float32), cost_only_diagnostic=True)
    _ppo_update(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        rollout=rollout, config=SafetyPPOConfig(update_epochs=1, minibatch_size=64), device='cpu')
    with torch.no_grad():
        after, _, _ = model.evaluate_actions(obs, actions)
    assert (after[:32].mean()-after[32:].mean()) > (before[:32].mean()-before[32:].mean())


def test_cost_returns_match_hand_calculation_and_stop_at_episode_end():
    # Two episodes: [1, 2], [10, 20]. No cost continuation past the budget horizon.
    _, returns = _compute_gae(rewards=np.array([1,2,10,20], dtype=np.float32),
        values=np.array([7,8,9,10], dtype=np.float32), dones=np.array([0,1,0,1], dtype=np.float32),
        next_value=999., gamma=1., gae_lambda=1., normalize=False)
    np.testing.assert_allclose(returns, [3,2,30,20])


def test_raw_action_probability_matches_collection_even_when_clipped():
    torch.manual_seed(7)
    model = SafetyPPOPilot(observation_dim=3, action_dim=1, action_low=[-.01], action_high=[.01])
    obs = np.zeros(3, dtype=np.float32)
    clipped, raw, old_logp, _ = model.act(obs, return_raw=True)
    with torch.no_grad():
        logp, _, _ = model.evaluate_actions(torch.tensor(obs), torch.tensor(raw))
    np.testing.assert_allclose(logp.item(), old_logp, rtol=1e-6)
    np.testing.assert_allclose(clipped, np.clip(raw, -.01, .01))
    assert not np.array_equal(clipped, raw)

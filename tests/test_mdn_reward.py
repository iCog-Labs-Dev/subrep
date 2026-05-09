from __future__ import annotations

import numpy as np
import pytest
import torch

from utils.mdn_reward import compute_advantage, compute_mdn_policy_loss, compute_mdn_utility


def test_compute_mdn_utility_matches_weighted_motives_dot_product():
    actual_motives = np.array([0.8, 0.2], dtype=np.float32)
    weights_used = np.array([0.25, 0.75], dtype=np.float32)

    utility = compute_mdn_utility(actual_motives, weights_used)
    expected = 0.25 * 0.8 + 0.75 * 0.2

    assert np.isclose(utility, expected)


def test_compute_mdn_utility_optionally_includes_payoff():
    actual_motives = np.array([0.8, 0.2], dtype=np.float32)
    weights_used = np.array([0.25, 0.75], dtype=np.float32)

    utility = compute_mdn_utility(actual_motives, weights_used, actual_payoff=2.0, payoff_weight=0.5)
    expected = 0.25 * 0.8 + 0.75 * 0.2 + 0.5 * 2.0

    assert np.isclose(utility, expected)


def test_compute_mdn_utility_requires_payoff_when_payoff_weight_positive():
    with pytest.raises(ValueError, match="actual_payoff"):
        compute_mdn_utility(
            actual_motives=np.array([0.8, 0.2], dtype=np.float32),
            weights_used=np.array([0.25, 0.75], dtype=np.float32),
            payoff_weight=0.5,
        )


def test_compute_advantage_uses_explicit_baseline_when_present():
    advantage = compute_advantage(utility=1.2, baseline_utility=0.7)
    assert np.isclose(advantage, 0.5)


def test_compute_advantage_falls_back_to_running_baseline():
    advantage = compute_advantage(utility=1.2, running_baseline=1.0)
    assert np.isclose(advantage, 0.2)


def test_compute_advantage_returns_utility_when_no_baseline_exists():
    advantage = compute_advantage(utility=1.2)
    assert np.isclose(advantage, 1.2)


def test_compute_mdn_policy_loss_returns_finite_value_for_positive_advantage():
    log_prob = torch.tensor(-0.7, dtype=torch.float32)
    loss = compute_mdn_policy_loss(log_prob, advantage=1.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_compute_mdn_policy_loss_returns_finite_value_for_negative_advantage():
    log_prob = torch.tensor(-0.7, dtype=torch.float32)
    loss = compute_mdn_policy_loss(log_prob, advantage=-1.5)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_compute_mdn_policy_loss_rejects_non_scalar_log_prob():
    with pytest.raises(ValueError, match="log_prob"):
        compute_mdn_policy_loss(torch.tensor([-0.7, -0.8], dtype=torch.float32), advantage=1.0)

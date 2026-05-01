"""Tests for the baseline policy and improvement calculator."""

from __future__ import annotations

import numpy as np

from baseline.idle_policy import IdlePolicy
from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate


class _DeterministicBaselineEnv:
    """Small deterministic environment for baseline regression tests."""

    def __init__(self):
        self._seed = 0
        self._step_count = 0

    def reset(self, seed=None):
        if seed is not None:
            self._seed = int(seed)
        self._step_count = 0
        obs = np.zeros(8, dtype=np.float32)
        return obs, {"seed": self._seed}

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(8, dtype=np.float32)
        seed_term = (self._seed % 5) * 0.1
        reward = np.array([1.0 + seed_term, 0.5 - seed_term], dtype=np.float32)
        terminated = self._step_count >= 3
        truncated = False
        return obs, reward, terminated, truncated, {}


def test_idle_policy_returns_consistent_action():
    env = _DeterministicBaselineEnv()
    policy = IdlePolicy(env)

    action_a = policy.get_action(np.zeros(8, dtype=np.float32))
    action_b = policy.get_action(np.ones(8, dtype=np.float32))

    assert action_a == 0
    assert action_b == 0
    assert action_a == action_b


def test_baseline_episodes_are_stable_with_same_seed():
    env_one = _DeterministicBaselineEnv()
    env_two = _DeterministicBaselineEnv()

    policy_one = IdlePolicy(env_one, gamma=0.9)
    policy_two = IdlePolicy(env_two, gamma=0.9)

    stats_one = policy_one.run_baseline_episodes(num_episodes=8, seed=123)
    stats_two = policy_two.run_baseline_episodes(num_episodes=8, seed=123)

    assert np.isclose(stats_one["baseline_payoff"], stats_two["baseline_payoff"])
    assert np.allclose(stats_one["baseline_motives"], stats_two["baseline_motives"])
    assert stats_one["episode_payoffs"].shape == (8,)
    assert stats_one["episode_motives"].shape == (8, 2)


def test_improvement_calculator_computes_expected_values():
    baseline_stats = {
        "baseline_payoff": 1.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }
    calculator = ImprovementCalculator(baseline_stats)

    delta_r, delta_n = calculator.compute_improvements(
        skill_payoff=1.5,
        skill_motives=np.array([0.2, 0.8], dtype=np.float32),
    )

    assert np.isclose(delta_r, 0.5)
    assert np.allclose(delta_n, np.array([-0.3, 0.6], dtype=np.float32))


def test_improvement_calculator_handles_zero_and_negative_improvements():
    baseline_stats = {
        "baseline_payoff": 2.0,
        "baseline_motives": np.array([1.0, 1.0], dtype=np.float32),
    }
    calculator = ImprovementCalculator(baseline_stats)

    zero_delta_r, zero_delta_n = calculator.compute_improvements(
        skill_payoff=2.0,
        skill_motives=np.array([1.0, 1.0], dtype=np.float32),
    )
    negative_delta_r, negative_delta_n = calculator.compute_improvements(
        skill_payoff=1.5,
        skill_motives=np.array([0.25, 0.75], dtype=np.float32),
    )

    assert np.isclose(zero_delta_r, 0.0)
    assert np.allclose(zero_delta_n, np.zeros(2, dtype=np.float32))
    assert np.isclose(negative_delta_r, -0.5)
    assert np.allclose(negative_delta_n, np.array([-0.75, -0.25], dtype=np.float32))


def test_improvement_calculator_rejects_shape_and_finite_errors():
    baseline_stats = {
        "baseline_payoff": 1.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }
    calculator = ImprovementCalculator(baseline_stats)

    try:
        calculator.compute_improvements(1.0, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for shape mismatch")

    try:
        calculator.validate_improvements(np.inf, np.array([0.0, 0.0], dtype=np.float32))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError for infinite delta_r")


def test_improvements_can_be_passed_to_cds_and_pds():
    baseline_stats = {
        "baseline_payoff": 1.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }
    calculator = ImprovementCalculator(baseline_stats)
    delta_r, delta_n = calculator.compute_improvements(
        skill_payoff=1.7,
        skill_motives=np.array([0.8, 0.4], dtype=np.float32),
    )

    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=0.1)

    assert cds_gate.admit(delta_r, delta_n) is True
    assert pds_gate.admit(delta_r, delta_n) is True
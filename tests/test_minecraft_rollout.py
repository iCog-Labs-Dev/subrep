"""Tests for env/minecraft_rollout.py -- the MetaMo path's delta estimation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from baseline.idle_policy import IdlePolicy
from env.minecraft_rollout import (
    DEFAULT_HORIZON,
    MINECRAFT_INITIAL_GOALS,
    appraisal_scales,
    evaluate_candidates,
    evaluate_option,
    execute_option,
    run_option,
)
from env.minecraft_stub import SKILL_NAMES, MinecraftStubEnv

GAMMA = 0.99
SEED = 42


def _fresh_env(noise_scale: float = 0.0, seed: int = SEED) -> MinecraftStubEnv:
    env = MinecraftStubEnv(seed=seed, noise_scale=noise_scale)
    env.reset(seed=seed)
    return env


def _snapshot(env: MinecraftStubEnv):
    return (
        env._t,
        env._threat,
        env._totals.copy(),
        env._rng.bit_generator.state,
    )


# ---------------------------------------------------------------------------
# run_option / evaluate_option
# ---------------------------------------------------------------------------


def test_run_option_without_horizon_runs_to_episode_end():
    env = _fresh_env()
    run_option(env, 1, horizon=None, gamma=GAMMA)
    assert env._t == env.episode_length


def test_run_option_takes_exactly_horizon_steps():
    env = _fresh_env()
    run_option(env, 1, horizon=DEFAULT_HORIZON, gamma=GAMMA)
    assert env._t == DEFAULT_HORIZON


def test_run_option_stops_at_episode_end_mid_option():
    env = _fresh_env()
    for _ in range(env.episode_length - 1):
        env.step(0)

    run_option(env, 1, horizon=DEFAULT_HORIZON, gamma=GAMMA)

    assert env._t == env.episode_length  # one step, not three
    assert env.episode_over


def test_run_option_rejects_non_positive_horizon():
    env = _fresh_env()
    with pytest.raises(ValueError):
        run_option(env, 1, horizon=0, gamma=GAMMA)


def test_evaluate_option_leaves_the_live_env_untouched():
    env = _fresh_env(noise_scale=0.02)
    for _ in range(5):  # mid-episode, non-trivial state
        env.step(2)
    before = _snapshot(env)

    evaluate_option(env, 3, horizon=None, gamma=GAMMA)

    after = _snapshot(env)
    assert before[0] == after[0]
    assert before[1] == after[1]
    np.testing.assert_array_equal(before[2], after[2])
    assert before[3] == after[3]


def test_evaluate_option_matches_run_option_from_the_same_state():
    env = _fresh_env(noise_scale=0.02)
    for _ in range(3):
        env.step(1)

    expected = run_option(
        _clone_by_replay(noise_scale=0.02, prefix=[1, 1, 1]),
        4,
        horizon=None,
        gamma=GAMMA,
    )
    actual = evaluate_option(env, 4, horizon=None, gamma=GAMMA)

    assert actual[0] == pytest.approx(expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


def _clone_by_replay(*, noise_scale: float, prefix):
    """An independent env brought to the same state by replaying `prefix`."""
    env = _fresh_env(noise_scale=noise_scale)
    for action in prefix:
        env.step(action)
    return env


# ---------------------------------------------------------------------------
# execute_option
# ---------------------------------------------------------------------------


def test_execute_option_advances_the_live_env_by_the_horizon():
    env = _fresh_env()
    execute_option(env, 2, horizon=DEFAULT_HORIZON, gamma=GAMMA)
    assert env._t == DEFAULT_HORIZON


def test_execute_option_measures_against_idle_from_the_same_state():
    env = _fresh_env(noise_scale=0.02)
    for _ in range(6):
        env.step(1)
    prefix = [1] * 6

    delta_r, delta_n = execute_option(env, 4, horizon=DEFAULT_HORIZON, gamma=GAMMA)

    option = run_option(
        _clone_by_replay(noise_scale=0.02, prefix=prefix),
        4,
        horizon=DEFAULT_HORIZON,
        gamma=GAMMA,
    )
    idle = run_option(
        _clone_by_replay(noise_scale=0.02, prefix=prefix),
        0,
        horizon=DEFAULT_HORIZON,
        gamma=GAMMA,
    )
    assert delta_r == pytest.approx(option[0] - idle[0], abs=1e-5)
    np.testing.assert_allclose(delta_n, option[1] - idle[1], atol=1e-5)


def test_executing_the_baseline_action_is_a_zero_improvement():
    env = _fresh_env(noise_scale=0.02)
    delta_r, delta_n = execute_option(env, 0, horizon=DEFAULT_HORIZON, gamma=GAMMA)
    assert delta_r == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(delta_n, 0.0, atol=1e-6)


def test_evaluation_and_execution_use_the_same_horizon():
    """A candidate's evaluated delta is exactly what executing it realizes."""
    env = _fresh_env(noise_scale=0.02)
    for _ in range(5):
        env.step(3)

    _, records = evaluate_candidates(env, gamma=GAMMA, horizon=DEFAULT_HORIZON)
    record = next(r for r in records if r.metadata["action"] == 2)

    delta_r, delta_n = execute_option(env, 2, horizon=DEFAULT_HORIZON, gamma=GAMMA)

    assert delta_r == pytest.approx(record.delta_r, abs=1e-5)
    np.testing.assert_allclose(delta_n, record.delta_n, atol=1e-5)


# ---------------------------------------------------------------------------
# evaluate_candidates
# ---------------------------------------------------------------------------


def test_baseline_matches_idle_policy_on_a_noiseless_stub():
    """The refactor away from IdlePolicy changed nothing numerically.

    Valid only while the rollout keeps IdlePolicy's accumulation rules; it is
    removed when the task reward is separated from the motive vector.
    """
    env = _fresh_env()
    baseline, _ = evaluate_candidates(env, gamma=GAMMA, horizon=None)

    reference = IdlePolicy(
        env=MinecraftStubEnv(seed=SEED, noise_scale=0.0),
        idle_action=0,
        gamma=GAMMA,
    ).run_baseline_episodes(num_episodes=1, seed=SEED)

    assert baseline["baseline_payoff"] == pytest.approx(
        reference["baseline_payoff"]
    )
    np.testing.assert_allclose(
        baseline["baseline_motives"], reference["baseline_motives"], rtol=1e-6
    )


def test_one_candidate_per_non_baseline_action():
    env = _fresh_env()
    _, records = evaluate_candidates(env, gamma=GAMMA, horizon=None)

    assert [r.skill_id for r in records] == list(SKILL_NAMES[1:])
    assert [r.metadata["action"] for r in records] == list(range(1, len(SKILL_NAMES)))
    assert all(r.gate_type == "PDS" and not r.is_certified for r in records)


def test_baseline_action_is_configurable():
    env = _fresh_env()
    _, records = evaluate_candidates(
        env, gamma=GAMMA, horizon=None, baseline_action=2
    )
    assert SKILL_NAMES[2] not in [r.skill_id for r in records]
    assert SKILL_NAMES[0] in [r.skill_id for r in records]


def test_candidate_deltas_are_rollout_minus_baseline():
    env = _fresh_env()
    baseline, records = evaluate_candidates(env, gamma=GAMMA, horizon=None)

    for record in records:
        payoff, motives = run_option(
            _fresh_env(), record.metadata["action"], horizon=None, gamma=GAMMA
        )
        assert record.delta_r == pytest.approx(
            payoff - baseline["baseline_payoff"], abs=1e-5
        )
        np.testing.assert_allclose(
            record.delta_n,
            motives - baseline["baseline_motives"],
            atol=1e-5,
        )


def test_evaluate_candidates_leaves_the_live_env_untouched():
    env = _fresh_env(noise_scale=0.02)
    for _ in range(4):
        env.step(5)
    before = _snapshot(env)

    evaluate_candidates(env, gamma=GAMMA, horizon=None)

    after = _snapshot(env)
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[2], after[2])
    assert before[3] == after[3]


# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------


def test_appraisal_scales_are_mean_absolute_deltas():
    candidates = [
        SimpleNamespace(delta_r=2.0, delta_n=(1.0, -3.0)),
        SimpleNamespace(delta_r=-4.0, delta_n=(0.0, 2.0)),
    ]
    payoff_scale, motive_scale = appraisal_scales(candidates)
    assert payoff_scale == pytest.approx(3.0)
    assert motive_scale == pytest.approx(1.5)


def test_appraisal_scales_fall_back_to_one_when_all_deltas_are_zero():
    candidates = [SimpleNamespace(delta_r=0.0, delta_n=(0.0, 0.0))]
    assert appraisal_scales(candidates) == (1.0, 1.0)


def test_appraisal_scales_floor_raises_small_scales_only():
    candidates = [SimpleNamespace(delta_r=0.01, delta_n=(2.0, 2.0))]
    payoff_scale, motive_scale = appraisal_scales(candidates, floor=0.5)
    assert payoff_scale == pytest.approx(0.5)
    assert motive_scale == pytest.approx(2.0)


def test_initial_goals_cover_metamos_eight_goals():
    assert MINECRAFT_INITIAL_GOALS.shape == (8,)
    assert np.all((MINECRAFT_INITIAL_GOALS >= 0.0) & (MINECRAFT_INITIAL_GOALS <= 1.0))

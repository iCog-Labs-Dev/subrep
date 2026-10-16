"""Tests for env/lunar_lander_rollout.py -- same-seed candidate evaluation.

The core tests use the fixed-action and random candidates, which need no PPO
checkpoint. Tests that fly the PPO candidates skip when models/pilot_ppo.pt is
missing; the MetaMo test skips without a MetaMo checkout.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bridge._loader import is_available as metamo_is_available
from bridge.protocol import SkillOutcome
from bridge.stimulus import build_stimulus_values
from data_collector.collect_candidate_sets import (
    CandidatePolicy,
    _fixed_action_policy,
    _random_policy,
)
from env.lunar_lander_rollout import (
    APPRAISAL_SCALE_FLOOR,
    EPISODE_STEP_LIMIT,
    NOOP_BASELINE,
    appraisal_scales,
    episode_outcome,
    evaluate_candidates,
    execute_candidate,
    run_episode,
)
from env.lunar_lander_wrapper import SubRepEnv

GAMMA = 0.99
SEED = 1
WEIGHTS = np.array([0.5, 0.5])
PILOT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "pilot_ppo.pt"

requires_pilot = pytest.mark.skipif(
    not PILOT_CHECKPOINT.is_file(),
    reason="PPO checkpoint models/pilot_ppo.pt not found",
)


@pytest.fixture()
def env():
    lander = SubRepEnv(seed=SEED)
    yield lander
    lander.close()


def _ci_candidates(env: SubRepEnv):
    """The default candidates that need no PPO checkpoint."""
    return (
        CandidatePolicy("noop", _fixed_action_policy(0)),
        CandidatePolicy("left_engine", _fixed_action_policy(1)),
        CandidatePolicy("main_engine", _fixed_action_policy(2)),
        CandidatePolicy("right_engine", _fixed_action_policy(3)),
        CandidatePolicy("random", _random_policy(env)),
    )


def _by_id(records):
    return {r.skill_id: r for r in records}


class _RecordingEnv:
    """Delegates to a real env; records each start and every step's info."""

    def __init__(self, inner: SubRepEnv) -> None:
        self.inner = inner
        self.env = inner.env
        self.infos = []
        self.starts = []

    def reset(self, seed=None):
        self.infos = []
        obs, info = self.inner.reset(seed=seed)
        self.starts.append(np.array(obs, copy=True))
        return obs, info

    def step(self, action):
        result = self.inner.step(action)
        self.infos.append(result[4])
        return result


class _StubEnv:
    """Ends its episode after `length` steps and refuses to step past the end."""

    def __init__(self, length: int = 3, *, truncate: bool = False, starts=None) -> None:
        self.length = length
        self.truncate = truncate
        self.starts = list(starts) if starts is not None else None
        self.env = SimpleNamespace(action_space=SimpleNamespace(seed=lambda _seed: None))
        self.steps = 0
        self.ended = False

    def reset(self, seed=None):
        self.steps = 0
        self.ended = False
        start = self.starts.pop(0) if self.starts else np.zeros(8)
        return np.asarray(start, dtype=np.float32), {}

    def step(self, action):
        if self.ended:
            raise AssertionError("env.step called after the episode ended")
        self.steps += 1
        over = self.steps >= self.length
        self.ended = over
        terminated = over and not self.truncate
        truncated = over and self.truncate
        info = {
            "terminal_reward": -100.0 if terminated else 0.0,
            "landing_success": False,
            "crashed": terminated,
        }
        return np.zeros(8), np.array([1.0, -0.5]), terminated, truncated, info


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_same_seed_gives_the_same_result(env, action):
    policy = _fixed_action_policy(action)
    first = run_episode(env, policy, seed=SEED, gamma=GAMMA)
    second = run_episode(env, policy, seed=SEED, gamma=GAMMA)

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2]["steps"] == second[2]["steps"]


def test_random_policy_replays_with_the_same_seed(env):
    policy = _random_policy(env)
    first = run_episode(env, policy, seed=SEED, gamma=GAMMA)
    # Advance the action space between runs; the seed must still replay.
    env.env.action_space.sample()
    second = run_episode(env, policy, seed=SEED, gamma=GAMMA)

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2]["steps"] == second[2]["steps"]


def test_task_reward_is_the_discounted_landing_result(env):
    recording = _RecordingEnv(env)
    r_hat, _, info = run_episode(recording, _fixed_action_policy(0), seed=SEED, gamma=GAMMA)

    expected = sum(
        GAMMA ** t * float(step_info["terminal_reward"])
        for t, step_info in enumerate(recording.infos)
    )
    assert info["steps"] == len(recording.infos)
    assert r_hat == pytest.approx(expected, rel=1e-9)


def test_a_crash_gives_a_negative_task_reward(env):
    # Doing nothing free-falls into the ground.
    r_hat, _, info = run_episode(env, _fixed_action_policy(0), seed=SEED, gamma=GAMMA)
    assert info["crashed"]
    assert not info["landing_success"]
    assert r_hat < 0.0


def test_an_episode_cut_off_by_max_steps_has_no_task_reward(env):
    r_hat, n_hat, info = run_episode(
        env, _fixed_action_policy(0), seed=SEED, gamma=GAMMA, max_steps=10
    )
    assert info["steps"] == 10
    assert info["stop_reason"] == "max_steps"
    assert not info["crashed"] and not info["landing_success"]
    assert r_hat == 0.0
    assert np.any(n_hat != 0.0)  # the objectives still accumulate


def test_task_reward_is_not_the_sum_of_the_objectives(env):
    for action in range(4):
        r_hat, n_hat, _ = run_episode(
            env, _fixed_action_policy(action), seed=SEED, gamma=GAMMA
        )
        assert r_hat != pytest.approx(float(np.sum(n_hat)), abs=1e-3), action


def test_info_reports_the_episode(env):
    _, n_hat, info = run_episode(env, _fixed_action_policy(0), seed=SEED, gamma=GAMMA)
    obs, _ = env.reset(seed=SEED)

    assert n_hat.shape == (2,)
    assert info["seed"] == SEED
    np.testing.assert_allclose(info["start_obs"], obs, rtol=1e-6, atol=1e-6)
    for key in ("steps", "landing_success", "crashed", "truncated"):
        assert key in info


def test_policies_returning_a_bare_action_are_accepted(env):
    tuple_result = run_episode(env, _fixed_action_policy(2), seed=SEED, gamma=GAMMA)
    bare_result = run_episode(env, lambda _obs: 2, seed=SEED, gamma=GAMMA)
    assert tuple_result[0] == bare_result[0]
    np.testing.assert_array_equal(tuple_result[1], bare_result[1])


def test_policy_reset_is_called_with_the_seed(env):
    class StatefulPolicy:
        def __init__(self) -> None:
            self.reset_seeds = []

        def reset(self, seed=None, **kwargs):
            self.reset_seeds.append(seed)

        def __call__(self, _obs):
            return 0, 1.0

    policy = StatefulPolicy()
    run_episode(env, CandidatePolicy("stateful", policy), seed=7, gamma=GAMMA)
    assert policy.reset_seeds == [7]


@pytest.mark.parametrize("truncate", [False, True])
def test_no_step_after_the_episode_ends(truncate):
    stub = _StubEnv(length=3, truncate=truncate)
    r_hat, n_hat, info = run_episode(stub, lambda _obs: 0, seed=SEED, gamma=GAMMA)

    assert stub.steps == 3
    assert info["steps"] == 3
    assert info["truncated"] is truncate
    assert r_hat == pytest.approx(0.0 if truncate else -100.0 * GAMMA ** 2)
    np.testing.assert_allclose(n_hat, np.array([1.0, -0.5]) * (1 + GAMMA + GAMMA ** 2))


def test_max_steps_beyond_the_episode_still_stops_at_the_end():
    stub = _StubEnv(length=3)
    _, _, info = run_episode(stub, lambda _obs: 0, seed=SEED, gamma=GAMMA, max_steps=50)
    assert info["steps"] == 3
    assert info["stop_reason"] == "terminated"


@pytest.mark.parametrize("gamma", [-0.1, 1.5])
def test_invalid_gamma_is_rejected(gamma):
    with pytest.raises(ValueError):
        run_episode(_StubEnv(), lambda _obs: 0, seed=SEED, gamma=gamma)


def test_invalid_max_steps_is_rejected():
    with pytest.raises(ValueError):
        run_episode(_StubEnv(), lambda _obs: 0, seed=SEED, gamma=GAMMA, max_steps=0)


# ---------------------------------------------------------------------------
# evaluate_candidates
# ---------------------------------------------------------------------------


def test_random_candidate_replays_across_evaluations(env):
    candidates = _ci_candidates(env)
    _, first = evaluate_candidates(env, candidates, seed=SEED, gamma=GAMMA)
    env.env.action_space.sample()
    _, second = evaluate_candidates(env, candidates, seed=SEED, gamma=GAMMA)

    a, b = _by_id(first)["random"], _by_id(second)["random"]
    assert a.delta_r == b.delta_r
    assert a.delta_n == b.delta_n
    assert a.metadata["steps"] == b.metadata["steps"]


def test_every_run_starts_from_the_same_observation(env):
    recording = _RecordingEnv(env)
    candidates = _ci_candidates(env)
    baseline, records = evaluate_candidates(recording, candidates, seed=SEED, gamma=GAMMA)

    # One reset for the baseline and one per non-baseline candidate.
    assert len(recording.starts) == 1 + len(records)
    start = baseline["info"]["start_obs"]
    for obs in recording.starts:
        np.testing.assert_allclose(obs, start, rtol=1e-6, atol=1e-6)


def test_a_different_start_is_an_error():
    stub = _StubEnv(starts=[np.zeros(8), np.ones(8)])
    candidates = (CandidatePolicy("left_engine", _fixed_action_policy(1)),)
    with pytest.raises(RuntimeError, match="did not reproduce"):
        evaluate_candidates(stub, candidates, seed=SEED, gamma=GAMMA)


def test_a_different_start_is_an_error_on_execution():
    stub = _StubEnv(starts=[np.zeros(8), np.ones(8)])
    candidate = CandidatePolicy("left_engine", _fixed_action_policy(1))
    with pytest.raises(RuntimeError, match="did not reproduce"):
        execute_candidate(stub, candidate, seed=SEED, gamma=GAMMA)


def test_gaps_are_against_the_baseline_from_the_same_seed(env):
    candidates = _ci_candidates(env)
    baseline, records = evaluate_candidates(env, candidates, seed=SEED, gamma=GAMMA)

    base_r, base_n, _ = run_episode(env, NOOP_BASELINE, seed=SEED, gamma=GAMMA)
    assert baseline["baseline_payoff"] == pytest.approx(base_r)
    np.testing.assert_allclose(baseline["baseline_motives"], base_n)

    assert "noop" not in _by_id(records)
    assert len(records) == len(candidates) - 1
    for record in records:
        candidate = next(c for c in candidates if c.skill_id == record.skill_id)
        r_hat, n_hat, _ = run_episode(env, candidate, seed=SEED, gamma=GAMMA)
        assert record.delta_r == pytest.approx(r_hat - base_r, abs=1e-4)
        np.testing.assert_allclose(record.delta_n, n_hat - base_n, atol=1e-3)


def test_record_format(env):
    _, records = evaluate_candidates(env, _ci_candidates(env), seed=SEED, gamma=GAMMA)
    assert records
    for record in records:
        assert len(record.delta_n) == 2
        assert record.gate_type == "PDS"
        assert record.is_certified is False
        assert record.baseline_id == "noop"
        assert record.metadata["seed"] == SEED
        for key in ("steps", "landing_success", "crashed"):
            assert key in record.metadata


def test_evaluate_candidates_leaves_the_torch_state_untouched(env):
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    evaluate_candidates(env, _ci_candidates(env), seed=SEED, gamma=GAMMA)
    assert torch.equal(torch.get_rng_state(), before)


# ---------------------------------------------------------------------------
# execute_candidate / episode_outcome
# ---------------------------------------------------------------------------


def test_execution_matches_evaluation(env):
    candidates = _ci_candidates(env)
    _, records = evaluate_candidates(env, candidates, seed=SEED, gamma=GAMMA)

    for record in records:
        candidate = next(c for c in candidates if c.skill_id == record.skill_id)
        delta_r, delta_n, info = execute_candidate(env, candidate, seed=SEED, gamma=GAMMA)
        assert delta_r == pytest.approx(record.delta_r, abs=1e-6), record.skill_id
        np.testing.assert_allclose(delta_n, record.delta_n, atol=1e-6)
        assert info["steps"] == record.metadata["steps"]


def test_executing_the_baseline_is_a_zero_improvement(env):
    delta_r, delta_n, _ = execute_candidate(env, NOOP_BASELINE, seed=SEED, gamma=GAMMA)
    assert delta_r == 0.0
    np.testing.assert_array_equal(delta_n, 0.0)


@pytest.mark.parametrize("max_steps", [None, 50])
def test_episode_outcome_is_a_valid_skill_outcome(env, max_steps):
    delta_r, delta_n, info = execute_candidate(
        env, CandidatePolicy("main_engine", _fixed_action_policy(2)),
        seed=SEED, gamma=GAMMA, max_steps=max_steps,
    )
    outcome = episode_outcome(delta_r, delta_n, info, admitted=False, max_steps=max_steps)

    assert isinstance(outcome, SkillOutcome)
    assert np.isfinite(outcome.delta_r)
    assert np.all(np.isfinite(outcome.delta_n))
    assert outcome.delta_n.shape == (2,)
    assert outcome.admitted is False
    assert 0.0 <= outcome.effort <= 1.0
    limit = EPISODE_STEP_LIMIT if max_steps is None else max_steps
    assert outcome.effort == pytest.approx(info["steps"] / limit)


def test_effort_is_clipped_to_one():
    outcome = episode_outcome(0.0, [0.0, 0.0], {"steps": 80}, admitted=True, max_steps=40)
    assert outcome.effort == 1.0


def test_effort_defaults_to_the_limit_the_episode_ran_with(env):
    delta_r, delta_n, info = execute_candidate(
        env, CandidatePolicy("noop", _fixed_action_policy(0)), seed=SEED, gamma=GAMMA, max_steps=50
    )
    outcome = episode_outcome(delta_r, delta_n, info, admitted=True)
    assert outcome.effort == pytest.approx(info["steps"] / 50)


def test_effort_limit_is_capped_by_the_env_time_limit():
    info = {"steps": EPISODE_STEP_LIMIT}
    outcome = episode_outcome(
        0.0, [0.0, 0.0], info, admitted=True, max_steps=10 * EPISODE_STEP_LIMIT
    )
    assert outcome.effort == 1.0


def _stimulus(outcome: SkillOutcome, scales):
    payoff_scale, motive_scale = scales
    return build_stimulus_values(
        outcome, weights=WEIGHTS, payoff_scale=payoff_scale, motive_scale=motive_scale
    )


def _assert_worse_reads_worse(worse: SkillOutcome, better: SkillOutcome, scales) -> None:
    bad, good = _stimulus(worse, scales), _stimulus(better, scales)
    assert bad.conduciveness < good.conduciveness
    assert bad.risk > good.risk
    for values in (bad, good):
        for name in ("novelty", "conduciveness", "risk", "effort"):
            assert 0.0 <= getattr(values, name) <= 1.0


def test_worse_than_baseline_reads_as_less_conducive_and_riskier(env):
    _, records = evaluate_candidates(env, _ci_candidates(env), seed=SEED, gamma=GAMMA)
    scales = appraisal_scales(records)
    worst = min(records, key=lambda r: r.delta_r + float(np.dot(WEIGHTS, r.delta_n)))

    info = {"steps": worst.metadata["steps"]}
    worse = episode_outcome(worst.delta_r, worst.delta_n, info, admitted=True)
    # The same pair with the roles swapped: the baseline measured against the
    # candidate is exactly as much better as the candidate was worse.
    better = episode_outcome(
        -worst.delta_r, -np.asarray(worst.delta_n), info, admitted=True
    )
    _assert_worse_reads_worse(worse, better, scales)


@pytest.mark.parametrize("weights", [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])
def test_outcome_works_with_any_two_objective_weights(env, weights):
    """The adapter takes no weights itself; any valid 2-element vector works."""
    _, records = evaluate_candidates(env, _ci_candidates(env), seed=SEED, gamma=GAMMA)
    payoff_scale, motive_scale = appraisal_scales(records)
    delta_r, delta_n, info = execute_candidate(
        env, CandidatePolicy("right_engine", _fixed_action_policy(3)), seed=SEED, gamma=GAMMA
    )
    values = build_stimulus_values(
        episode_outcome(delta_r, delta_n, info, admitted=True),
        weights=np.asarray(weights),
        payoff_scale=payoff_scale,
        motive_scale=motive_scale,
    )
    for name in ("novelty", "conduciveness", "risk", "effort"):
        assert 0.0 <= getattr(values, name) <= 1.0


# ---------------------------------------------------------------------------
# appraisal_scales
# ---------------------------------------------------------------------------


def test_appraisal_scales_are_mean_absolute_gaps():
    records = [
        SimpleNamespace(delta_r=20.0, delta_n=(100.0, -300.0)),
        SimpleNamespace(delta_r=-40.0, delta_n=(0.0, 200.0)),
    ]
    assert appraisal_scales(records) == pytest.approx((30.0, 150.0))


def test_appraisal_scales_never_fall_below_the_floor():
    zero = [SimpleNamespace(delta_r=0.0, delta_n=(0.0, 0.0))]
    payoff_scale, motive_scale = appraisal_scales(zero)
    assert payoff_scale >= APPRAISAL_SCALE_FLOOR
    assert motive_scale >= APPRAISAL_SCALE_FLOOR

    tiny = [SimpleNamespace(delta_r=0.5, delta_n=(0.1, -0.2))]
    assert appraisal_scales(tiny, floor=4.0) == (4.0, 4.0)


def test_appraisal_scales_on_real_records_respect_the_floor(env):
    _, records = evaluate_candidates(env, _ci_candidates(env), seed=SEED, gamma=GAMMA)
    payoff_scale, motive_scale = appraisal_scales(records)
    assert payoff_scale >= APPRAISAL_SCALE_FLOOR
    assert motive_scale >= APPRAISAL_SCALE_FLOOR


# ---------------------------------------------------------------------------
# PPO candidates
# ---------------------------------------------------------------------------


@pytest.fixture()
def ppo_candidates(env):
    from data_collector.collect_candidate_sets import build_default_candidate_policies

    return build_default_candidate_policies(env, pilot_checkpoint=str(PILOT_CHECKPOINT))


@requires_pilot
def test_stochastic_ppo_replays_and_leaves_the_torch_state_untouched(env, ppo_candidates):
    stochastic = next(c for c in ppo_candidates if c.skill_id == "ppo_stochastic")

    torch.manual_seed(99)
    before = torch.get_rng_state().clone()
    first = run_episode(env, stochastic, seed=SEED, gamma=GAMMA)
    assert torch.equal(torch.get_rng_state(), before)
    torch.rand(5)  # move the global state; the seed must still replay
    second = run_episode(env, stochastic, seed=SEED, gamma=GAMMA)

    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2]["steps"] == second[2]["steps"]


@requires_pilot
def test_evaluating_ppo_candidates_leaves_the_torch_state_untouched(env, ppo_candidates):
    torch.manual_seed(4321)
    before = torch.get_rng_state().clone()
    evaluate_candidates(env, ppo_candidates, seed=SEED, gamma=GAMMA)
    assert torch.equal(torch.get_rng_state(), before)


@requires_pilot
def test_crash_reads_worse_than_landing(env, ppo_candidates):
    _, records = evaluate_candidates(env, ppo_candidates, seed=SEED, gamma=GAMMA)
    scales = appraisal_scales(records)
    by_id = _by_id(records)
    landed, crashed = by_id["ppo_deterministic"], by_id["left_engine"]
    assert landed.metadata["landing_success"]
    assert crashed.metadata["crashed"]

    def outcome(record):
        return episode_outcome(
            record.delta_r, record.delta_n, {"steps": record.metadata["steps"]}, admitted=True
        )

    _assert_worse_reads_worse(outcome(crashed), outcome(landed), scales)


# ---------------------------------------------------------------------------
# MetaMo
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not metamo_is_available(),
    reason="MetaMo checkout not found (see bridge/_loader.py for search paths)",
)
def test_governor_accepts_the_episode_outcome(env):
    from bridge.governor import MetaMoGovernor
    from bridge.weights import DEFAULT_GOAL_AFFINITY, DEFAULT_MODULATOR_GAIN

    _, records = evaluate_candidates(env, _ci_candidates(env), seed=SEED, gamma=GAMMA)
    payoff_scale, motive_scale = appraisal_scales(records)
    # Any valid 2-objective mapping will do; the adapter is weight-agnostic.
    governor = MetaMoGovernor(
        goal_affinity=DEFAULT_GOAL_AFFINITY[:2],
        modulator_gain=DEFAULT_MODULATOR_GAIN[:2],
        payoff_scale=payoff_scale,
        motive_scale=motive_scale,
    )
    assert governor.num_objectives == 2

    delta_r, delta_n, info = execute_candidate(
        env, CandidatePolicy("left_engine", _fixed_action_policy(1)), seed=SEED, gamma=GAMMA
    )
    signal = governor.step(episode_outcome(delta_r, delta_n, info, admitted=False))

    signal.validate()
    assert signal.weights.shape == (2,)
    assert float(np.sum(signal.weights)) == pytest.approx(1.0, abs=1e-9)

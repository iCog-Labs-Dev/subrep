"""End-to-end tests for the MetaMo -> SubRep loop on MO-LunarLander.

The two LunarLander pieces are each tested on their own:

    tests/test_lunar_lander_metamo.py    the Safety/Fuel weight mapping
    tests/test_lunar_lander_rollout.py   same-seed scoring and episode outcomes

This file checks that they work TOGETHER in the real loop, on the real
environment rather than the Minecraft stub:

  1. the mapped governor emits weights and risk budgets (env/lunar_lander_metamo.py);
  2. the adapter scores every candidate against the noop baseline from the same
     seed (env/lunar_lander_rollout.py);
  3. SubRep's certification pipeline admits candidates under those budgets and
     the controller selects one under those weights (bridge/controller.py);
  4. the chosen policy flies, or the baseline when nothing is admitted, and its
     outcome is fed back into MetaMo through the governor.

One decision step is one episode, on a new seed each step, so the start
observation (the certification context) differs every step and the
certification cache never reuses an earlier verdict.

Known behaviour these tests record without changing:

* The risk budget cannot decide a LunarLander admission. Objective changes are
  in the hundreds while epsilon is at most 0.5, so crash-only candidates are
  rejected and landings admitted whatever epsilon is.
* MetaMo raises caution on every step, so the tests that check MetaMo's reaction
  to an outcome compare two fresh governors fed different outcomes, rather than
  one governor before and after.

The core tests use the fixed-action and random candidates, which need no PPO
checkpoint. The PPO tests skip when models/pilot_ppo.pt is missing, and the
whole file skips without a MetaMo checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pytest

from bridge._loader import is_available
from bridge.budget import BudgetCoefficients
from bridge.protocol import SkillOutcome
from data_collector.collect_candidate_sets import (
    CandidatePolicy,
    _fixed_action_policy,
    _random_policy,
)
from env.lunar_lander_metamo import make_lunar_lander_governor
from env.lunar_lander_rollout import (
    NOOP_BASELINE,
    appraisal_scales,
    episode_outcome,
    evaluate_candidates,
    execute_candidate,
)
from env.lunar_lander_wrapper import SubRepEnv
from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_runtime_pipeline import (
    RuntimeCertificationPipeline,
    RuntimePipelineConfig,
)
from utils.weight_set_store import WeightSetStore

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="MetaMo checkout not found (see bridge/_loader.py for search paths)",
)

SEEDS = (1, 2, 3)
CONTROLLER_SEED = 42
NUM_OBJECTIVES = 2
OBS_DIM = 8
SAFETY = 0
EPSILON_MAX = BudgetCoefficients().epsilon_max
# left_engine, main_engine, right_engine and random; noop is the baseline.
CI_SCORED_CANDIDATES = 4
PILOT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "pilot_ppo.pt"

requires_pilot = pytest.mark.skipif(
    not PILOT_CHECKPOINT.is_file(),
    reason="PPO checkpoint models/pilot_ppo.pt not found",
)


# ---------------------------------------------------------------------------
# Building and running the loop
# ---------------------------------------------------------------------------


def ci_candidates(env: SubRepEnv) -> Tuple[CandidatePolicy, ...]:
    """The default candidates that need no PPO checkpoint."""
    return (
        CandidatePolicy("noop", _fixed_action_policy(0)),
        CandidatePolicy("left_engine", _fixed_action_policy(1)),
        CandidatePolicy("main_engine", _fixed_action_policy(2)),
        CandidatePolicy("right_engine", _fixed_action_policy(3)),
        CandidatePolicy("random", _random_policy(env)),
    )


def ppo_candidates(env: SubRepEnv) -> Tuple[CandidatePolicy, ...]:
    from data_collector.collect_candidate_sets import build_default_candidate_policies

    return build_default_candidate_policies(env, pilot_checkpoint=str(PILOT_CHECKPOINT))


def make_pipeline() -> RuntimeCertificationPipeline:
    """A fresh, seeded pipeline with an untrained MDN and the PDS gate only.

    Seeded so every run certifies with the same model. CVaR stays off: an
    untrained MDN makes the CVaR gate admit nearly everything, which would hide
    the PDS decisions these tests are about.
    """
    import torch

    torch.manual_seed(CONTROLLER_SEED)
    model = MotiveDecompositionNetwork(input_dim=OBS_DIM, num_objectives=NUM_OBJECTIVES)
    model.eval()
    return RuntimeCertificationPipeline(
        model=model,
        weight_store=WeightSetStore(num_objectives=NUM_OBJECTIVES),
        config=RuntimePipelineConfig(
            gate_type="PDS",
            use_cvar=False,
            train_support_after_certify=False,
        ),
    )


def run_loop(
    candidates_for: Callable[[SubRepEnv], Tuple[CandidatePolicy, ...]],
    seeds=SEEDS,
):
    """Run one decision step per seed through the real stack.

    Returns (controller, governor, records_per_step), where records_per_step
    holds each step's scored candidate records before certification.
    """
    from bridge.controller import MetaMoController

    env = SubRepEnv(seed=seeds[0])
    candidates = candidates_for(env)
    policies: Dict[str, CandidatePolicy] = {c.skill_id: c for c in candidates}

    # The appraisal scales come from the first step's candidates.
    first = evaluate_candidates(env, candidates, seed=seeds[0])
    payoff_scale, motive_scale = appraisal_scales(first[1])
    governor = make_lunar_lander_governor(
        payoff_scale=payoff_scale, motive_scale=motive_scale
    )
    controller = MetaMoController(governor, make_pipeline(), seed=CONTROLLER_SEED)

    records_per_step: List[list] = []
    try:
        for index, seed in enumerate(seeds):
            baseline_stats, records = (
                first if index == 0 else evaluate_candidates(env, candidates, seed=seed)
            )

            def outcome_for(selected, _seed=seed) -> SkillOutcome:
                # Nothing admitted: fly the baseline, as the demos do.
                policy = NOOP_BASELINE if selected is None else policies[selected.skill_id]
                delta_r, delta_n, info = execute_candidate(env, policy, seed=_seed)
                return episode_outcome(delta_r, delta_n, info, admitted=selected is not None)

            controller.step(
                context=np.asarray(baseline_stats["info"]["start_obs"]),
                candidate_skills=records,
                baseline_stats=baseline_stats,
                outcome_for=outcome_for,
            )
            records_per_step.append(records)
    finally:
        env.close()
    return controller, governor, records_per_step


def fresh_governor_after(policy: CandidatePolicy, *, seed: int = SEEDS[0]):
    """A fresh mapped governor stepped once with `policy`'s real outcome.

    The appraisal scales come from the CI candidates on the same seed, so every
    governor built here uses the same scales. Returns (governor, flown-episode
    info, (payoff_scale, motive_scale)).
    """
    env = SubRepEnv(seed=seed)
    try:
        _, records = evaluate_candidates(env, ci_candidates(env), seed=seed)
        payoff_scale, motive_scale = appraisal_scales(records)
        delta_r, delta_n, info = execute_candidate(env, policy, seed=seed)
    finally:
        env.close()
    governor = make_lunar_lander_governor(
        payoff_scale=payoff_scale, motive_scale=motive_scale
    )
    governor.step(episode_outcome(delta_r, delta_n, info, admitted=True))
    return governor, info, (payoff_scale, motive_scale)


def securing(governor) -> float:
    return float(governor.modulator_summary()["securing"])


@pytest.fixture(scope="module")
def ci_run():
    """One CI-candidate loop, shared by the tests that only read it."""
    return run_loop(ci_candidates)


# ---------------------------------------------------------------------------
# The loop, with candidates that need no checkpoint
# ---------------------------------------------------------------------------


def test_the_loop_runs_end_to_end_with_both_pieces(ci_run):
    """Mapping and adapter connect through the controller on every step."""
    controller, governor, _ = ci_run

    assert governor.num_objectives == NUM_OBJECTIVES
    assert len(controller.history) == len(SEEDS)
    for record in controller.history:
        weights = np.asarray(record.weights)
        assert weights.shape == (NUM_OBJECTIVES,)
        assert np.all(np.isfinite(weights)) and np.all(weights >= 0.0)
        assert float(np.sum(weights)) == pytest.approx(1.0)
        assert 0.0 <= record.pds_epsilon <= EPSILON_MAX
        # Every candidate except the noop baseline is scored and certified.
        assert record.candidate_count == CI_SCORED_CANDIDATES


def test_crash_only_candidates_are_rejected_and_the_loop_abstains(ci_run):
    """Every CI candidate crashes like the baseline, so PDS admits none.

    The loop must still complete: it flies the baseline, reports the
    abstention, and MetaMo's state still moves.
    """
    controller, governor, _ = ci_run

    for record in controller.history:
        assert record.admitted_count == 0
        assert record.selected_skill_id is None
    assert not np.allclose(governor.state.M, 0.5)


def test_epsilon_cannot_change_a_lunar_lander_admission(ci_run):
    """Records the risk-budget scale on MO-LunarLander.

    Every CI candidate's PDS margin (delta_r + min delta_n) lies below
    -EPSILON_MAX, the loosest budget MetaMo can emit, so no epsilon could admit
    it. If objective changes are normalised later, this test is expected to
    fail and should be updated together with that change.
    """
    _, _, records_per_step = ci_run

    margins = [
        float(r.delta_r) + float(np.min(r.delta_n))
        for records in records_per_step
        for r in records
    ]
    assert margins
    assert max(margins) < -EPSILON_MAX


def test_the_same_seeds_reproduce_the_same_run(ci_run):
    """A second, fresh loop over the same seeds matches the first exactly."""
    first, _, _ = ci_run
    second, _, _ = run_loop(ci_candidates)

    assert len(first.history) == len(second.history)
    for a, b in zip(first.history, second.history):
        assert a.selected_skill_id == b.selected_skill_id
        assert a.admitted_count == b.admitted_count
        np.testing.assert_allclose(a.weights, b.weights)
        assert a.pds_epsilon == pytest.approx(b.pds_epsilon)
        assert a.cvar_tail_level == pytest.approx(b.cvar_tail_level)
        assert a.modulators == pytest.approx(b.modulators)


def test_a_real_crash_raises_caution_through_the_mapped_governor():
    """A crash from the adapter reaches the mapping as caution.

    MetaMo raises caution a little on every step, so the crash is compared with
    a no-change outcome of the same effort fed to an identical fresh governor.
    """
    crash_policy = CandidatePolicy("left_engine", _fixed_action_policy(1))
    after_crash, info, (payoff_scale, motive_scale) = fresh_governor_after(crash_policy)
    assert info["crashed"]

    # Same effort as the crash, but no change in reward or objectives.
    no_change = episode_outcome(0.0, np.zeros(NUM_OBJECTIVES), info, admitted=True)
    after_nothing = make_lunar_lander_governor(
        payoff_scale=payoff_scale, motive_scale=motive_scale
    )
    after_nothing.step(no_change)

    crash_signal, nothing_signal = after_crash.signal(), after_nothing.signal()
    assert securing(after_crash) > securing(after_nothing)
    assert crash_signal.weights[SAFETY] > nothing_signal.weights[SAFETY]
    assert crash_signal.pds_epsilon <= nothing_signal.pds_epsilon


# ---------------------------------------------------------------------------
# With the PPO candidates
# ---------------------------------------------------------------------------


@requires_pilot
def test_ppo_landings_are_admitted_and_selected():
    """With landings available, the loop makes a real choice.

    PPO policies are the only candidates that land, so they are the only ones
    that can pass PDS against the crashing baseline.
    """
    controller, _, _ = run_loop(ppo_candidates)

    assert any(record.admitted_count > 0 for record in controller.history)
    for record in controller.history:
        if record.selected_skill_id is not None:
            assert record.admitted_count > 0
            assert record.selected_skill_id.startswith("ppo_")


@requires_pilot
def test_a_landing_reads_safer_than_a_crash():
    """Two identical fresh governors, one fed a real landing, one a real crash.

    MetaMo should end up more cautious, and SubRep's Safety weight higher, after
    the crash than after the landing.
    """
    env = SubRepEnv(seed=SEEDS[0])
    try:
        landing_policy = next(c for c in ppo_candidates(env) if c.skill_id == "ppo_deterministic")
    finally:
        env.close()

    after_landing, landing_info, _ = fresh_governor_after(landing_policy)
    after_crash, crash_info, _ = fresh_governor_after(
        CandidatePolicy("left_engine", _fixed_action_policy(1))
    )
    assert landing_info["landing_success"]
    assert crash_info["crashed"]

    landing_signal, crash_signal = after_landing.signal(), after_crash.signal()
    assert securing(after_crash) > securing(after_landing)
    assert crash_signal.weights[SAFETY] > landing_signal.weights[SAFETY]
    assert crash_signal.pds_epsilon <= landing_signal.pds_epsilon

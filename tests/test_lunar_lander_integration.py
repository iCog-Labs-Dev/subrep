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

* Without an epsilon scale, the risk budget cannot decide a LunarLander
  admission. Objective changes are in the hundreds while epsilon is at most
  0.5, so crash-only candidates are rejected and landings admitted whatever
  epsilon is. The scaled tests show that with the scale (bridge/epsilon_scale.py)
  epsilon decides a real trade-off landing, and crashes stay rejected.
* MetaMo raises caution on every step, so the tests that check MetaMo's reaction
  to an outcome compare two fresh governors fed different outcomes, rather than
  one governor before and after.
* Certification and execution use the same seed. Each step certifies the
  candidates and flies the chosen one from the same start, so these tests show
  the loop is wired correctly, not that a certificate holds on unseen starts.

The core tests use the fixed-action and random candidates, which need no PPO
checkpoint. The PPO tests skip when models/pilot_ppo.pt is missing, and the
whole file skips without a MetaMo checkout.
"""

from __future__ import annotations

import itertools
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
    """Records the risk-budget scale on MO-LunarLander, without an epsilon scale.

    Every CI candidate's PDS margin (delta_r + min delta_n) lies below
    -EPSILON_MAX, the loosest raw budget MetaMo can emit, so no epsilon could
    admit it. The epsilon scale (bridge/epsilon_scale.py) is what lets epsilon
    decide LunarLander admissions; see the scaled tests below.
    """
    _, _, records_per_step = ci_run

    margins = [
        float(r.delta_r) + float(np.min(r.delta_n))
        for records in records_per_step
        for r in records
    ]
    assert margins
    assert max(margins) < -EPSILON_MAX


def _certify_with_scale(baseline_stats, records, epsilon: float):
    """Certify `records` once, with the LunarLander epsilon scale, on a fresh pipeline."""
    from bridge.controller import MetaMoController
    from bridge.governor import FakeGovernor
    from bridge.protocol import GovernorSignal
    from env.lunar_lander_metamo import lunar_lander_epsilon_scale

    sig = GovernorSignal(
        weights=np.full(NUM_OBJECTIVES, 1.0 / NUM_OBJECTIVES),
        pds_epsilon=epsilon,
        cvar_tail_level=0.1,
    )
    controller = MetaMoController(
        FakeGovernor([sig]),
        make_pipeline(),
        seed=CONTROLLER_SEED,
        epsilon_scale=lunar_lander_epsilon_scale(),
    )
    return controller.certify(
        context=np.asarray(baseline_stats["info"]["start_obs"]),
        candidate_skills=records,
        baseline_stats=baseline_stats,
        signal=sig,
    )


def _switch_weights(records) -> List[float]:
    """Safety weights in (0, 1) where the selector's pick among `records` changes.

    Two objectives, w = (s, 1 - s): each record's score is a line in s, so the
    pick can only change where two lines cross. Ties go to the lower skill_id,
    as in `MetaMoController.select`.
    """
    def pick(s: float) -> str:
        return min(
            records,
            key=lambda r: (-(r.delta_r + s * r.delta_n[0] + (1.0 - s) * r.delta_n[1]), r.skill_id),
        ).skill_id

    crossings = set()
    for a, b in itertools.combinations(records, 2):
        slope = (a.delta_n[0] - a.delta_n[1]) - (b.delta_n[0] - b.delta_n[1])
        if slope:
            s = -((a.delta_r + a.delta_n[1]) - (b.delta_r + b.delta_n[1])) / slope
            if 0.0 < s < 1.0:
                crossings.add(float(s))
    edges = [0.0] + sorted(crossings) + [1.0]
    picks = [pick((lo + hi) / 2.0) for lo, hi in zip(edges[:-1], edges[1:])]
    return [s for s, below, above in zip(edges[1:-1], picks[:-1], picks[1:]) if below != above]


def test_real_crashes_are_rejected_at_the_loosest_epsilon():
    """With the epsilon scale, eps_max tolerates a loss of about 150, more than
    some crash margins; the task-failure guard still rejects every crash."""
    from bridge.epsilon_scale import TASK_FAILURE

    env = SubRepEnv(seed=SEEDS[0])
    try:
        for seed in SEEDS:
            baseline_stats, records = evaluate_candidates(env, ci_candidates(env), seed=seed)
            certified = _certify_with_scale(baseline_stats, records, EPSILON_MAX)
            assert not any(r.is_certified for r in certified), seed
            for r in certified:
                if r.metadata.get("crashed"):
                    assert r.metadata["rejection_reason"] == TASK_FAILURE, (seed, r.skill_id)
    finally:
        env.close()


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
def test_a_tighter_epsilon_rejects_a_real_trade_off_landing():
    """A real fuel-wasting landing is admitted at eps_max and rejected at a tighter eps.

    The candidate is chosen at runtime: the first seed in 1-20 where a
    trade-off candidate lands with a margin in (-0.5 * U, 0), the band the
    scaled tolerance spans. The tighter eps puts the tolerance at 0.9 of that
    margin, so nothing is hard-coded but the band.
    """
    from env.lunar_lander_candidates import build_tradeoff_candidate_policies
    from env.lunar_lander_metamo import LUNAR_LANDER_EPSILON_UNIT

    unit = LUNAR_LANDER_EPSILON_UNIT
    env = SubRepEnv(seed=1)
    try:
        candidates = build_tradeoff_candidate_policies(pilot_checkpoint=str(PILOT_CHECKPOINT))
        found = None
        for seed in range(1, 21):
            baseline_stats, records = evaluate_candidates(env, candidates, seed=seed)
            for r in records:
                margin = float(r.delta_r) + float(np.min(r.delta_n))
                if r.metadata["landing_success"] and -EPSILON_MAX * unit < margin < 0.0:
                    found = (baseline_stats, r, margin)
                    break
            if found:
                break
    finally:
        env.close()
    assert found, "no trade-off landing within the scaled band on seeds 1-20"
    baseline_stats, trade_off, margin = found

    (loose,) = _certify_with_scale(baseline_stats, [trade_off], EPSILON_MAX)
    (tight,) = _certify_with_scale(baseline_stats, [trade_off], 0.9 * abs(margin) / unit)
    assert loose.is_certified, (trade_off.skill_id, margin)
    assert not tight.is_certified, (trade_off.skill_id, margin)
    assert "rejection_reason" not in tight.metadata  # rejected by PDS, not by the guard


@requires_pilot
def test_safety_and_fuel_priorities_select_different_admitted_skills():
    """Two priority settings pick different skills from the same admitted set.

    Real records from the demo's trade-off candidates, certified with the
    epsilon scale at eps 0.1. The seed is chosen at runtime: the first in 1-10
    whose admitted set has a pick switch with the Safety weight s* in
    (0.2, 0.8). The two settings are s* - delta and s* + delta, with delta at
    most 0.05 and less than half the distance to any other switch on that seed.
    """
    from bridge.controller import MetaMoController
    from bridge.governor import FakeGovernor
    from bridge.protocol import GovernorSignal
    from data_collector.collect_candidate_sets import build_extended_candidate_policies
    from env.lunar_lander_candidates import build_tradeoff_candidate_policies
    from env.lunar_lander_metamo import lunar_lander_epsilon_scale

    epsilon = 0.1
    env = SubRepEnv(seed=1)
    try:
        candidates = tuple(
            build_extended_candidate_policies(env, pilot_checkpoint=str(PILOT_CHECKPOINT))
        ) + build_tradeoff_candidate_policies(pilot_checkpoint=str(PILOT_CHECKPOINT))
        found = None
        for seed in range(1, 11):
            baseline_stats, records = evaluate_candidates(env, candidates, seed=seed)
            admitted = [
                r for r in _certify_with_scale(baseline_stats, records, epsilon) if r.is_certified
            ]
            points = _switch_weights(admitted)
            inside = [s for s in points if 0.2 < s < 0.8]
            if inside:
                found = (seed, baseline_stats, records, inside[0], points)
                break
    finally:
        env.close()
    assert found, "no seed in 1-10 has a pick switch with the Safety weight in (0.2, 0.8)"
    seed, baseline_stats, records, s_star, points = found
    delta = min([0.05] + [abs(s - s_star) / 2.0 for s in points if s != s_star])

    def certify_and_pick(safety_weight: float):
        sig = GovernorSignal(
            weights=np.array([safety_weight, 1.0 - safety_weight]),
            pds_epsilon=epsilon,
            cvar_tail_level=0.1,
        )
        controller = MetaMoController(
            FakeGovernor([sig]),
            make_pipeline(),
            seed=CONTROLLER_SEED,
            epsilon_scale=lunar_lander_epsilon_scale(),
        )
        certified = controller.certify(
            context=np.asarray(baseline_stats["info"]["start_obs"]),
            candidate_skills=records,
            baseline_stats=baseline_stats,
            signal=sig,
        )
        return {r.skill_id for r in certified if r.is_certified}, controller.select(certified, sig)[0]

    fuel_set, fuel_pick = certify_and_pick(s_star - delta)
    safety_set, safety_pick = certify_and_pick(s_star + delta)
    assert fuel_set == safety_set, seed
    assert fuel_pick != safety_pick, (seed, s_star, fuel_pick)
    assert fuel_pick in fuel_set and safety_pick in safety_set


@requires_pilot
def test_real_crashes_then_real_landings_raise_then_relax_caution():
    """One governor, fed real flown episodes with the task risk.

    4 left_engine crashes, then 8 ppo_deterministic landings, on seeds 1-12.
    Caution peaks with the crashes, then falls, and eps comes back up.
    """
    env = SubRepEnv(seed=1)
    try:
        landing_policy = next(c for c in ppo_candidates(env) if c.skill_id == "ppo_deterministic")
        crash_policy = CandidatePolicy("left_engine", _fixed_action_policy(1))
        _, records = evaluate_candidates(env, ci_candidates(env), seed=1)
        payoff_scale, motive_scale = appraisal_scales(records)
        governor = make_lunar_lander_governor(
            payoff_scale=payoff_scale, motive_scale=motive_scale
        )
        trace = []
        for step, seed in enumerate(range(1, 13)):
            policy = crash_policy if step < 4 else landing_policy
            delta_r, delta_n, info = execute_candidate(env, policy, seed=seed)
            assert info["crashed"] if step < 4 else info["landing_success"], (step, seed)
            governor.step(episode_outcome(delta_r, delta_n, info, admitted=True, task_risk=True))
            trace.append((securing(governor), governor.signal().pds_epsilon))
    finally:
        env.close()

    peak = max(range(len(trace)), key=lambda i: trace[i][0])
    assert peak <= 4, "caution should peak during or just after the crashes"
    assert trace[-1][0] < trace[peak][0] - 0.3
    assert trace[-1][1] > trace[peak][1]


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


# ---------------------------------------------------------------------------
# The trained MDN and the gate configurations
# ---------------------------------------------------------------------------

TRAINED_MDN = Path(__file__).resolve().parent.parent / "models" / "lunar_mdn.pth"


def test_the_trained_lunar_mdn_loads_into_the_runtime_pipeline():
    """The tracked MDN gives valid concentrations and a non-empty region at start observations."""
    from demo.run_lunar_lander_metamo import build_pipeline, mdn_view

    pipeline = build_pipeline(seed=CONTROLLER_SEED, gates="cvar", cvar_samples=1000, mdn_path=TRAINED_MDN)
    env = SubRepEnv(seed=SEEDS[0])
    try:
        for seed in SEEDS:
            observation, _ = env.reset(seed=seed)
            view = mdn_view(pipeline.model, observation)
            concentration = np.asarray(view["concentration"])
            support = np.asarray(view["support"])
            assert concentration.shape == (NUM_OBJECTIVES,)
            assert np.all(np.isfinite(concentration)) and np.all(concentration > 0.0)
            assert np.all((support >= 0.0) & (support <= 1.0))
            assert support.sum() >= 1.0 - 1e-6
            # Fitted to a schedule that weighs Safety 0.7-0.8.
            assert 0.65 <= view["mean"][SAFETY] <= 0.85
    finally:
        env.close()


@pytest.fixture(scope="module")
def ci_records():
    """(baseline_stats, records) on SEEDS for the candidates that all crash."""
    env = SubRepEnv(seed=SEEDS[0])
    try:
        candidates = ci_candidates(env)
        return [evaluate_candidates(env, candidates, seed=seed) for seed in SEEDS]
    finally:
        env.close()


@pytest.mark.parametrize("mdn", ["untrained", "trained"])
@pytest.mark.parametrize("gates", ["pds", "cvar", "pds-or-cvar"])
def test_no_configuration_admits_a_task_failure(ci_records, gates, mdn):
    """With the epsilon scale, crashes are rejected before any gate, at the loosest budgets."""
    from bridge.controller import MetaMoController
    from bridge.governor import FakeGovernor
    from bridge.protocol import GovernorSignal
    from demo.run_lunar_lander_metamo import build_pipeline
    from env.lunar_lander_metamo import lunar_lander_epsilon_scale

    scale = lunar_lander_epsilon_scale()
    sig = GovernorSignal(
        weights=np.full(NUM_OBJECTIVES, 1.0 / NUM_OBJECTIVES),
        pds_epsilon=EPSILON_MAX,
        cvar_tail_level=BudgetCoefficients().alpha_max,
    )
    for baseline_stats, records in ci_records:
        assert all(scale.fails_task(r, baseline_stats) for r in records)
        controller = MetaMoController(
            FakeGovernor([sig]),
            build_pipeline(
                seed=CONTROLLER_SEED,
                gates=gates,
                cvar_samples=1000,
                mdn_path=None if mdn == "untrained" else TRAINED_MDN,
            ),
            seed=CONTROLLER_SEED,
            epsilon_scale=scale,
        )
        certified = controller.certify(
            context=np.asarray(baseline_stats["info"]["start_obs"]),
            candidate_skills=records,
            baseline_stats=baseline_stats,
            signal=sig,
        )
        assert not any(r.is_certified for r in certified), (gates, mdn)

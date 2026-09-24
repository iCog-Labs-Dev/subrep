"""End-to-end tests for the MetaMo -> SubRep loop.

These are the only tests that assemble the REAL stack together: real
`MetaMoGovernor` (driving real MetaMo), real `RuntimeCertificationPipeline`,
real CDS/PDS/CVaR gates, real selection. Every other bridge test substitutes a
fake for one half or the other:

    test_bridge_governor.py     real governor, no pipeline
    test_bridge_controller.py   FakeGovernor + RecordingPipeline
    test_mdn_runtime_pipeline.py    real pipeline, no governor

so emergent coupling bugs between the two halves are invisible to them. That
is what this file exists to catch.

Scope discipline: assert only what REQUIRES the real stack. Forwarding,
selection mechanics and stimulus scaling are already covered with fakes in
test_bridge_controller.py and are not restated here.

------------------------------------------------------------------------------
EVIDENCE CAVEAT
------------------------------------------------------------------------------
Everything here runs against `MinecraftStubEnv`, a deterministic numpy
stand-in. These are CONTROLLED TEST OUTCOMES demonstrating that the coupling
behaves as designed. They are NOT real-environment validation -- no real
environment and no live agent are involved.

------------------------------------------------------------------------------
THREE TRAPS THIS FILE HAS TO AVOID
------------------------------------------------------------------------------
1. Certification is cached by `(context_key, skill_id)`
   (utils/mdn_runtime_pipeline.py:219-227). With a CONSTANT context, a second
   certification returns the cached verdict and budget changes have literally
   no effect. Tests either vary the context every step or build a fresh
   pipeline per certification.
2. OR gate semantics (`use_cvar=True, require_cds_or_cvar=True`) let CVaR
   admit what PDS rejects, and an untrained MDN admits nearly everything. Any
   assertion about epsilon or abstention runs with `use_cvar=False`.
3. Stub noise perturbs rollout margins, so the env is built with
   `noise_scale=0.0` wherever a margin matters.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pytest

from baseline.idle_policy import IdlePolicy
from baseline.improvement_calculator import ImprovementCalculator
from bridge._loader import is_available
from bridge.protocol import GovernorSignal, SkillOutcome
from env.minecraft_stub import SKILL_NAMES, MinecraftStubEnv
from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_runtime_pipeline import (
    RuntimeCertificationPipeline,
    RuntimePipelineConfig,
)
from utils.weight_set_store import WeightSetStore

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="MetaMo checkout not found (see bridge/_loader.py for search paths)",
)

GAMMA = 0.99
SEED = 42
NUM_OBJECTIVES = 6
SAFETY = 0  # index into phi(x)

# Survival-oriented starting goals, matching demo/run_metamo_pipeline.py.
# MetaMo's own default is tuned for a chat assistant and makes Reputation
# dominate from step 0, which is the wrong prior here.
MINECRAFT_INITIAL_GOALS = np.array(
    [0.70, 0.40, 0.30, 0.50, 0.40, 0.60, 0.50, 0.30], dtype=np.float64
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _discounted_rollout(env, action: int, *, seed: int) -> Tuple[float, np.ndarray]:
    """One episode always taking `action`.

    Matches IdlePolicy.run_baseline_episodes' discounting exactly
    (baseline/idle_policy.py:35-56) so results are directly comparable.
    """
    env.reset(seed=seed)
    discount = 1.0
    total_payoff = 0.0
    motives: Optional[np.ndarray] = None

    while True:
        _, reward_vec, terminated, truncated, _ = env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32)
        if motives is None:
            motives = np.zeros_like(reward_vec)
        total_payoff += discount * float(np.sum(reward_vec))
        motives += discount * reward_vec
        if terminated or truncated:
            break
        discount *= GAMMA

    return float(total_payoff), np.asarray(motives, dtype=np.float32)


def build_world(seed: int = SEED):
    """Noiseless env + idle baseline + one candidate per non-idle action."""
    env = MinecraftStubEnv(seed=seed, noise_scale=0.0)
    baseline = IdlePolicy(env=env, idle_action=0, gamma=GAMMA).run_baseline_episodes(
        num_episodes=2, seed=seed
    )
    calculator = ImprovementCalculator(baseline)

    candidates: List[CandidateSkillRecord] = []
    for action in range(1, len(SKILL_NAMES)):
        payoff, motives = _discounted_rollout(env, action, seed=seed)
        delta_r, delta_n = calculator.compute_improvements(
            skill_payoff=payoff, skill_motives=motives
        )
        candidates.append(
            CandidateSkillRecord(
                skill_id=SKILL_NAMES[action],
                delta_r=float(delta_r),
                delta_n=tuple(float(v) for v in delta_n),
                is_certified=False,
                gate_type="PDS",
                metadata={"action": action},
            )
        )

    obs, _ = env.reset(seed=seed)
    return env, baseline, candidates, obs


def make_pipeline(obs_dim: int, *, use_cvar: bool, cvar_samples: int = 1000):
    """A fresh pipeline. Fresh matters: permanence caches across calls."""
    model = MotiveDecompositionNetwork(
        input_dim=obs_dim, num_objectives=NUM_OBJECTIVES
    )
    model.eval()
    return RuntimeCertificationPipeline(
        model=model,
        weight_store=WeightSetStore(num_objectives=NUM_OBJECTIVES),
        config=RuntimePipelineConfig(
            gate_type="PDS",
            use_cvar=use_cvar,
            require_cds_or_cvar=True,
            cvar_samples=cvar_samples,
            train_support_after_certify=False,
        ),
    )


def make_governor(candidates: List[CandidateSkillRecord]):
    """Governor with appraisal scales sized to this environment.

    Leaving the scales at 1.0 while deltas run to |10| saturates risk on the
    first step and flattens the coupling into a constant.
    """
    from bridge.governor import MetaMoGovernor

    payoff_scale = float(np.mean([abs(c.delta_r) for c in candidates])) or 1.0
    motive_scale = float(
        np.mean([np.mean(np.abs(c.delta_n)) for c in candidates])
    ) or 1.0
    return MetaMoGovernor(
        initial_goals=MINECRAFT_INITIAL_GOALS,
        payoff_scale=payoff_scale,
        motive_scale=motive_scale,
    )


def threatening_outcome() -> SkillOutcome:
    """A rejected option whose Safety motive collapses.

    Deliberately synthetic rather than a realized rollout: Task 11 asks for a
    *threatening-outcome* scenario, so the appraisal input must reliably push
    the modulators toward caution rather than depending on what the stub
    happened to return.
    """
    delta_n = np.zeros(NUM_OBJECTIVES, dtype=np.float64)
    delta_n[SAFETY] = -4.0
    return SkillOutcome(
        delta_r=-3.0, delta_n=delta_n, admitted=False, effort=0.6
    )


def uniform_signal(epsilon: float, tail: float = 0.1) -> GovernorSignal:
    return GovernorSignal(
        weights=np.full(NUM_OBJECTIVES, 1.0 / NUM_OBJECTIVES),
        pds_epsilon=epsilon,
        cvar_tail_level=tail,
    )


def admitted_ids(records) -> set:
    return {r.skill_id for r in records if getattr(r, "is_certified", False)}


def margin_of(record) -> float:
    return float(record.delta_r) + float(np.min(np.asarray(record.delta_n)))


def run_loop(steps: int = 10, *, seed: int = SEED, use_cvar: bool = True):
    """Drive `steps` threatening steps through the real stack.

    The context is the live env observation, which changes every step. That is
    load-bearing: a constant context would hit the permanence cache and make
    budget changes invisible.
    """
    from bridge.controller import MetaMoController

    env, baseline, candidates, obs = build_world(seed)
    pipeline = make_pipeline(int(obs.shape[0]), use_cvar=use_cvar)
    governor = make_governor(candidates)
    controller = MetaMoController(governor, pipeline, seed=seed)

    for _ in range(steps):
        record = controller.step(
            context=obs,
            candidate_skills=candidates,
            baseline_stats=baseline,
            outcome_for=lambda _rec: threatening_outcome(),
        )
        action = (
            0
            if record.selected_skill_id is None
            else SKILL_NAMES.index(record.selected_skill_id)
        )
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=seed)

    return controller, governor


# ---------------------------------------------------------------------------
# S1 -- budgets reach certification (Scope 4b)
# ---------------------------------------------------------------------------


def test_s1_epsilon_change_flips_a_borderline_admission():
    """A change in epsilon must change what the PDS gate admits.

    RiskyForage exists precisely so this is observable: its margin sits inside
    the (-0.1, 0) band that MetaMo's epsilon range can straddle. Every other
    candidate's margin is orders of magnitude outside it, which is why epsilon
    could not flip anything before that action was added.

    Self-calibrating: epsilon values are derived from the measured margin, so
    this keeps working if the stub's reward table is retuned.

    `use_cvar=False` is required -- under OR semantics an untrained MDN's CVaR
    gate admits nearly everything and would mask the PDS verdict entirely.
    """
    _, baseline, candidates, obs = build_world()

    borderline = next(c for c in candidates if c.skill_id == "RiskyForage")
    margin = margin_of(borderline)

    assert -0.1 < margin < 0.0, (
        f"RiskyForage margin {margin:.6f} is outside the band epsilon can "
        "reach; epsilon cannot flip it and this test proves nothing"
    )

    eps_reject = abs(margin) * 0.5
    eps_admit = abs(margin) * 1.5

    from bridge.controller import MetaMoController

    def certify_at(epsilon: float) -> set:
        # Fresh pipeline per call: certification is cached by
        # (context_key, skill_id), so reusing one would return the first
        # verdict regardless of epsilon.
        pipeline = make_pipeline(int(obs.shape[0]), use_cvar=False)
        controller = MetaMoController(_StubGovernor(), pipeline, seed=SEED)
        records = controller.certify(
            context=obs,
            candidate_skills=candidates,
            baseline_stats=baseline,
            signal=uniform_signal(epsilon),
        )
        return admitted_ids(records)

    assert "RiskyForage" not in certify_at(eps_reject)
    assert "RiskyForage" in certify_at(eps_admit)


class _StubGovernor:
    """Minimal governor for certification-only tests.

    `MetaMoController.certify` never consults the governor -- it takes the
    signal as a parameter -- so this only has to satisfy the constructor.
    """

    num_objectives = NUM_OBJECTIVES

    def signal(self) -> GovernorSignal:
        return uniform_signal(0.1)

    def step(self, outcome: SkillOutcome) -> GovernorSignal:
        return self.signal()


def test_s1b_governor_epsilon_change_alters_admissions():
    """MetaMo's OWN budget change must alter what gets certified.

    S1 proves that *some* epsilon change flips an admission, using hand-picked
    values. S2 proves MetaMo's real epsilon moves. This test joins them: the
    epsilon MetaMo actually emits across a threatening run, fed through the
    real controller, must change the admitted set.

    Why the endpoints are guaranteed rather than tuned:
      * At step 0 the modulators are neutral, so epsilon == epsilon_0 == 0.1
        exactly (pinned by test_bridge_budget.py). RiskyForage's margin of
        ~-0.05 satisfies -0.05 >= -0.1, so it is admitted.
      * Under sustained threat epsilon floors at 0.0 within a few steps (S2
        already asserts it ends no higher than it started). -0.05 >= -0.0 is
        false, so RiskyForage is rejected.
    No intermediate epsilon value is hardcoded, so the assertion survives
    coefficient retuning as long as epsilon still spans the margin.

    `use_cvar=False` is required: under OR semantics the untrained MDN's CVaR
    gate admits everything and the count would sit flat at 6/6 regardless.
    """
    controller, _ = run_loop(steps=10, use_cvar=False)

    eps = [r.pds_epsilon for r in controller.history]
    admitted = [r.admitted_count for r in controller.history]

    assert eps[0] > eps[-1], "epsilon did not tighten; nothing to test"
    assert admitted[0] > admitted[-1], (
        f"epsilon fell {eps[0]:.4f} -> {eps[-1]:.4f} but the admitted count "
        f"stayed {admitted[0]} -> {admitted[-1]}; MetaMo's budget is not "
        "reaching the PDS gate"
    )
    assert len(set(admitted)) > 1, (
        f"admitted count was constant at {admitted[0]} across all "
        f"{len(admitted)} steps despite epsilon spanning "
        f"[{min(eps):.4f}, {max(eps):.4f}]"
    )

    # The count must never rise while epsilon is falling -- a rise would mean
    # something other than the budget is deciding admissions.
    for i in range(1, len(eps)):
        if eps[i] < eps[i - 1]:
            assert admitted[i] <= admitted[i - 1], (
                f"step {i}: epsilon fell {eps[i-1]:.4f} -> {eps[i]:.4f} but "
                f"admitted count rose {admitted[i-1]} -> {admitted[i]}"
            )


# ---------------------------------------------------------------------------
# S2 -- epsilon/alpha coherence (Issue 2, the headline)
# ---------------------------------------------------------------------------


def test_s2_epsilon_and_alpha_never_diverge_under_sustained_threat():
    """The regression guard for the CVaR sign correction.

    The published formula `alpha = alpha_0 + b1*securing + ...` is
    directionally wrong against SubRep's lower-tail CVaR gate: applied
    literally it makes rising `securing` TIGHTEN the PDS budget while
    LOOSENING the CVaR gate. bridge/budget.py flips the b-coefficient signs so
    both tighten together.

    Asserts coherence, not strict monotonicity -- the modulators saturate
    partway through, after which both values hold constant at their floors.
    Holding still is a pass. Moving in opposite directions is the failure.
    """
    controller, _ = run_loop(steps=10)

    eps = [r.pds_epsilon for r in controller.history]
    alpha = [r.cvar_tail_level for r in controller.history]

    assert len(eps) == 10

    for i in range(1, len(eps)):
        d_eps = eps[i] - eps[i - 1]
        d_alpha = alpha[i] - alpha[i - 1]
        assert d_eps * d_alpha >= 0.0, (
            f"step {i}: epsilon moved {d_eps:+.6f} while alpha moved "
            f"{d_alpha:+.6f} -- the budgets diverged, which is exactly the "
            "sign bug bridge/budget.py corrects"
        )

    # And over the episode as a whole, sustained threat must tighten both.
    assert eps[-1] <= eps[0]
    assert alpha[-1] <= alpha[0]
    assert eps[-1] < eps[0] or alpha[-1] < alpha[0], (
        "neither budget moved across 10 threatening steps; the feedback loop "
        "is not reaching the governor"
    )


def test_s2_budgets_stay_inside_their_declared_contracts():
    """Whatever the governor emits must be legal for the gates."""
    controller, _ = run_loop(steps=10)

    for record in controller.history:
        assert record.pds_epsilon >= 0.0            # PDSGate contract
        assert 0.0 < record.cvar_tail_level <= 1.0  # CVaRGate contract


# ---------------------------------------------------------------------------
# S3 -- safety priority adapts (Scope 4a)
# ---------------------------------------------------------------------------


def test_s3_safety_weight_rises_under_sustained_threat():
    """Priorities must actually move, not just the budgets."""
    controller, _ = run_loop(steps=10)

    first = controller.history[0].weights
    last = controller.history[-1].weights

    assert last[SAFETY] > first[SAFETY], (
        f"Safety weight went {first[SAFETY]:.4f} -> {last[SAFETY]:.4f} under "
        "sustained threat; w_meta is not responding to the modulators"
    )
    assert float(np.sum(last)) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# S4 -- abstention (AC3b)
# ---------------------------------------------------------------------------


def test_s4_abstains_when_nothing_qualifies_and_still_feeds_back():
    """With no admissible option the system selects nothing -- and keeps learning.

    The second half matters as much as the first: abstaining must not break
    the feedback loop, or the agent would freeze in whatever motivational
    state caused the abstention.
    """
    from bridge.controller import MetaMoController

    _, baseline, candidates, obs = build_world()

    # Every candidate ruinous: a large negative motive swamps any payoff.
    hostile = [
        CandidateSkillRecord(
            skill_id=f"hostile_{i}",
            delta_r=-5.0,
            delta_n=tuple([-50.0] * NUM_OBJECTIVES),
            is_certified=False,
            gate_type="PDS",
        )
        for i in range(3)
    ]

    pipeline = make_pipeline(int(obs.shape[0]), use_cvar=False)
    governor = make_governor(candidates)
    controller = MetaMoController(governor, pipeline, seed=SEED)

    before = governor.modulator_summary()
    record = controller.step(
        context=obs,
        candidate_skills=hostile,
        baseline_stats=baseline,
        outcome_for=lambda _rec: threatening_outcome(),
    )
    after = governor.modulator_summary()

    assert record.admitted_count == 0
    assert record.selected_skill_id is None
    assert after != before, "governor did not receive feedback while abstaining"


# ---------------------------------------------------------------------------
# S5 -- seed determinism (Scope 5)
# ---------------------------------------------------------------------------


def test_s5_identical_seeds_reproduce_identical_runs():
    """CVaRGate samples from the unseeded global torch RNG
    (certification/cvar_test.py:51), so without the controller's explicit
    seeding these runs would diverge."""
    a, _ = run_loop(steps=10, seed=SEED)
    b, _ = run_loop(steps=10, seed=SEED)

    assert len(a.history) == len(b.history)

    for i, (ra, rb) in enumerate(zip(a.history, b.history)):
        assert ra.pds_epsilon == pytest.approx(rb.pds_epsilon), f"step {i}"
        assert ra.cvar_tail_level == pytest.approx(rb.cvar_tail_level), f"step {i}"
        assert np.allclose(ra.weights, rb.weights), f"step {i}"
        assert ra.selected_skill_id == rb.selected_skill_id, f"step {i}"
        assert ra.admitted_count == rb.admitted_count, f"step {i}"


# ---------------------------------------------------------------------------
# S6 -- priorities reach selection (Scope 4b, the other half)
# ---------------------------------------------------------------------------


def test_s6_different_weights_select_different_skills():
    """S1 proves budgets reach certification; this proves weights reach selection.

    Without this, `w_t` could be silently dropped between governor and
    selector and every other test here would still pass -- the budgets would
    still move, safety weight would still rise, runs would still reproduce.

    The two weight vectors are constructed rather than drawn from two governor
    states, to isolate the variable: softmax outputs from nearby motivational
    states can be too similar to force a ranking change, which would make the
    test flaky for reasons unrelated to the coupling.
    """
    from bridge.controller import MetaMoController

    _, baseline, candidates, obs = build_world()
    pipeline = make_pipeline(int(obs.shape[0]), use_cvar=False)
    controller = MetaMoController(_StubGovernor(), pipeline, seed=SEED)

    records = controller.certify(
        context=obs,
        candidate_skills=candidates,
        baseline_stats=baseline,
        signal=uniform_signal(0.1),
    )
    admitted = [r for r in records if r.is_certified]
    assert len(admitted) >= 2, "need at least two admitted skills to rank"

    def weight_on(index: int) -> GovernorSignal:
        w = np.full(NUM_OBJECTIVES, 0.01)
        w[index] = 1.0 - 0.01 * (NUM_OBJECTIVES - 1)
        return GovernorSignal(weights=w, pds_epsilon=0.1, cvar_tail_level=0.1)

    safety_pick, _ = controller.select(records, weight_on(SAFETY))
    reputation_pick, _ = controller.select(records, weight_on(1))  # Reputation

    assert safety_pick is not None and reputation_pick is not None
    assert safety_pick != reputation_pick, (
        f"both weightings selected {safety_pick!r}; the selection weight is "
        "not influencing which skill is chosen"
    )


def test_s6b_governor_weights_change_skill_scores():
    """MetaMo's OWN weight shift must reach the scoring function.

    S6 proves the selector honours whatever weights it is handed, using
    constructed vectors. S3 proves MetaMo's real weights move. This test joins
    them at the scoring layer: the weights MetaMo actually emits at the start
    and end of a threatening run must produce different scores for the same
    candidates, via the exact function selection uses
    (library/skill_selector.py:21).

    Why scores rather than the selected skill: with this candidate set the
    argmax does NOT flip -- SwingGateBarricade has both the highest delta_r
    and strong Safety, so it wins under early and late weightings alike (the
    demo shows it selected at every step). Asserting on argmax would require
    engineering a candidate specifically to lose under low Safety weight,
    which would test the stub rather than the coupling. Scores are the honest
    observable: if they change, the weights reached selection; S6 covers the
    argmax flipping once the weights differ enough.
    """
    from library.skill_selector import score_skill_entry

    controller, _ = run_loop(steps=10)

    w_early = controller.history[0].weights
    w_late = controller.history[-1].weights

    assert not np.allclose(w_early, w_late), (
        "governor weights did not move across 10 threatening steps"
    )
    assert w_late[SAFETY] > w_early[SAFETY], "Safety weight should have risen"

    _, _, candidates, _ = build_world()

    # Score every candidate under both weightings. The scalarisation is
    # delta_r + w . delta_n, so with delta_r fixed any score change is
    # attributable purely to the weights.
    changed = 0
    for candidate in candidates:
        early = score_skill_entry(candidate, w_early)
        late = score_skill_entry(candidate, w_late)
        if abs(early - late) > 1e-9:
            changed += 1

    assert changed == len(candidates), (
        f"only {changed}/{len(candidates)} candidate scores responded to the "
        "weight shift; the selection weight is not reaching score_skill_entry"
    )

    # And the shift must favour Safety-heavy candidates: the score gap between
    # the safest and the least-safe candidate should widen as Safety weight
    # rises. Rank candidates by their Safety motive delta.
    by_safety = sorted(candidates, key=lambda c: c.delta_n[SAFETY])
    least_safe, most_safe = by_safety[0], by_safety[-1]

    gap_early = (
        score_skill_entry(most_safe, w_early)
        - score_skill_entry(least_safe, w_early)
    )
    gap_late = (
        score_skill_entry(most_safe, w_late)
        - score_skill_entry(least_safe, w_late)
    )
    assert gap_late > gap_early, (
        f"Safety weight rose {w_early[SAFETY]:.3f} -> {w_late[SAFETY]:.3f} "
        f"but the safest/least-safe score gap went {gap_early:+.4f} -> "
        f"{gap_late:+.4f}; the weights are not tilting selection toward Safety"
    )


# ---------------------------------------------------------------------------
# S7 -- selected skills stay admissible (AC3a)
# ---------------------------------------------------------------------------


def test_s7_selection_is_always_drawn_from_the_admitted_set():
    """Holds by construction today -- MetaMoController.select filters on
    is_certified -- which is exactly the kind of invariant that breaks
    unnoticed during a refactor."""
    from bridge.controller import MetaMoController

    _, baseline, candidates, obs = build_world()

    for epsilon in (0.0, 0.02, 0.05, 0.1):
        fresh = make_pipeline(int(obs.shape[0]), use_cvar=False)
        step_controller = MetaMoController(_StubGovernor(), fresh, seed=SEED)
        signal = uniform_signal(epsilon)

        records = step_controller.certify(
            context=obs,
            candidate_skills=candidates,
            baseline_stats=baseline,
            signal=signal,
        )
        selected, _ = step_controller.select(records, signal)
        admitted = admitted_ids(records)

        if selected is not None:
            assert selected in admitted, (
                f"epsilon={epsilon}: selected {selected!r} which was not "
                f"admitted (admitted={sorted(admitted)})"
            )

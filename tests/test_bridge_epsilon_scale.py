"""Tests for bridge/epsilon_scale.py and its hook in the controller.

No MetaMo and no LunarLander environment: synthetic records go through a real
`RuntimeCertificationPipeline` (untrained MDN, PDS only), driven by
`FakeGovernor` where a governor is needed. The six-objective test uses the
Minecraft stub, which needs no MetaMo either.

The synthetic records mimic MO-LunarLander against a crashing noop baseline:
baseline payoff -100, so a crash has delta_r 0 (own payoff -100) and a landing
delta_r +200 (own payoff +100).
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from bridge.controller import MetaMoController
from bridge.epsilon_scale import TASK_FAILURE, EpsilonScale
from bridge.governor import FakeGovernor
from bridge.protocol import GovernorSignal, SkillOutcome
from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_runtime_pipeline import RuntimeCertificationPipeline, RuntimePipelineConfig
from utils.weight_set_store import WeightSetStore

M = 2
OBS_DIM = 4
SEED = 7
UNIT = 300.0
BASELINE = {"baseline_payoff": -100.0, "baseline_motives": np.zeros(M)}
CONTEXT = np.full(OBS_DIM, 0.25)


def make_pipeline(num_objectives: int = M, obs_dim: int = OBS_DIM) -> RuntimeCertificationPipeline:
    """A fresh pipeline: the PDS gate only, full simplex (W_x stays empty)."""
    import torch

    torch.manual_seed(SEED)
    model = MotiveDecompositionNetwork(input_dim=obs_dim, num_objectives=num_objectives)
    model.eval()
    return RuntimeCertificationPipeline(
        model=model,
        weight_store=WeightSetStore(num_objectives=num_objectives),
        config=RuntimePipelineConfig(
            gate_type="PDS",
            use_cvar=False,
            train_support_after_certify=False,
        ),
    )


def signal(epsilon: float, num_objectives: int = M) -> GovernorSignal:
    return GovernorSignal(
        weights=np.full(num_objectives, 1.0 / num_objectives),
        pds_epsilon=epsilon,
        cvar_tail_level=0.1,
    )


def record(skill_id: str, delta_r: float, delta_n) -> CandidateSkillRecord:
    return CandidateSkillRecord(
        skill_id=skill_id,
        delta_r=float(delta_r),
        delta_n=tuple(float(v) for v in delta_n),
        is_certified=False,
        gate_type="PDS",
        baseline_id="noop",
    )


def crash(skill_id: str = "crash", margin: float = -1.0) -> CandidateSkillRecord:
    """Crashes like the baseline (delta_r 0), with the given PDS margin."""
    return record(skill_id, 0.0, (margin, 5.0))


def landing(skill_id: str, margin: float) -> CandidateSkillRecord:
    """Lands (delta_r +200) but loses fuel, so delta_r + min(delta_n) == margin."""
    return record(skill_id, 200.0, (150.0, margin - 200.0))


def controller(scale=None, pipeline=None) -> MetaMoController:
    return MetaMoController(
        FakeGovernor([signal(0.1)]),
        make_pipeline() if pipeline is None else pipeline,
        seed=SEED,
        epsilon_scale=scale,
    )


def admitted(records) -> set:
    return {r.skill_id for r in records if r.is_certified}


GUARDED = EpsilonScale(unit=UNIT, payoff_floor=0.0)


# ---------------------------------------------------------------------------
# EpsilonScale on its own
# ---------------------------------------------------------------------------


def test_tolerance_is_epsilon_times_unit():
    scale = EpsilonScale(unit=UNIT)
    assert scale.tolerance(0.1) == pytest.approx(30.0)
    assert scale.tolerance(0.0) == 0.0


@pytest.mark.parametrize("unit", [0.0, -1.0, math.nan, math.inf])
def test_invalid_units_are_rejected(unit):
    with pytest.raises(ValueError):
        EpsilonScale(unit=unit)


def test_a_non_finite_floor_is_rejected():
    with pytest.raises(ValueError):
        EpsilonScale(unit=UNIT, payoff_floor=math.nan)


def test_the_guard_needs_a_floor_and_a_baseline_payoff():
    failing = crash()
    assert GUARDED.fails_task(failing, BASELINE)
    assert not EpsilonScale(unit=UNIT).fails_task(failing, BASELINE)
    assert not GUARDED.fails_task(failing, {"baseline_motives": np.zeros(M)})


# ---------------------------------------------------------------------------
# The controller hook
# ---------------------------------------------------------------------------


def test_without_a_scale_the_controller_is_unchanged():
    """No scale: the same verdicts, epsilons and selection as stamping eps directly."""
    candidates = [landing("near", -0.05), landing("far", -20.0), crash(margin=-0.02)]
    sig = signal(0.1)

    records = controller().certify(
        context=CONTEXT, candidate_skills=candidates, baseline_stats=BASELINE, signal=sig
    )
    stamped = [replace(c, epsilon=0.1) for c in candidates]
    direct = make_pipeline().certify_candidate_skills(
        context=CONTEXT,
        candidate_skills=stamped,
        baseline_stats=BASELINE,
        weights_used=sig.weights,
        cvar_confidence=sig.cvar_tail_level,
    )

    assert [r.skill_id for r in records] == [r.skill_id for r in direct]
    assert [r.is_certified for r in records] == [r.is_certified for r in direct]
    assert [r.epsilon for r in records] == [r.epsilon for r in direct]
    assert controller().select(records, sig) == controller().select(direct, sig)
    # Raw eps 0.1 admits the -0.05 landing and the -0.02 crash, not the -20 landing.
    assert admitted(records) == {"near", "crash"}


def test_task_failures_are_rejected_at_every_epsilon():
    pipeline = make_pipeline()
    ctrl = controller(GUARDED, pipeline)
    for epsilon in np.linspace(0.0, 0.5, 51):
        (result,) = ctrl.certify(
            context=CONTEXT,
            candidate_skills=[crash(margin=-1.0)],
            baseline_stats=BASELINE,
            signal=signal(float(epsilon)),
        )
        assert not result.is_certified, epsilon
        assert result.metadata["rejection_reason"] == TASK_FAILURE


def test_task_failures_never_enter_the_certificate_cache():
    pipeline = make_pipeline()
    controller(GUARDED, pipeline).certify(
        context=CONTEXT,
        candidate_skills=[crash(margin=-1.0)],
        baseline_stats=BASELINE,
        signal=signal(0.5),
    )
    assert pipeline.get_certification_result(context=CONTEXT, skill_id="crash") is None


def test_a_tighter_epsilon_rejects_a_trade_off_landing():
    """Margin -20 with U = 300: eps 0.1 tolerates 30 and admits; eps 0.05 tolerates 15 and rejects."""
    trade_off = landing("trade_off", -20.0)

    def admitted_at(epsilon: float) -> bool:
        (result,) = controller(GUARDED).certify(
            context=CONTEXT,
            candidate_skills=[trade_off],
            baseline_stats=BASELINE,
            signal=signal(epsilon),
        )
        return result.is_certified

    assert admitted_at(0.1)
    assert not admitted_at(0.05)


def test_admission_is_monotonic_in_epsilon():
    margins = np.linspace(-200.0, 50.0, 26)
    candidates = [landing(f"m{i}", m) for i, m in enumerate(margins)]

    previous: set = set()
    for epsilon in np.linspace(0.0, 0.5, 11):
        current = admitted(
            controller(GUARDED).certify(
                context=CONTEXT,
                candidate_skills=candidates,
                baseline_stats=BASELINE,
                signal=signal(float(epsilon)),
            )
        )
        assert previous <= current, epsilon
        previous = current
    assert previous != {c.skill_id for c in candidates}  # the deepest margins stay out


def test_a_cached_admission_is_rejected_after_epsilon_tightens():
    """Same pipeline and context: admitted at eps 0.1, then rejected and not selected at eps 0."""
    trade_off = landing("trade_off", -20.0)
    pipeline = make_pipeline()
    ctrl = controller(GUARDED, pipeline)

    def certify_and_select(epsilon: float):
        sig = signal(epsilon)
        records = ctrl.certify(
            context=CONTEXT, candidate_skills=[trade_off], baseline_stats=BASELINE, signal=sig
        )
        selected, _ = ctrl.select(records, sig)
        return records[0], selected

    loose, selected_loose = certify_and_select(0.1)
    assert loose.is_certified and selected_loose == "trade_off"

    tight, selected_tight = certify_and_select(0.0)
    assert not tight.is_certified, "a cached admission survived the tighter tolerance"
    assert selected_tight is None

    historical = pipeline.get_certification_result(context=CONTEXT, skill_id="trade_off")
    assert historical.is_certified
    assert historical.epsilon == pytest.approx(GUARDED.tolerance(0.1))


def test_records_keep_their_input_order():
    candidates = [crash("c1"), landing("l1", 10.0), crash("c2"), landing("l2", -500.0)]
    records = controller(GUARDED).certify(
        context=CONTEXT, candidate_skills=candidates, baseline_stats=BASELINE, signal=signal(0.1)
    )
    assert [r.skill_id for r in records] == ["c1", "l1", "c2", "l2"]
    assert [r.is_certified for r in records] == [False, True, False, False]
    reasons = [r.metadata.get("rejection_reason") for r in records]
    assert reasons == [TASK_FAILURE, None, TASK_FAILURE, None]


def test_all_failing_candidates_give_an_empty_admission():
    ctrl = controller(GUARDED)
    records = ctrl.certify(
        context=CONTEXT,
        candidate_skills=[crash("c1"), crash("c2", margin=-0.5)],
        baseline_stats=BASELINE,
        signal=signal(0.5),
    )
    assert len(records) == 2
    assert admitted(records) == set()
    assert ctrl.select(records, signal(0.5)) == (None, None)


def test_step_records_the_tolerance_and_the_task_failures():
    ctrl = controller(GUARDED)
    step = ctrl.step(
        context=CONTEXT,
        candidate_skills=[crash("c1"), landing("l1", -20.0), crash("c2")],
        baseline_stats=BASELINE,
        outcome_for=lambda _rec: SkillOutcome(delta_r=0.0, delta_n=np.zeros(M), admitted=False),
    )
    assert step.pds_tolerance == pytest.approx(30.0)
    assert step.task_failures == 2
    assert step.admitted_count == 1 and step.selected_skill_id == "l1"


def test_step_without_a_scale_leaves_the_new_fields_unset():
    step = controller().step(
        context=CONTEXT,
        candidate_skills=[landing("l1", -20.0)],
        baseline_stats=BASELINE,
        outcome_for=lambda _rec: SkillOutcome(delta_r=0.0, delta_n=np.zeros(M), admitted=False),
    )
    assert step.pds_tolerance is None
    assert step.task_failures == 0


# ---------------------------------------------------------------------------
# Dimension-generic
# ---------------------------------------------------------------------------


def test_the_scale_works_for_six_objectives():
    """The Minecraft stub (m = 6): a unit-1 scale without a floor matches the raw path."""
    from env.minecraft_rollout import DEFAULT_HORIZON, evaluate_candidates
    from env.minecraft_stub import MinecraftStubEnv

    env = MinecraftStubEnv(seed=SEED, noise_scale=0.0)
    env.reset(seed=SEED)
    baseline, candidates = evaluate_candidates(env, gamma=0.99, horizon=DEFAULT_HORIZON)
    obs, _ = env.reset(seed=SEED)
    m = len(candidates[0].delta_n)
    assert m == 6

    def run(scale):
        ctrl = MetaMoController(
            FakeGovernor([signal(0.05, m)]),
            make_pipeline(num_objectives=m, obs_dim=int(obs.shape[0])),
            seed=SEED,
            epsilon_scale=scale,
        )
        return ctrl.certify(
            context=obs, candidate_skills=candidates, baseline_stats=baseline, signal=signal(0.05, m)
        )

    raw = run(None)
    unit_one = run(EpsilonScale(unit=1.0))
    assert [r.is_certified for r in unit_one] == [r.is_certified for r in raw]
    assert [r.epsilon for r in unit_one] == [r.epsilon for r in raw]

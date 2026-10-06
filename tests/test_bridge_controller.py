"""Controller tests -- no MetaMo required, driven by FakeGovernor.

These verify the part that is easiest to get silently wrong: that the
per-step budgets actually REACH the gates, and that certification is
reproducible despite CVaRGate sampling from the unseeded global torch RNG
(certification/cvar_test.py:51).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from bridge.controller import MetaMoController, StepRecord
from bridge.governor import FakeGovernor
from bridge.protocol import GovernorSignal, SkillOutcome

M = 6


def signal(epsilon: float = 0.1, tail: float = 0.1) -> GovernorSignal:
    return GovernorSignal(
        weights=np.full(M, 1.0 / M),
        pds_epsilon=epsilon,
        cvar_tail_level=tail,
    )


class RecordingPipeline:
    """Captures what the controller passes down to certification."""

    def __init__(self, certify_ids: Optional[List[str]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.certify_ids = certify_ids

    def certify_candidate_skills(
        self,
        *,
        context,
        candidate_skills,
        baseline_stats,
        weights_used=None,
        cvar_confidence=None,
    ):
        self.calls.append(
            {
                "context": np.asarray(context).copy(),
                "weights_used": None if weights_used is None
                else np.asarray(weights_used).copy(),
                "cvar_confidence": cvar_confidence,
                "n_candidates": len(candidate_skills),
            }
        )
        out = []
        for record in candidate_skills:
            certified = (
                True if self.certify_ids is None
                else record.skill_id in self.certify_ids
            )
            out.append(_Rec(record.skill_id, record.delta_r, record.delta_n, certified))
        return out


class _Rec:
    def __init__(self, skill_id, delta_r, delta_n, is_certified):
        self.skill_id = skill_id
        self.delta_r = float(delta_r)
        self.delta_n = tuple(float(v) for v in delta_n)
        self.is_certified = bool(is_certified)


def make_candidates() -> List[_Rec]:
    return [
        _Rec("alpha_skill", 1.0, np.full(M, 0.1), False),
        _Rec("beta_skill", 2.0, np.full(M, -0.05), False),
        _Rec("gamma_skill", 0.5, np.full(M, 0.3), False),
    ]


def outcome_for(record):
    if record is None:
        return SkillOutcome(
            delta_r=-0.2, delta_n=np.full(M, -0.1), admitted=False
        )
    return SkillOutcome(
        delta_r=record.delta_r,
        delta_n=np.asarray(record.delta_n),
        admitted=True,
    )


# --------------------------------------------------------------------------
# The budgets must actually reach the gates.
# --------------------------------------------------------------------------


def test_cvar_tail_level_is_forwarded_to_the_pipeline():
    pipeline = RecordingPipeline()
    governor = FakeGovernor([signal(epsilon=0.07, tail=0.03)])
    controller = MetaMoController(governor, pipeline, seed=1)

    controller.step(
        context=np.zeros(4),
        candidate_skills=make_candidates(),
        baseline_stats={},
        outcome_for=outcome_for,
    )

    assert pipeline.calls[0]["cvar_confidence"] == pytest.approx(0.03)


def test_weights_are_forwarded_to_the_pipeline():
    pipeline = RecordingPipeline()
    governor = FakeGovernor([signal()])
    controller = MetaMoController(governor, pipeline, seed=1)

    controller.step(
        context=np.zeros(4),
        candidate_skills=make_candidates(),
        baseline_stats={},
        outcome_for=outcome_for,
    )

    assert np.allclose(pipeline.calls[0]["weights_used"], np.full(M, 1.0 / M))


def test_changing_signals_change_what_reaches_the_gates():
    pipeline = RecordingPipeline()
    governor = FakeGovernor(
        [signal(0.10, 0.10), signal(0.05, 0.06), signal(0.01, 0.02)]
    )
    controller = MetaMoController(governor, pipeline, seed=1)

    for _ in range(3):
        controller.step(
            context=np.zeros(4),
            candidate_skills=make_candidates(),
            baseline_stats={},
            outcome_for=outcome_for,
        )

    forwarded = [call["cvar_confidence"] for call in pipeline.calls]
    assert forwarded == pytest.approx([0.10, 0.06, 0.02])
    assert forwarded == sorted(forwarded, reverse=True), "budgets should tighten"


# --------------------------------------------------------------------------
# Selection and feedback.
# --------------------------------------------------------------------------


def test_selects_highest_scoring_admitted_skill():
    pipeline = RecordingPipeline()
    governor = FakeGovernor([signal()])
    controller = MetaMoController(governor, pipeline, seed=1)

    record = controller.step(
        context=np.zeros(4),
        candidate_skills=make_candidates(),
        baseline_stats={},
        outcome_for=outcome_for,
    )

    # Uniform weights: score = delta_r + mean(delta_n).
    # alpha 1.0+0.1=1.1 | beta 2.0-0.05=1.95 | gamma 0.5+0.3=0.8
    assert record.selected_skill_id == "beta_skill"
    assert record.admitted_count == 3


def test_no_admitted_skill_yields_none_and_still_feeds_back():
    pipeline = RecordingPipeline(certify_ids=[])
    governor = FakeGovernor([signal()])
    controller = MetaMoController(governor, pipeline, seed=1)

    record = controller.step(
        context=np.zeros(4),
        candidate_skills=make_candidates(),
        baseline_stats={},
        outcome_for=outcome_for,
    )

    assert record.selected_skill_id is None
    assert record.admitted_count == 0
    assert len(governor.observed) == 1
    assert governor.observed[0].admitted is False


def test_outcome_is_fed_back_to_the_governor():
    pipeline = RecordingPipeline()
    governor = FakeGovernor([signal()])
    controller = MetaMoController(governor, pipeline, seed=1)

    controller.step(
        context=np.zeros(4),
        candidate_skills=make_candidates(),
        baseline_stats={},
        outcome_for=outcome_for,
    )

    assert len(governor.observed) == 1
    assert governor.observed[0].admitted is True


def test_history_accumulates_records():
    pipeline = RecordingPipeline()
    governor = FakeGovernor([signal()])
    controller = MetaMoController(governor, pipeline, seed=1)

    for _ in range(3):
        controller.step(
            context=np.zeros(4),
            candidate_skills=make_candidates(),
            baseline_stats={},
            outcome_for=outcome_for,
        )

    assert len(controller.history) == 3
    assert [r.step for r in controller.history] == [0, 1, 2]
    assert all(isinstance(r, StepRecord) for r in controller.history)


# --------------------------------------------------------------------------
# Determinism.
# --------------------------------------------------------------------------


def test_seed_is_recorded_on_every_step():
    pipeline = RecordingPipeline()
    governor = FakeGovernor([signal()])
    controller = MetaMoController(governor, pipeline, seed=123)

    record = controller.step(
        context=np.zeros(4),
        candidate_skills=make_candidates(),
        baseline_stats={},
        outcome_for=outcome_for,
    )
    assert record.seed == 123


def test_seeding_makes_cvar_sampling_reproducible():
    """CVaRGate draws from the global torch RNG without seeding itself.

    The controller seeds before certification, so two identically seeded
    passes must produce identical CVaR values.
    """
    torch = pytest.importorskip("torch")
    from certification.cvar_test import CVaRGate

    gate = CVaRGate(confidence=0.1, n_samples=500)
    delta_r = 0.05
    delta_n = np.array([0.2, -0.3, 0.1, 0.0, -0.1, 0.25])
    concentration = np.array([2.0, 1.5, 1.0, 1.0, 0.8, 1.2])

    torch.manual_seed(7)
    first = gate.get_cvar(delta_r, delta_n, mdn_alpha=concentration)
    torch.manual_seed(7)
    second = gate.get_cvar(delta_r, delta_n, mdn_alpha=concentration)

    assert first == pytest.approx(second)


# --------------------------------------------------------------------------
# FakeGovernor itself.
# --------------------------------------------------------------------------


def test_fake_governor_repeats_final_signal_by_default():
    governor = FakeGovernor([signal(0.1, 0.1), signal(0.05, 0.05)])
    governor.step(outcome_for(None))
    governor.step(outcome_for(None))
    governor.step(outcome_for(None))
    assert governor.signal().pds_epsilon == pytest.approx(0.05)


def test_fake_governor_can_refuse_to_repeat():
    governor = FakeGovernor([signal()], repeat_last=False)
    with pytest.raises(IndexError):
        governor.step(outcome_for(None))


def test_fake_governor_requires_at_least_one_signal():
    with pytest.raises(ValueError):
        FakeGovernor([])


# --------------------------------------------------------------------------
# Stimulus scaling -- pure, no MetaMo, so it runs even when the adapter
# tests skip.
# --------------------------------------------------------------------------


def test_motive_scale_controls_risk_sensitivity():
    from bridge.stimulus import build_stimulus_values

    outcome = SkillOutcome(
        delta_r=-1.5, delta_n=np.full(M, -2.0), admitted=True, effort=0.3
    )
    weights = np.full(M, 1.0 / M)

    unscaled = build_stimulus_values(outcome, weights=weights, motive_scale=1.0)
    scaled = build_stimulus_values(outcome, weights=weights, motive_scale=8.0)

    assert unscaled.risk > scaled.risk


def test_risk_saturates_for_large_deltas():
    """Why the governor test must use moderate inputs.

    tanh + clip means a large enough delta pins risk at 1.0 for any scale,
    erasing the difference the scaling is supposed to create.
    """
    from bridge.stimulus import build_stimulus_values

    extreme = SkillOutcome(
        delta_r=-8.0, delta_n=np.full(M, -6.0), admitted=False, effort=0.5
    )
    weights = np.full(M, 1.0 / M)

    a = build_stimulus_values(extreme, weights=weights, motive_scale=1.0)
    b = build_stimulus_values(extreme, weights=weights, motive_scale=6.0)

    assert a.risk == pytest.approx(1.0)
    assert b.risk == pytest.approx(1.0)


def test_rejection_adds_risk():
    from bridge.stimulus import build_stimulus_values

    delta_n = np.full(M, -0.2)
    weights = np.full(M, 1.0 / M)

    admitted = build_stimulus_values(
        SkillOutcome(delta_r=0.1, delta_n=delta_n, admitted=True), weights=weights
    )
    rejected = build_stimulus_values(
        SkillOutcome(delta_r=0.1, delta_n=delta_n, admitted=False), weights=weights
    )

    assert rejected.risk > admitted.risk


def test_stimulus_values_stay_in_unit_range():
    from bridge.stimulus import build_stimulus_values

    weights = np.full(M, 1.0 / M)
    rng = np.random.default_rng(0)
    for _ in range(50):
        outcome = SkillOutcome(
            delta_r=float(rng.normal(0, 5)),
            delta_n=rng.normal(0, 5, size=M),
            admitted=bool(rng.integers(0, 2)),
            effort=float(rng.uniform(0, 1)),
        )
        values = build_stimulus_values(outcome, weights=weights)
        for field in ("novelty", "conduciveness", "risk", "effort"):
            assert 0.0 <= getattr(values, field) <= 1.0

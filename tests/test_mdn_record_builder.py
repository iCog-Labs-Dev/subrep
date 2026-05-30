from __future__ import annotations

import numpy as np

from utils.mdn_record_builder import (
    PreparedCandidateOutcome,
    build_candidate_skill_record,
    build_candidate_skill_records,
    build_decision_record_from_outcome,
    group_candidate_outcomes_by_context,
)


def _baseline_stats() -> dict[str, object]:
    return {
        "baseline_payoff": 1.0,
        "baseline_motives": np.array([0.5, 0.2], dtype=np.float32),
    }


def test_build_candidate_skill_record_populates_delta_and_certification_fields():
    record = build_candidate_skill_record(
        skill_id="skill_a",
        skill_payoff=1.7,
        skill_motives=np.array([0.8, 0.4], dtype=np.float32),
        baseline_stats=_baseline_stats(),
        gate_type="CDS",
        metadata={"source": "test"},
        baseline_id="baseline_v1",
    )

    assert record.skill_id == "skill_a"
    assert record.gate_type == "CDS"
    assert np.isclose(record.delta_r, 0.7)
    assert record.baseline_id == "baseline_v1"
    assert isinstance(record.is_certified, bool)


def test_build_candidate_skill_record_supports_pds_epsilon():
    record = build_candidate_skill_record(
        skill_id="skill_a",
        skill_payoff=1.2,
        skill_motives=np.array([0.5, 0.0], dtype=np.float32),
        baseline_stats=_baseline_stats(),
        gate_type="PDS",
        epsilon=0.2,
    )

    assert record.gate_type == "PDS"
    assert np.isclose(record.epsilon, 0.2)


def test_prepared_candidate_outcome_normalizes_context_and_preserves_metadata():
    outcome = PreparedCandidateOutcome(
        context=np.array([0.1] * 14, dtype=np.float32),
        skill_id="skill_a",
        payoff=1.7,
        motives=(0.8, 0.4),
        metadata={"source": "prepared"},
        gate_type="cds",
    )

    assert len(outcome.context) == 14
    assert outcome.gate_type == "CDS"
    assert outcome.metadata["source"] == "prepared"


def test_build_candidate_skill_records_rejects_missing_outcome_fields():
    try:
        build_candidate_skill_records(
            skill_outcomes=({"skill_id": "skill_a", "payoff": 1.0},),
            baseline_stats=_baseline_stats(),
        )
    except ValueError as exc:
        assert "skill_id" in str(exc)
        assert "motives" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing skill outcome fields")


def test_build_candidate_skill_records_accepts_prepared_outcomes():
    outcomes = (
        PreparedCandidateOutcome(
            context=(0.1,) * 14,
            skill_id="skill_a",
            payoff=1.7,
            motives=(0.8, 0.4),
        ),
        PreparedCandidateOutcome(
            context=(0.1,) * 14,
            skill_id="skill_b",
            payoff=1.1,
            motives=(0.3, 0.7),
        ),
    )

    records = build_candidate_skill_records(skill_outcomes=outcomes, baseline_stats=_baseline_stats())

    assert len(records) == 2
    assert records[0].skill_id == "skill_a"


def test_group_candidate_outcomes_by_context_groups_records_correctly():
    context_a = PreparedCandidateOutcome(context=(0.1,) * 14, skill_id="skill_a", payoff=1.7, motives=(0.8, 0.4)).context
    context_b = PreparedCandidateOutcome(context=(0.2,) * 14, skill_id="skill_c", payoff=1.4, motives=(0.5, 0.6)).context
    outcomes = (
        PreparedCandidateOutcome(context=context_a, skill_id="skill_a", payoff=1.7, motives=(0.8, 0.4)),
        PreparedCandidateOutcome(context=context_a, skill_id="skill_b", payoff=1.1, motives=(0.3, 0.7)),
        PreparedCandidateOutcome(context=context_b, skill_id="skill_c", payoff=1.4, motives=(0.5, 0.6)),
    )

    grouped = group_candidate_outcomes_by_context(outcomes)

    assert len(grouped) == 2
    assert len(grouped[context_a]) == 2
    assert len(grouped[context_b]) == 1


def test_build_decision_record_from_outcome_produces_valid_record():
    candidate_skills = build_candidate_skill_records(
        skill_outcomes=(
            PreparedCandidateOutcome(
                context=(0.1,) * 14,
                skill_id="skill_a",
                payoff=1.7,
                motives=(0.8, 0.4),
            ),
            PreparedCandidateOutcome(
                context=(0.1,) * 14,
                skill_id="skill_b",
                payoff=1.1,
                motives=(0.3, 0.7),
            ),
        ),
        baseline_stats=_baseline_stats(),
    )
    record = build_decision_record_from_outcome(
        context=(0.1,) * 14,
        alpha=(2.0, 3.0),
        support_values=(0.7, 0.3),
        weights_used=(0.4, 0.6),
        candidate_skills=candidate_skills,
        selected_skill_id="skill_a",
        selected_score=0.55,
        actual_payoff=1.7,
        actual_motives=(0.8, 0.4),
        utility=0.56,
    )

    assert record.selected_skill_id == "skill_a"
    assert np.isclose(record.utility, 0.56)

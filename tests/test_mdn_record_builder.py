from __future__ import annotations

import numpy as np

from utils.mdn_record_builder import (
    build_candidate_skill_record,
    build_candidate_skill_records,
    build_decision_record_from_outcome,
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


def test_build_decision_record_from_outcome_produces_valid_record():
    candidate_skills = build_candidate_skill_records(
        skill_outcomes=(
            {"skill_id": "skill_a", "payoff": 1.7, "motives": np.array([0.8, 0.4], dtype=np.float32)},
            {"skill_id": "skill_b", "payoff": 1.1, "motives": np.array([0.3, 0.7], dtype=np.float32)},
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

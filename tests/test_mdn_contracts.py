from __future__ import annotations

import numpy as np
import pytest

from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord, validate_decision_record
from utils.mdn_selection import alpha_to_mean_weights


def test_candidate_skill_record_accepts_valid_values():
    record = CandidateSkillRecord(
        skill_id="skill_a",
        delta_r=0.5,
        delta_n=(0.2, -0.1),
        is_certified=True,
        gate_type="cds",
        metadata={"source": "test"},
        admission_margin=0.3,
        epsilon=0.0,
        baseline_id="baseline_v1",
    )

    assert record.gate_type == "CDS"
    assert record.delta_n == (0.2, -0.1)


def test_candidate_skill_record_rejects_invalid_gate_type():
    with pytest.raises(ValueError, match="gate_type"):
        CandidateSkillRecord(
            skill_id="skill_a",
            delta_r=0.5,
            delta_n=(0.2, -0.1),
            is_certified=True,
            gate_type="INVALID",
        )


def test_candidate_skill_record_rejects_invalid_delta_n_length():
    with pytest.raises(ValueError, match="delta_n"):
        CandidateSkillRecord(
            skill_id="skill_a",
            delta_r=0.5,
            delta_n=(0.2, -0.1, 0.4),
            is_certified=True,
            gate_type="CDS",
        )


def test_alpha_to_mean_weights_normalizes_single_vector():
    alpha = np.array([2.0, 3.0], dtype=np.float32)
    weights = alpha_to_mean_weights(alpha)

    assert weights.shape == (2,)
    assert np.allclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)


def test_alpha_to_mean_weights_normalizes_batches():
    alpha = np.array([[1.0, 1.0], [2.0, 6.0]], dtype=np.float32)
    weights = alpha_to_mean_weights(alpha)

    assert weights.shape == (2, 2)
    assert np.allclose(weights.sum(axis=1), np.ones(2))


def test_alpha_to_mean_weights_rejects_non_positive_values():
    with pytest.raises(ValueError, match="strictly positive"):
        alpha_to_mean_weights(np.array([1.0, 0.0], dtype=np.float32))


def test_mdn_decision_record_accepts_valid_values():
    candidates = (
        CandidateSkillRecord(
            skill_id="skill_a",
            delta_r=0.5,
            delta_n=(0.2, -0.1),
            is_certified=True,
            gate_type="CDS",
        ),
        CandidateSkillRecord(
            skill_id="skill_b",
            delta_r=0.1,
            delta_n=(0.0, 0.3),
            is_certified=False,
            gate_type="PDS",
            epsilon=0.1,
        ),
    )
    record = MDNDecisionRecord(
        context=(0.1,) * 14,
        alpha=(2.0, 3.0),
        support_values=(0.7, 0.3),
        weights_used=(0.4, 0.6),
        candidate_skills=candidates,
        selected_skill_id="skill_a",
        selected_score=0.55,
        behavior_probability=0.75,
        actual_payoff=1.2,
        actual_motives=(0.8, 0.1),
        utility=0.9,
    )

    validate_decision_record(record)


def test_mdn_decision_record_rejects_invalid_behavior_probability():
    candidates = (
        CandidateSkillRecord(
            skill_id="skill_a",
            delta_r=0.5,
            delta_n=(0.2, -0.1),
            is_certified=True,
            gate_type="CDS",
        ),
    )
    with pytest.raises(ValueError, match="behavior_probability"):
        MDNDecisionRecord(
            context=(0.1,) * 8,
            alpha=(2.0, 3.0),
            support_values=(0.7, 0.3),
            weights_used=(0.4, 0.6),
            candidate_skills=candidates,
            selected_skill_id="skill_a",
            behavior_probability=0.0,
        )


def test_mdn_decision_record_rejects_invalid_weights():
    candidates = (
        CandidateSkillRecord(
            skill_id="skill_a",
            delta_r=0.5,
            delta_n=(0.2, -0.1),
            is_certified=True,
            gate_type="CDS",
        ),
    )
    with pytest.raises(ValueError, match="weights_used"):
        MDNDecisionRecord(
            context=(0.1,) * 14,
            alpha=(2.0, 3.0),
            support_values=(0.7, 0.3),
            weights_used=(0.4, 0.7),
            candidate_skills=candidates,
            selected_skill_id="skill_a",
        )


def test_mdn_decision_record_rejects_selected_skill_not_in_candidates():
    candidates = (
        CandidateSkillRecord(
            skill_id="skill_a",
            delta_r=0.5,
            delta_n=(0.2, -0.1),
            is_certified=True,
            gate_type="CDS",
        ),
    )
    with pytest.raises(ValueError, match="selected_skill_id"):
        MDNDecisionRecord(
            context=(0.1,) * 14,
            alpha=(2.0, 3.0),
            support_values=(0.7, 0.3),
            weights_used=(0.4, 0.6),
            candidate_skills=candidates,
            selected_skill_id="skill_b",
        )


def test_mdn_decision_record_rejects_negative_support_values():
    candidates = (
        CandidateSkillRecord(
            skill_id="skill_a",
            delta_r=0.5,
            delta_n=(0.2, -0.1),
            is_certified=True,
            gate_type="CDS",
        ),
    )
    with pytest.raises(ValueError, match="support_values"):
        MDNDecisionRecord(
            context=(0.1,) * 8,
            alpha=(2.0, 3.0),
            support_values=(-0.1, 0.3),
            weights_used=(0.4, 0.6),
            candidate_skills=candidates,
            selected_skill_id="skill_a",
        )

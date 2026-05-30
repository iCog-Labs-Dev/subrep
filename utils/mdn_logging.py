"""Helpers for building and serializing MDN decision records."""

from __future__ import annotations

from typing import Any

from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord, validate_decision_record


def build_decision_record(
    *,
    context,
    alpha,
    support_values,
    weights_used,
    candidate_skills,
    selected_skill_id: str,
    selected_score: float | None = None,
    actual_payoff: float | None = None,
    actual_motives=None,
    utility: float | None = None,
) -> MDNDecisionRecord:
    """Construct and validate an MDNDecisionRecord from runtime values."""
    record = MDNDecisionRecord(
        context=context,
        alpha=alpha,
        support_values=support_values,
        weights_used=weights_used,
        candidate_skills=tuple(candidate_skills),
        selected_skill_id=selected_skill_id,
        selected_score=selected_score,
        actual_payoff=actual_payoff,
        actual_motives=actual_motives,
        utility=utility,
    )
    validate_decision_record(record)
    return record


def serialize_decision_record(record: MDNDecisionRecord) -> dict[str, Any]:
    """Convert an MDN decision record into a serialization-ready dictionary."""
    validate_decision_record(record)
    return {
        "context": list(record.context),
        "alpha": list(record.alpha),
        "support_values": list(record.support_values),
        "weights_used": list(record.weights_used),
        "candidate_skills": [serialize_candidate_skill(candidate) for candidate in record.candidate_skills],
        "selected_skill_id": record.selected_skill_id,
        "selected_score": record.selected_score,
        "actual_payoff": record.actual_payoff,
        "actual_motives": None if record.actual_motives is None else list(record.actual_motives),
        "utility": record.utility,
    }


def serialize_candidate_skill(candidate: CandidateSkillRecord) -> dict[str, Any]:
    """Convert a candidate skill record into a serialization-ready dictionary."""
    return {
        "skill_id": candidate.skill_id,
        "delta_r": candidate.delta_r,
        "delta_n": list(candidate.delta_n),
        "is_certified": candidate.is_certified,
        "gate_type": candidate.gate_type,
        "metadata": dict(candidate.metadata),
        "admission_margin": candidate.admission_margin,
        "epsilon": candidate.epsilon,
        "baseline_id": candidate.baseline_id,
    }

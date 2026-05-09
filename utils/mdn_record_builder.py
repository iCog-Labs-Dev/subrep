"""Builders that connect baseline/certification outputs to MDN records."""

from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord
from utils.mdn_logging import build_decision_record


def build_candidate_skill_record(
    *,
    skill_id: str,
    skill_payoff: float,
    skill_motives,
    baseline_stats: dict[str, Any],
    gate_type: str = "CDS",
    metadata: Optional[dict[str, Any]] = None,
    baseline_id: str | None = None,
    epsilon: float | None = None,
) -> CandidateSkillRecord:
    """Build a certified-candidate record from baseline-relative improvements."""
    calculator = ImprovementCalculator(baseline_stats)
    delta_r, delta_n = calculator.compute_improvements(skill_payoff=skill_payoff, skill_motives=skill_motives)

    gate_type_normalized = gate_type.strip().upper()
    if gate_type_normalized == "CDS":
        gate = CDSGate()
        effective_epsilon = 0.0
        admission_margin = gate.get_admission_margin(delta_r, delta_n)
    elif gate_type_normalized == "PDS":
        gate = PDSGate(epsilon=0.1 if epsilon is None else float(epsilon))
        effective_epsilon = gate.get_epsilon()
        admission_margin = gate.get_admission_margin(delta_r, delta_n)
    else:
        raise ValueError(f"gate_type must be 'CDS' or 'PDS', got {gate_type!r}")

    is_certified = gate.admit(delta_r, delta_n)
    return CandidateSkillRecord(
        skill_id=skill_id,
        delta_r=delta_r,
        delta_n=tuple(float(v) for v in delta_n),
        is_certified=is_certified,
        gate_type=gate.get_gate_type(),
        metadata={} if metadata is None else dict(metadata),
        admission_margin=admission_margin,
        epsilon=effective_epsilon,
        baseline_id=baseline_id,
    )


def build_candidate_skill_records(
    *,
    skill_outcomes: Iterable[dict[str, Any]],
    baseline_stats: dict[str, Any],
    gate_type: str = "CDS",
    baseline_id: str | None = None,
    epsilon: float | None = None,
) -> tuple[CandidateSkillRecord, ...]:
    """Build a tuple of candidate records from iterable skill outcome payloads."""
    records = []
    for outcome in skill_outcomes:
        if "skill_id" not in outcome or "payoff" not in outcome or "motives" not in outcome:
            raise ValueError("Each skill outcome must contain 'skill_id', 'payoff', and 'motives'")
        records.append(
            build_candidate_skill_record(
                skill_id=str(outcome["skill_id"]),
                skill_payoff=float(outcome["payoff"]),
                skill_motives=outcome["motives"],
                baseline_stats=baseline_stats,
                gate_type=gate_type,
                metadata=outcome.get("metadata"),
                baseline_id=baseline_id,
                epsilon=epsilon,
            )
        )
    return tuple(records)


def build_decision_record_from_outcome(
    *,
    context,
    alpha,
    support_values,
    weights_used,
    candidate_skills: Iterable[CandidateSkillRecord],
    selected_skill_id: str,
    selected_score: float | None,
    actual_payoff: float | None,
    actual_motives,
    utility: float | None = None,
) -> MDNDecisionRecord:
    """Build an MDN decision record from a selected-skill outcome payload."""
    return build_decision_record(
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

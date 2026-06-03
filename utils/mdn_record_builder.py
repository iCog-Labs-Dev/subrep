"""Builders that connect prepared outcomes, baseline stats, and certification to MDN records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord
from utils.mdn_logging import build_decision_record
from utils.weight_set_store import WeightSet, WeightSetStore


@dataclass(frozen=True)
class PreparedCandidateOutcome:
    """Stable prepared candidate-outcome contract for MDN-side assembly."""

    context: tuple[float, ...]
    skill_id: str
    payoff: float
    motives: tuple[float, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    gate_type: str = "CDS"
    epsilon: float | None = None

    def __post_init__(self) -> None:
        context = np.asarray(self.context, dtype=np.float32).reshape(-1)
        if context.ndim != 1 or context.shape[0] == 0:
            raise ValueError("context must be a non-empty 1D vector")
        if not np.all(np.isfinite(context)):
            raise ValueError("context must contain only finite values")
        object.__setattr__(self, "context", tuple(float(v) for v in context))

        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ValueError("skill_id must be a non-empty string")

        payoff = float(self.payoff)
        if not np.isfinite(payoff):
            raise ValueError(f"payoff must be finite, got {self.payoff}")
        object.__setattr__(self, "payoff", payoff)

        motives = np.asarray(self.motives, dtype=np.float32).reshape(-1)
        if motives.shape != (2,):
            raise ValueError(f"motives must have shape (2,), got {motives.shape}")
        if not np.all(np.isfinite(motives)):
            raise ValueError("motives must contain only finite values")
        object.__setattr__(self, "motives", tuple(float(v) for v in motives))

        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dict, got {type(self.metadata).__name__}")

        gate_type = self.gate_type.strip().upper()
        if gate_type not in {"CDS", "PDS"}:
            raise ValueError(f"gate_type must be 'CDS' or 'PDS', got {self.gate_type!r}")
        object.__setattr__(self, "gate_type", gate_type)

        if self.epsilon is not None:
            epsilon = float(self.epsilon)
            if not np.isfinite(epsilon) or epsilon < 0.0:
                raise ValueError(f"epsilon must be finite and non-negative, got {self.epsilon}")
            object.__setattr__(self, "epsilon", epsilon)


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
    weight_set: WeightSet | None = None,
) -> CandidateSkillRecord:
    """Build a certified-candidate record from baseline-relative improvements."""
    calculator = ImprovementCalculator(baseline_stats)
    delta_r, delta_n = calculator.compute_improvements(skill_payoff=skill_payoff, skill_motives=skill_motives)

    gate_type_normalized = gate_type.strip().upper()
    if gate_type_normalized == "CDS":
        gate = CDSGate()
        effective_epsilon = 0.0
        admission_margin = gate.get_admission_margin(delta_r, delta_n, weight_set=weight_set)
    elif gate_type_normalized == "PDS":
        gate = PDSGate(epsilon=0.1 if epsilon is None else float(epsilon))
        effective_epsilon = gate.get_epsilon()
        admission_margin = gate.get_admission_margin(delta_r, delta_n, weight_set=weight_set)
    else:
        raise ValueError(f"gate_type must be 'CDS' or 'PDS', got {gate_type!r}")

    is_certified = gate.admit(delta_r, delta_n, weight_set=weight_set)
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
    skill_outcomes: Iterable[dict[str, Any] | PreparedCandidateOutcome],
    baseline_stats: dict[str, Any],
    gate_type: str = "CDS",
    baseline_id: str | None = None,
    epsilon: float | None = None,
    weight_store: WeightSetStore | None = None,
) -> tuple[CandidateSkillRecord, ...]:
    """Build a tuple of candidate records from iterable skill outcome payloads."""
    records = []
    for outcome in skill_outcomes:
        prepared = _coerce_prepared_candidate_outcome(outcome, default_gate_type=gate_type, default_epsilon=epsilon)
        weight_set = None
        if weight_store is not None:
            context_array = np.asarray(prepared.context, dtype=np.float32)
            weight_set = weight_store._store.get(weight_store._context_key(context_array))
        records.append(
            build_candidate_skill_record(
                skill_id=prepared.skill_id,
                skill_payoff=prepared.payoff,
                skill_motives=prepared.motives,
                baseline_stats=baseline_stats,
                gate_type=prepared.gate_type,
                metadata=prepared.metadata,
                baseline_id=baseline_id,
                epsilon=prepared.epsilon,
                weight_set=weight_set,
            )
        )
    return tuple(records)


def group_candidate_outcomes_by_context(
    skill_outcomes: Iterable[dict[str, Any] | PreparedCandidateOutcome],
    *,
    default_gate_type: str = "CDS",
    default_epsilon: float | None = None,
) -> dict[tuple[float, ...], tuple[PreparedCandidateOutcome, ...]]:
    """Group prepared candidate outcomes by exact context vector."""
    grouped: dict[tuple[float, ...], list[PreparedCandidateOutcome]] = {}
    for outcome in skill_outcomes:
        prepared = _coerce_prepared_candidate_outcome(
            outcome,
            default_gate_type=default_gate_type,
            default_epsilon=default_epsilon,
        )
        grouped.setdefault(prepared.context, []).append(prepared)
    return {context: tuple(records) for context, records in grouped.items()}


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


def _coerce_prepared_candidate_outcome(
    outcome: dict[str, Any] | PreparedCandidateOutcome,
    *,
    default_gate_type: str,
    default_epsilon: float | None,
) -> PreparedCandidateOutcome:
    if isinstance(outcome, PreparedCandidateOutcome):
        return outcome
    if "skill_id" not in outcome or "payoff" not in outcome or "motives" not in outcome:
        raise ValueError("Each skill outcome must contain 'skill_id', 'payoff', and 'motives'")

    context = outcome.get("context", outcome.get("obs"))
    if context is None:
        raise ValueError("Each skill outcome must contain either 'context' or 'obs'")

    return PreparedCandidateOutcome(
        context=context,
        skill_id=str(outcome["skill_id"]),
        payoff=float(outcome["payoff"]),
        motives=tuple(float(v) for v in np.asarray(outcome["motives"], dtype=np.float32).reshape(-1)),
        metadata={} if outcome.get("metadata") is None else dict(outcome["metadata"]),
        gate_type=str(outcome.get("gate_type", default_gate_type)),
        epsilon=default_epsilon if outcome.get("epsilon") is None else float(outcome["epsilon"]),
    )

"""Adapters from current rollout-style records into MDN prepared outcomes."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from utils.mdn_record_builder import PreparedCandidateOutcome


def record_to_prepared_candidate_outcome(
    record: dict[str, Any],
    *,
    default_gate_type: str = "CDS",
    default_epsilon: float | None = None,
) -> PreparedCandidateOutcome:
    """Convert a rollout-style record into a PreparedCandidateOutcome.
    """
    required = {"obs", "payoff", "motives", "skill_id"}
    missing = sorted(required - set(record.keys()))
    if missing:
        raise ValueError(f"record is missing required fields: {missing}")

    return PreparedCandidateOutcome(
        context=record["obs"],
        skill_id=str(record["skill_id"]),
        payoff=float(record["payoff"]),
        motives=tuple(float(v) for v in np.asarray(record["motives"], dtype=np.float32).reshape(-1)),
        metadata={} if record.get("metadata") is None else dict(record["metadata"]),
        gate_type=str(record.get("gate_type", default_gate_type)),
        epsilon=default_epsilon if record.get("epsilon") is None else float(record["epsilon"]),
    )


def records_to_prepared_candidate_outcomes(
    records: Iterable[dict[str, Any]],
    *,
    default_gate_type: str = "CDS",
    default_epsilon: float | None = None,
) -> tuple[PreparedCandidateOutcome, ...]:
    """Convert an iterable of rollout-style records into prepared candidate outcomes."""
    return tuple(
        record_to_prepared_candidate_outcome(
            record,
            default_gate_type=default_gate_type,
            default_epsilon=default_epsilon,
        )
        for record in records
    )

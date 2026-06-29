"""Adapters from current rollout-style records into MDN prepared outcomes."""

from __future__ import annotations

from pathlib import Path
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


def candidate_set_file_to_prepared_candidate_outcomes(
    path: str | Path,
    *,
    default_gate_type: str = "CDS",
    default_epsilon: float | None = None,
) -> tuple[PreparedCandidateOutcome, ...]:
    """Load one candidate-set `.npz` file into MDN prepared outcomes.

    A candidate-set file contains one shared context and K candidate outcomes
    collected from that same starting state.
    """
    data = np.load(path, allow_pickle=True)
    required = {
        "context",
        "candidate_skill_ids",
        "candidate_payoffs",
        "candidate_motives",
    }
    missing = sorted(required - set(data.files))
    if missing:
        raise ValueError(f"candidate-set file {path!s} is missing required fields: {missing}")

    context = np.asarray(data["context"], dtype=np.float32).reshape(-1)
    skill_ids = np.asarray(data["candidate_skill_ids"]).reshape(-1)
    payoffs = np.asarray(data["candidate_payoffs"], dtype=np.float32).reshape(-1)
    motives = np.asarray(data["candidate_motives"], dtype=np.float32)

    if len(skill_ids) < 2:
        raise ValueError("candidate-set files must contain at least two candidates")
    if payoffs.shape != (len(skill_ids),):
        raise ValueError(
            f"candidate_payoffs shape {payoffs.shape} must match candidate count {len(skill_ids)}"
        )
    if motives.ndim != 2 or motives.shape[0] != len(skill_ids):
        raise ValueError(
            f"candidate_motives must have shape (K, M) matching candidate count, got {motives.shape}"
        )
    if not np.all(np.isfinite(context)) or not np.all(np.isfinite(payoffs)) or not np.all(np.isfinite(motives)):
        raise ValueError("candidate-set fields must contain only finite context, payoff, and motive values")

    return tuple(
        PreparedCandidateOutcome(
            context=tuple(float(v) for v in context),
            skill_id=str(skill_ids[index]),
            payoff=float(payoffs[index]),
            motives=tuple(float(v) for v in motives[index]),
            metadata={"candidate_set_path": str(path)},
            gate_type=default_gate_type,
            epsilon=default_epsilon,
        )
        for index in range(len(skill_ids))
    )


def candidate_set_directory_to_prepared_candidate_outcomes(
    directory: str | Path,
    *,
    pattern: str = "*.npz",
    default_gate_type: str = "CDS",
    default_epsilon: float | None = None,
) -> tuple[PreparedCandidateOutcome, ...]:
    """Load all candidate-set files in a directory into prepared outcomes."""
    directory_path = Path(directory)
    files = sorted(directory_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No candidate-set files matching {pattern!r} found in {directory_path}")

    outcomes: list[PreparedCandidateOutcome] = []
    for file_path in files:
        outcomes.extend(
            candidate_set_file_to_prepared_candidate_outcomes(
                file_path,
                default_gate_type=default_gate_type,
                default_epsilon=default_epsilon,
            )
        )
    return tuple(outcomes)

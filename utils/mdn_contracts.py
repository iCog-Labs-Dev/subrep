"""Validated contracts for MDN-side candidate and decision records."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np

from utils.cone_utils import validate_simplex_weights


_VALID_GATE_TYPES = {"CDS", "PDS"}


@dataclass(frozen=True)
class CandidateSkillRecord:
    """Normalized candidate-skill record used by downstream MDN systems."""

    skill_id: str
    delta_r: float
    delta_n: tuple[float, float]
    is_certified: bool
    gate_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    admission_margin: float | None = None
    epsilon: float | None = None
    baseline_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ValueError("skill_id must be a non-empty string")

        gate_type = self.gate_type.strip().upper()
        object.__setattr__(self, "gate_type", gate_type)
        if gate_type not in _VALID_GATE_TYPES:
            raise ValueError(f"gate_type must be one of {_VALID_GATE_TYPES}, got {self.gate_type!r}")

        if not isinstance(self.is_certified, bool):
            raise ValueError(f"is_certified must be bool, got {type(self.is_certified).__name__}")

        delta_r = float(self.delta_r)
        if not isfinite(delta_r):
            raise ValueError(f"delta_r must be finite, got {self.delta_r}")
        object.__setattr__(self, "delta_r", delta_r)

        delta_n = tuple(float(v) for v in self.delta_n)
        if len(delta_n) != 2:
            raise ValueError(f"delta_n must have length 2, got {len(delta_n)}")
        if not all(isfinite(v) for v in delta_n):
            raise ValueError(f"delta_n must contain only finite values, got {delta_n}")
        object.__setattr__(self, "delta_n", delta_n)

        if self.admission_margin is not None:
            margin = float(self.admission_margin)
            if not isfinite(margin):
                raise ValueError(f"admission_margin must be finite, got {self.admission_margin}")
            object.__setattr__(self, "admission_margin", margin)

        if self.epsilon is not None:
            epsilon = float(self.epsilon)
            if not isfinite(epsilon) or epsilon < 0.0:
                raise ValueError(f"epsilon must be finite and non-negative, got {self.epsilon}")
            object.__setattr__(self, "epsilon", epsilon)

        if self.baseline_id is not None and (not isinstance(self.baseline_id, str) or not self.baseline_id.strip()):
            raise ValueError("baseline_id must be a non-empty string when provided")

        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dict, got {type(self.metadata).__name__}")


@dataclass(frozen=True)
class MDNDecisionRecord:
    """Decision-time MDN record for auditing and future training replay."""

    context: tuple[float, ...]
    alpha: tuple[float, ...]
    support_values: tuple[float, ...]
    weights_used: tuple[float, ...]
    candidate_skills: tuple[CandidateSkillRecord, ...]
    selected_skill_id: str
    selected_score: float | None = None
    actual_payoff: float | None = None
    actual_motives: tuple[float, float] | None = None
    utility: float | None = None
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        context = _as_finite_vector(self.context, field_name="context")
        alpha = _as_finite_vector(self.alpha, field_name="alpha")
        support_values = _as_finite_vector(self.support_values, field_name="support_values")
        weights_used = _as_finite_vector(self.weights_used, field_name="weights_used")

        if len(alpha) == 0:
            raise ValueError("alpha must be non-empty")
        if len(support_values) != len(alpha):
            raise ValueError("support_values length must match alpha length")
        if len(weights_used) != len(alpha):
            raise ValueError("weights_used length must match alpha length")
        if any(value <= 0.0 for value in alpha):
            raise ValueError(f"alpha must be strictly positive, got {alpha}")
        if any(value < 0.0 for value in support_values):
            raise ValueError(f"support_values must be non-negative, got {support_values}")

        weights_array = np.asarray(weights_used, dtype=float)
        if not validate_simplex_weights(weights_array):
            raise ValueError("weights_used must be a valid simplex vector")

        object.__setattr__(self, "context", context)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "support_values", support_values)
        object.__setattr__(self, "weights_used", tuple(float(v) for v in weights_array))

        if not isinstance(self.candidate_skills, tuple):
            candidates = tuple(self.candidate_skills)
            object.__setattr__(self, "candidate_skills", candidates)
        if len(self.candidate_skills) == 0:
            raise ValueError("candidate_skills must not be empty")
        if not all(isinstance(candidate, CandidateSkillRecord) for candidate in self.candidate_skills):
            raise ValueError("candidate_skills must contain only CandidateSkillRecord instances")

        if not isinstance(self.selected_skill_id, str) or not self.selected_skill_id.strip():
            raise ValueError("selected_skill_id must be a non-empty string")
        if self.selected_skill_id not in {candidate.skill_id for candidate in self.candidate_skills}:
            raise ValueError("selected_skill_id must be present in candidate_skills")

        if self.selected_score is not None:
            score = float(self.selected_score)
            if not isfinite(score):
                raise ValueError(f"selected_score must be finite, got {self.selected_score}")
            object.__setattr__(self, "selected_score", score)

        if self.actual_payoff is not None:
            payoff = float(self.actual_payoff)
            if not isfinite(payoff):
                raise ValueError(f"actual_payoff must be finite, got {self.actual_payoff}")
            object.__setattr__(self, "actual_payoff", payoff)

        if self.actual_motives is not None:
            actual_motives = _as_finite_vector(self.actual_motives, field_name="actual_motives")
            if len(actual_motives) != len(alpha):
                raise ValueError("actual_motives length must match alpha length")
            object.__setattr__(self, "actual_motives", actual_motives)

        if self.utility is not None:
            utility = float(self.utility)
            if not isfinite(utility):
                raise ValueError(f"utility must be finite, got {self.utility}")
            object.__setattr__(self, "utility", utility)


def validate_decision_record(record: MDNDecisionRecord) -> None:
    """Validate an MDN decision record by type and constructor invariants."""
    if not isinstance(record, MDNDecisionRecord):
        raise ValueError(f"record must be MDNDecisionRecord, got {type(record).__name__}")


def _as_finite_vector(values: Any, field_name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be a 1D vector")
    if len(array) == 0:
        raise ValueError(f"{field_name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain only finite values")
    return tuple(float(v) for v in array)

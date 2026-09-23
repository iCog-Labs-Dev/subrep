"""Versioned JSON contracts for recommendation requests and responses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import isclose, isfinite
from typing import Any, Mapping


REQUEST_SCHEMA_VERSION = "subrep.omegaclaw.recommendation.request.v1"
RESPONSE_SCHEMA_VERSION = "subrep.omegaclaw.recommendation.response.v1"
AUDIT_SCHEMA_VERSION = "subrep.omegaclaw.recommendation.audit.v1"
VALID_EVIDENCE_LABELS = {"OBSERVED", "SYNTHETIC"}
VALID_GATE_TYPES = {"CDS", "PDS"}
VALID_WEIGHT_REGION_TYPES = {"FULL_SIMPLEX", "MDN_WX"}
VALID_OUTCOME_STATUSES = {
    "accepted",
    "abstained",
    "invalid_response",
    "backend_error",
}


def _non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _finite(value: float, field_name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _json_safe(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON serializable") from exc


@dataclass(frozen=True)
class ObjectiveWeight:
    """One named objective and its normalized decision weight."""

    objective_id: str
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_id", _non_empty(self.objective_id, "objective_id"))
        weight = _finite(self.weight, "weight")
        if weight < 0.0:
            raise ValueError("weight must be non-negative")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class RiskBudget:
    """Additional local limits applied after SubRep admissibility checks."""

    max_certificate_epsilon: float | None = None
    minimum_admission_margin: float | None = None

    def __post_init__(self) -> None:
        if self.max_certificate_epsilon is not None:
            maximum = _finite(self.max_certificate_epsilon, "max_certificate_epsilon")
            if maximum < 0.0:
                raise ValueError("max_certificate_epsilon must be non-negative")
            object.__setattr__(self, "max_certificate_epsilon", maximum)
        if self.minimum_admission_margin is not None:
            minimum = _finite(self.minimum_admission_margin, "minimum_admission_margin")
            if minimum < 0.0:
                raise ValueError("minimum_admission_margin must be non-negative")
            object.__setattr__(
                self,
                "minimum_admission_margin",
                minimum,
            )


@dataclass(frozen=True)
class SkillEvidence:
    """SubRep-owned evidence for one locally admitted recommendation option."""

    skill_id: str
    score: float
    delta_r: float
    objective_deltas: Mapping[str, float]
    gate_type: str
    admission_margin: float
    epsilon: float
    weight_region_type: str
    evidence_label: str = "OBSERVED"

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", _non_empty(self.skill_id, "skill_id"))
        object.__setattr__(self, "score", _finite(self.score, "score"))
        object.__setattr__(self, "delta_r", _finite(self.delta_r, "delta_r"))
        gate_type = _non_empty(self.gate_type, "gate_type").upper()
        if gate_type not in VALID_GATE_TYPES:
            raise ValueError(f"gate_type must be one of {VALID_GATE_TYPES}")
        object.__setattr__(self, "gate_type", gate_type)
        weight_region_type = _non_empty(
            self.weight_region_type, "weight_region_type"
        ).upper()
        if weight_region_type not in VALID_WEIGHT_REGION_TYPES:
            raise ValueError(
                f"weight_region_type must be one of {VALID_WEIGHT_REGION_TYPES}"
            )
        object.__setattr__(
            self,
            "weight_region_type",
            weight_region_type,
        )
        admission_margin = _finite(self.admission_margin, "admission_margin")
        if admission_margin < 0.0:
            raise ValueError("admission_margin must be non-negative")
        object.__setattr__(self, "admission_margin", admission_margin)
        epsilon = _finite(self.epsilon, "epsilon")
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        object.__setattr__(self, "epsilon", epsilon)

        deltas = {
            _non_empty(key, "objective_deltas key"): _finite(value, "objective delta")
            for key, value in self.objective_deltas.items()
        }
        if not deltas:
            raise ValueError("objective_deltas must not be empty")
        object.__setattr__(self, "objective_deltas", deltas)

        label = _non_empty(self.evidence_label, "evidence_label").upper()
        if label not in VALID_EVIDENCE_LABELS:
            raise ValueError(f"evidence_label must be one of {VALID_EVIDENCE_LABELS}")
        object.__setattr__(self, "evidence_label", label)


@dataclass(frozen=True)
class SkillExclusion:
    """A SubRep-owned reason that a skill is not eligible for recommendation."""

    skill_id: str
    reason_code: str
    reason: str
    evidence_label: str = "OBSERVED"

    def __post_init__(self) -> None:
        object.__setattr__(self, "skill_id", _non_empty(self.skill_id, "skill_id"))
        object.__setattr__(self, "reason_code", _non_empty(self.reason_code, "reason_code").upper())
        object.__setattr__(self, "reason", _non_empty(self.reason, "reason"))
        label = _non_empty(self.evidence_label, "evidence_label").upper()
        if label not in VALID_EVIDENCE_LABELS:
            raise ValueError(f"evidence_label must be one of {VALID_EVIDENCE_LABELS}")
        object.__setattr__(self, "evidence_label", label)


@dataclass(frozen=True)
class RecommendationRequest:
    """Complete immutable snapshot sent to Omega for recommendation only."""

    request_id: str
    created_at: str
    task_context: Mapping[str, Any]
    objective_weights: tuple[ObjectiveWeight, ...]
    risk_budget: RiskBudget
    admitted_skills: tuple[SkillEvidence, ...]
    exclusions: tuple[SkillExclusion, ...] = ()
    allow_abstain: bool = True
    evidence_label: str = "OBSERVED"
    schema_version: str = REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _non_empty(self.request_id, "request_id"))
        if self.schema_version != REQUEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported request schema_version: {self.schema_version}")
        try:
            created_at = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("created_at must be an ISO-8601 timestamp") from exc
        if created_at.utcoffset() is None:
            raise ValueError("created_at must include a timezone offset")
        if not isinstance(self.task_context, Mapping):
            raise ValueError("task_context must be a mapping")
        _json_safe(self.task_context, "task_context")

        if not isinstance(self.risk_budget, RiskBudget):
            raise ValueError("risk_budget must be a RiskBudget")
        label = _non_empty(self.evidence_label, "evidence_label").upper()
        if label not in VALID_EVIDENCE_LABELS:
            raise ValueError(f"evidence_label must be one of {VALID_EVIDENCE_LABELS}")
        object.__setattr__(self, "evidence_label", label)

        objectives = tuple(self.objective_weights)
        if len(objectives) < 2:
            raise ValueError("objective_weights must contain at least two objectives")
        if not all(isinstance(item, ObjectiveWeight) for item in objectives):
            raise ValueError("objective_weights must contain only ObjectiveWeight values")
        objective_ids = [item.objective_id for item in objectives]
        if len(objective_ids) != len(set(objective_ids)):
            raise ValueError("objective_weights must have unique objective_id values")
        if not isclose(sum(item.weight for item in objectives), 1.0, abs_tol=1e-8):
            raise ValueError("objective weights must sum to 1")
        object.__setattr__(self, "objective_weights", objectives)

        admitted = tuple(self.admitted_skills)
        exclusions = tuple(self.exclusions)
        if not all(isinstance(item, SkillEvidence) for item in admitted):
            raise ValueError("admitted_skills must contain only SkillEvidence values")
        if not all(isinstance(item, SkillExclusion) for item in exclusions):
            raise ValueError("exclusions must contain only SkillExclusion values")
        admitted_ids = [item.skill_id for item in admitted]
        excluded_ids = [item.skill_id for item in exclusions]
        if len(admitted_ids) != len(set(admitted_ids)):
            raise ValueError("admitted_skills must have unique skill_id values")
        if len(excluded_ids) != len(set(excluded_ids)):
            raise ValueError("exclusions must have unique skill_id values")
        overlap = set(admitted_ids).intersection(excluded_ids)
        if overlap:
            raise ValueError(f"skills cannot be both admitted and excluded: {sorted(overlap)}")

        weights = {item.objective_id: item.weight for item in objectives}
        for skill in admitted:
            if skill.evidence_label != label:
                raise ValueError(
                    f"skill {skill.skill_id!r} evidence_label must match the request"
                )
            if set(skill.objective_deltas) != set(weights):
                raise ValueError(
                    f"skill {skill.skill_id!r} objective_deltas must match objective_weights"
                )
            expected_score = skill.delta_r + sum(
                weights[name] * skill.objective_deltas[name] for name in weights
            )
            if not isclose(skill.score, expected_score, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"skill {skill.skill_id!r} score does not match delta_r + weighted deltas"
                )
            if (
                self.risk_budget.max_certificate_epsilon is not None
                and skill.epsilon > self.risk_budget.max_certificate_epsilon
            ):
                raise ValueError(
                    f"skill {skill.skill_id!r} exceeds max_certificate_epsilon"
                )
            if (
                self.risk_budget.minimum_admission_margin is not None
                and skill.admission_margin < self.risk_budget.minimum_admission_margin
            ):
                raise ValueError(
                    f"skill {skill.skill_id!r} is below minimum_admission_margin"
                )
        for exclusion in exclusions:
            if exclusion.evidence_label != label:
                raise ValueError(
                    f"exclusion {exclusion.skill_id!r} evidence_label must match the request"
                )
        object.__setattr__(self, "admitted_skills", admitted)
        object.__setattr__(self, "exclusions", exclusions)

        if not isinstance(self.allow_abstain, bool):
            raise ValueError("allow_abstain must be bool")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OmegaRecommendationResponse:
    """Structured response expected from Omega."""

    request_id: str
    selected_skill_id: str | None
    abstain: bool
    explanation: str
    cited_skill_ids: tuple[str, ...] = ()
    schema_version: str = RESPONSE_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OmegaRecommendationResponse":
        required = {"schema_version", "request_id", "selected_skill_id", "abstain", "explanation"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"response is missing required fields: {sorted(missing)}")
        citations = payload.get("cited_skill_ids", ())
        if not isinstance(citations, (list, tuple)):
            raise ValueError("cited_skill_ids must be an array")
        return cls(
            schema_version=payload["schema_version"],
            request_id=payload["request_id"],
            selected_skill_id=payload["selected_skill_id"],
            abstain=payload["abstain"],
            explanation=payload["explanation"],
            cited_skill_ids=tuple(citations),
        )

    def __post_init__(self) -> None:
        if self.schema_version != RESPONSE_SCHEMA_VERSION:
            raise ValueError(f"unsupported response schema_version: {self.schema_version}")
        object.__setattr__(self, "request_id", _non_empty(self.request_id, "request_id"))
        if not isinstance(self.abstain, bool):
            raise ValueError("abstain must be bool")
        explanation = _non_empty(self.explanation, "explanation")
        if len(explanation) > 1000:
            raise ValueError("explanation must be at most 1000 characters")
        object.__setattr__(self, "explanation", explanation)

        if self.abstain:
            if self.selected_skill_id is not None:
                raise ValueError("selected_skill_id must be null when abstain is true")
        else:
            object.__setattr__(
                self,
                "selected_skill_id",
                _non_empty(self.selected_skill_id, "selected_skill_id"),
            )

        citations = tuple(_non_empty(item, "cited_skill_ids item") for item in self.cited_skill_ids)
        if len(citations) != len(set(citations)):
            raise ValueError("cited_skill_ids must not contain duplicates")
        object.__setattr__(self, "cited_skill_ids", citations)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_response_for_request(
    request: RecommendationRequest,
    response: OmegaRecommendationResponse,
) -> None:
    """Validate identity, eligibility, abstention, and evidence references."""

    if response.request_id != request.request_id:
        raise ValueError("response request_id does not match the request")

    admitted_ids = {item.skill_id for item in request.admitted_skills}
    known_ids = admitted_ids.union(item.skill_id for item in request.exclusions)
    unknown_citations = set(response.cited_skill_ids).difference(known_ids)
    if unknown_citations:
        raise ValueError(f"response cites unknown skills: {sorted(unknown_citations)}")

    if response.abstain:
        if not request.allow_abstain:
            raise ValueError("response abstained when the request disallowed abstention")
        return

    if response.selected_skill_id not in admitted_ids:
        raise ValueError("selected_skill_id is not in the supplied admitted set")
    if response.selected_skill_id not in response.cited_skill_ids:
        raise ValueError("selected_skill_id must appear in cited_skill_ids")


@dataclass(frozen=True)
class RecommendationOutcome:
    """Validated adapter outcome; it never represents skill execution."""

    request_id: str
    status: str
    selected_skill_id: str | None = None
    explanation: str | None = None
    error: str | None = None
    raw_response: str | None = field(default=None, repr=False)
    response: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _non_empty(self.request_id, "request_id"))
        if self.status not in VALID_OUTCOME_STATUSES:
            raise ValueError(f"status must be one of {VALID_OUTCOME_STATUSES}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

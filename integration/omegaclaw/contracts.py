"""Versioned JSON contracts for SubRep decisions and Omega explanations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import isclose, isfinite
from types import MappingProxyType
from typing import Any, Mapping


REQUEST_SCHEMA_VERSION = "subrep.omegaclaw.explanation.request.v1"
RESPONSE_SCHEMA_VERSION = "subrep.omegaclaw.explanation.response.v1"
AUDIT_SCHEMA_VERSION = "subrep.omegaclaw.explanation.audit.v1"
SELECTION_RULE = "MAX_FINAL_SELECTION_SCORE"
TIE_BREAKER = "LEXICOGRAPHIC_SKILL_ID"
INTEGRATION_MODE = "EXPLANATION_ONLY"
SELECTION_BASIS = "SUBREP_DECISION"
NO_ADMISSIBLE_SKILLS = "NO_ADMISSIBLE_SKILLS"
VALID_ADVISORY_CONCERN_CODES = {
    "SMALL_SCORE_GAP",
    "NEGATIVE_OBJECTIVE_TRADEOFF",
    "PDS_EPSILON_USED",
    "LOW_CERTIFICATE_MARGIN",
    "HIGH_PRIORITY_OBJECTIVE_DECLINE",
}
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


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(child) for child in value]
    return value


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
        object.__setattr__(self, "objective_deltas", MappingProxyType(deltas))

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
class SubRepDecision:
    """Authoritative selection or abstention computed entirely by SubRep."""

    selected_skill_id: str | None
    selected_score: float | None
    runner_up_skill_id: str | None
    runner_up_score: float | None
    score_gap: float | None
    abstain: bool
    reason_code: str | None = None
    selection_rule: str = SELECTION_RULE
    tie_breaker: str = TIE_BREAKER

    def __post_init__(self) -> None:
        if not isinstance(self.abstain, bool):
            raise ValueError("abstain must be bool")
        if self.selection_rule != SELECTION_RULE:
            raise ValueError(f"selection_rule must be {SELECTION_RULE!r}")
        if self.tie_breaker != TIE_BREAKER:
            raise ValueError(f"tie_breaker must be {TIE_BREAKER!r}")

        if self.abstain:
            if any(
                value is not None
                for value in (
                    self.selected_skill_id,
                    self.selected_score,
                    self.runner_up_skill_id,
                    self.runner_up_score,
                    self.score_gap,
                )
            ):
                raise ValueError("abstaining decisions must not contain selected or runner-up values")
            if self.reason_code != NO_ADMISSIBLE_SKILLS:
                raise ValueError(f"abstaining reason_code must be {NO_ADMISSIBLE_SKILLS!r}")
            return

        object.__setattr__(
            self,
            "selected_skill_id",
            _non_empty(self.selected_skill_id, "selected_skill_id"),
        )
        object.__setattr__(
            self,
            "selected_score",
            _finite(self.selected_score, "selected_score"),
        )
        if self.reason_code is not None:
            raise ValueError("non-abstaining decisions must have reason_code == None")

        if self.runner_up_skill_id is None:
            if self.runner_up_score is not None or self.score_gap is not None:
                raise ValueError("runner_up_score and score_gap require runner_up_skill_id")
            return

        runner_id = _non_empty(self.runner_up_skill_id, "runner_up_skill_id")
        if runner_id == self.selected_skill_id:
            raise ValueError("runner_up_skill_id must differ from selected_skill_id")
        runner_score = _finite(self.runner_up_score, "runner_up_score")
        score_gap = _finite(self.score_gap, "score_gap")
        if score_gap < 0.0:
            raise ValueError("score_gap must be non-negative")
        if not isclose(
            score_gap,
            float(self.selected_score) - runner_score,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("score_gap must equal selected_score - runner_up_score")
        object.__setattr__(self, "runner_up_skill_id", runner_id)
        object.__setattr__(self, "runner_up_score", runner_score)
        object.__setattr__(self, "score_gap", score_gap)


def build_subrep_decision(skills: tuple[SkillEvidence, ...]) -> SubRepDecision:
    """Create the authoritative decision using SubRep's stable score ordering."""

    ranked = sorted(skills, key=lambda skill: (-skill.score, skill.skill_id))
    if not ranked:
        return SubRepDecision(
            selected_skill_id=None,
            selected_score=None,
            runner_up_skill_id=None,
            runner_up_score=None,
            score_gap=None,
            abstain=True,
            reason_code=NO_ADMISSIBLE_SKILLS,
        )
    selected = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    return SubRepDecision(
        selected_skill_id=selected.skill_id,
        selected_score=selected.score,
        runner_up_skill_id=runner_up.skill_id if runner_up else None,
        runner_up_score=runner_up.score if runner_up else None,
        score_gap=selected.score - runner_up.score if runner_up else None,
        abstain=False,
    )


@dataclass(frozen=True)
class RecommendationRequest:
    """Complete immutable SubRep decision sent to OmegaClaw for explanation."""

    request_id: str
    created_at: str
    task_context: Mapping[str, Any]
    objective_weights: tuple[ObjectiveWeight, ...]
    risk_budget: RiskBudget
    admitted_skills: tuple[SkillEvidence, ...]
    subrep_decision: SubRepDecision | None = None
    exclusions: tuple[SkillExclusion, ...] = ()
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
        task_context = _thaw_json(self.task_context)
        _json_safe(task_context, "task_context")
        object.__setattr__(self, "task_context", _freeze_json(task_context))

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

        expected_decision = build_subrep_decision(admitted)
        decision = self.subrep_decision or expected_decision
        if not isinstance(decision, SubRepDecision):
            raise ValueError("subrep_decision must be a SubRepDecision")
        if decision != expected_decision:
            raise ValueError("subrep_decision does not match the admitted skill scores")
        object.__setattr__(self, "subrep_decision", decision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "created_at": self.created_at,
            "mode": INTEGRATION_MODE,
            "subrep_decision": asdict(self.subrep_decision),
            "task_context": _thaw_json(self.task_context),
            "admitted_skills": {
                skill.skill_id: {
                    "skill_id": skill.skill_id,
                    "final_selection_score": skill.score,
                }
                for skill in self.admitted_skills
            },
            "risk_budget": asdict(self.risk_budget),
            "exclusions": [asdict(item) for item in self.exclusions],
            "explanation_evidence": {
                "objective_weights": {
                    item.objective_id: item.weight for item in self.objective_weights
                },
                "skills": {
                    skill.skill_id: {
                        "delta_r": skill.delta_r,
                        "objective_deltas": dict(skill.objective_deltas),
                    }
                    for skill in self.admitted_skills
                },
            },
            "audit_only_certificate_evidence": {
                "instruction": "Do not use these fields to rank or abstain.",
                "skills": {
                    skill.skill_id: {
                        "gate_type": skill.gate_type,
                        "certificate_admission_margin": skill.admission_margin,
                        "epsilon": skill.epsilon,
                        "weight_region_type": skill.weight_region_type,
                    }
                    for skill in self.admitted_skills
                },
            },
            "evidence_label": self.evidence_label,
        }

    def valid_evidence_refs(self) -> set[str]:
        """Return every leaf path OmegaClaw may cite in its explanation."""

        payload = self.to_dict()
        references: set[str] = set()

        def visit(value: Any, prefix: str) -> None:
            if isinstance(value, Mapping):
                for key, child in value.items():
                    visit(child, f"{prefix}.{key}" if prefix else str(key))
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    visit(child, f"{prefix}.{index}")
            else:
                references.add(prefix)

        for root in (
            "subrep_decision",
            "task_context",
            "risk_budget",
            "admitted_skills",
            "exclusions",
            "explanation_evidence",
            "audit_only_certificate_evidence",
        ):
            visit(payload[root], root)
        return references


@dataclass(frozen=True)
class AdvisoryConcern:
    """A non-binding, evidence-grounded concern about the SubRep decision."""

    code: str
    message: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        code = _non_empty(self.code, "advisory concern code").upper()
        if code not in VALID_ADVISORY_CONCERN_CODES:
            raise ValueError(
                f"advisory concern code must be one of {VALID_ADVISORY_CONCERN_CODES}"
            )
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", _non_empty(self.message, "advisory concern message"))
        refs = tuple(_non_empty(item, "advisory evidence reference") for item in self.evidence_refs)
        if not refs:
            raise ValueError("advisory concerns require at least one evidence reference")
        if len(refs) != len(set(refs)):
            raise ValueError("advisory evidence references must not contain duplicates")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class OmegaRecommendationResponse:
    """Structured explanation response expected from OmegaClaw."""

    request_id: str
    selected_skill_id: str | None
    abstain: bool
    explanation: str
    selection_basis: str
    evidence_refs: tuple[str, ...]
    advisory_concerns: tuple[AdvisoryConcern, ...] = ()
    mode: str = INTEGRATION_MODE
    schema_version: str = RESPONSE_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OmegaRecommendationResponse":
        required = {
            "schema_version",
            "request_id",
            "selected_skill_id",
            "abstain",
            "selection_basis",
            "explanation",
            "evidence_refs",
            "advisory_concerns",
            "mode",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"response is missing required fields: {sorted(missing)}")
        unexpected = set(payload).difference(required)
        if unexpected:
            raise ValueError(f"response contains unexpected fields: {sorted(unexpected)}")
        refs = payload["evidence_refs"]
        if not isinstance(refs, (list, tuple)):
            raise ValueError("evidence_refs must be an array")
        concerns_payload = payload["advisory_concerns"]
        if not isinstance(concerns_payload, (list, tuple)):
            raise ValueError("advisory_concerns must be an array")
        concerns = tuple(
            item if isinstance(item, AdvisoryConcern) else AdvisoryConcern(**item)
            for item in concerns_payload
        )
        return cls(
            schema_version=payload["schema_version"],
            request_id=payload["request_id"],
            selected_skill_id=payload["selected_skill_id"],
            abstain=payload["abstain"],
            selection_basis=payload["selection_basis"],
            explanation=payload["explanation"],
            evidence_refs=tuple(refs),
            advisory_concerns=concerns,
            mode=payload["mode"],
        )

    def __post_init__(self) -> None:
        if self.schema_version != RESPONSE_SCHEMA_VERSION:
            raise ValueError(f"unsupported response schema_version: {self.schema_version}")
        object.__setattr__(self, "request_id", _non_empty(self.request_id, "request_id"))
        if not isinstance(self.abstain, bool):
            raise ValueError("abstain must be bool")
        if self.selection_basis != SELECTION_BASIS:
            raise ValueError(f"selection_basis must be {SELECTION_BASIS!r}")
        if self.mode != INTEGRATION_MODE:
            raise ValueError(f"mode must be {INTEGRATION_MODE!r}")
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

        refs = tuple(_non_empty(item, "evidence_refs item") for item in self.evidence_refs)
        if not refs:
            raise ValueError("evidence_refs must not be empty")
        if len(refs) != len(set(refs)):
            raise ValueError("evidence_refs must not contain duplicates")
        object.__setattr__(self, "evidence_refs", refs)
        concerns = tuple(self.advisory_concerns)
        if not all(isinstance(item, AdvisoryConcern) for item in concerns):
            raise ValueError("advisory_concerns must contain AdvisoryConcern values")
        object.__setattr__(self, "advisory_concerns", concerns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_response_for_request(
    request: RecommendationRequest,
    response: OmegaRecommendationResponse,
) -> None:
    """Validate identity, eligibility, abstention, and evidence references."""

    if response.request_id != request.request_id:
        raise ValueError("response request_id does not match the request")

    decision = request.subrep_decision
    if response.selected_skill_id != decision.selected_skill_id:
        raise ValueError("selected_skill_id does not echo the authoritative SubRep decision")
    if response.abstain != decision.abstain:
        raise ValueError("response abstain does not echo the authoritative SubRep decision")

    valid_refs = request.valid_evidence_refs()
    unknown_refs = set(response.evidence_refs).difference(valid_refs)
    if unknown_refs:
        raise ValueError(f"response contains unknown evidence_refs: {sorted(unknown_refs)}")

    required_ref = (
        "subrep_decision.reason_code"
        if decision.abstain
        else "subrep_decision.selected_score"
    )
    if required_ref not in response.evidence_refs:
        raise ValueError(f"response must cite {required_ref!r}")
    if decision.runner_up_skill_id is not None and not decision.abstain:
        if "subrep_decision.runner_up_score" not in response.evidence_refs:
            raise ValueError("response must cite 'subrep_decision.runner_up_score'")

    explanation_lower = response.explanation.lower()
    if decision.abstain:
        if not any(term in explanation_lower for term in ("abstain", "no admissible", "no recommendation")):
            raise ValueError("abstention explanation must describe the lack of a recommendation")
    elif decision.selected_skill_id.lower() not in explanation_lower:
        raise ValueError("explanation must mention the selected skill ID")

    for concern in response.advisory_concerns:
        unknown_concern_refs = set(concern.evidence_refs).difference(valid_refs)
        if unknown_concern_refs:
            raise ValueError(
                f"advisory concern contains unknown evidence_refs: {sorted(unknown_concern_refs)}"
            )
        _validate_advisory_concern(request, concern)


def _validate_advisory_concern(
    request: RecommendationRequest,
    concern: AdvisoryConcern,
) -> None:
    """Require each advisory code to cite evidence that is relevant to its claim."""

    decision = request.subrep_decision
    if decision.abstain:
        raise ValueError("advisory concerns are not allowed for an abstaining decision")

    selected = next(
        skill for skill in request.admitted_skills if skill.skill_id == decision.selected_skill_id
    )
    refs = set(concern.evidence_refs)
    decision_prefix = "subrep_decision"
    explanation_prefix = f"explanation_evidence.skills.{selected.skill_id}"
    certificate_prefix = f"audit_only_certificate_evidence.skills.{selected.skill_id}"

    if concern.code == "SMALL_SCORE_GAP":
        if decision.runner_up_skill_id is None or f"{decision_prefix}.score_gap" not in refs:
            raise ValueError("SMALL_SCORE_GAP must cite subrep_decision.score_gap")
        return

    if concern.code == "NEGATIVE_OBJECTIVE_TRADEOFF":
        negative_refs = {
            f"{explanation_prefix}.objective_deltas.{objective_id}"
            for objective_id, delta in selected.objective_deltas.items()
            if delta < 0.0
        }
        if not refs.intersection(negative_refs):
            raise ValueError(
                "NEGATIVE_OBJECTIVE_TRADEOFF must cite a negative selected-skill objective delta"
            )
        return

    if concern.code == "PDS_EPSILON_USED":
        required = {
            f"{certificate_prefix}.gate_type",
            f"{certificate_prefix}.epsilon",
        }
        if selected.gate_type != "PDS" or selected.epsilon <= 0.0 or not required.issubset(refs):
            raise ValueError(
                "PDS_EPSILON_USED requires a positive PDS epsilon and its gate_type and epsilon refs"
            )
        return

    if concern.code == "LOW_CERTIFICATE_MARGIN":
        required_ref = f"{certificate_prefix}.certificate_admission_margin"
        if required_ref not in refs:
            raise ValueError(
                "LOW_CERTIFICATE_MARGIN must cite the selected skill's certificate margin"
            )
        return

    if concern.code == "HIGH_PRIORITY_OBJECTIVE_DECLINE":
        weights = {item.objective_id: item.weight for item in request.objective_weights}
        maximum_weight = max(weights.values())
        supported = any(
            delta < 0.0
            and weights[objective_id] == maximum_weight
            and {
                f"explanation_evidence.objective_weights.{objective_id}",
                f"{explanation_prefix}.objective_deltas.{objective_id}",
            }.issubset(refs)
            for objective_id, delta in selected.objective_deltas.items()
        )
        if not supported:
            raise ValueError(
                "HIGH_PRIORITY_OBJECTIVE_DECLINE must cite the weight and negative delta of a highest-weight objective"
            )


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
    attempts: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _non_empty(self.request_id, "request_id"))
        if self.status not in VALID_OUTCOME_STATUSES:
            raise ValueError(f"status must be one of {VALID_OUTCOME_STATUSES}")
        attempts = tuple(dict(item) for item in self.attempts)
        _json_safe(attempts, "attempts")
        object.__setattr__(self, "attempts", attempts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

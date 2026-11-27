"""Versioned JSON contracts for SubRep decisions and Omega explanations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import isclose, isfinite
from types import MappingProxyType
from typing import Any, Mapping


REQUEST_SCHEMA_VERSION = "subrep.omegaclaw.explanation.request.v3"
# v2: Omega returns structured claims instead of a free-text explanation, and SubRep
# writes the explanation from the claims it has verified.
RESPONSE_SCHEMA_VERSION = "subrep.omegaclaw.explanation.response.v2"
AUDIT_SCHEMA_VERSION = "subrep.omegaclaw.explanation.audit.v4"
SELECTION_RULE = "MAX_FINAL_SELECTION_SCORE"
TIE_BREAKER = "LEXICOGRAPHIC_SKILL_ID"
INTEGRATION_MODE = "EXPLANATION_ONLY"
SELECTION_BASIS = "SUBREP_DECISION"
NO_ADMISSIBLE_SKILLS = "NO_ADMISSIBLE_SKILLS"
# A response lists only the facts that explain one decision; this bounds the work.
MAX_CLAIMS = 40
# A claimed number matches the request when it is equal up to float rounding. The
# tolerance is relative, so a small value such as an epsilon of 5e-7 cannot pass as 0.
# The prompt shows numbers to 12 significant digits, well inside this tolerance.
CLAIM_RELATIVE_TOLERANCE = 1e-9
CLAIM_ABSOLUTE_TOLERANCE = 1e-12
# Advisory thresholds. Defaults to confirm with the product owner: a score gap is
# small at no more than 10% of the selected score, and a certificate margin is low at
# no more than 10% above the request's minimum admission margin.
SMALL_SCORE_GAP_FRACTION = 0.10
LOW_CERTIFICATE_MARGIN_FRACTION = 0.10
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

CERTIFICATION_STAGE = "CERTIFICATION"
RUNTIME_ADMISSIBILITY_STAGE = "RUNTIME_ADMISSIBILITY"
RISK_BUDGET_STAGE = "RISK_BUDGET"
OPERATOR_STAGE = "OPERATOR"
OBJECTIVE_COMPATIBILITY_STAGE = "OBJECTIVE_COMPATIBILITY"


@dataclass(frozen=True)
class ExclusionReason:
    """The SubRep stage that removes a skill for one reason code, and what it means."""

    stage: str
    description: str


# Descriptions avoid apostrophes because Omega rewrites them before its model sees the text.
EXCLUSION_REASONS: Mapping[str, ExclusionReason] = MappingProxyType(
    {
        "GATE_FAILED": ExclusionReason(
            CERTIFICATION_STAGE,
            "Failed the SubRep CDS/PDS certification gates, so it holds no certificate.",
        ),
        "LIBRARY_REVERIFICATION_FAILED": ExclusionReason(
            CERTIFICATION_STAGE,
            "Failed SubRep re-verification of its certificate at the skill library, "
            "so it holds no valid certificate.",
        ),
        "NOT_RUNTIME_ADMISSIBLE": ExclusionReason(
            RUNTIME_ADMISSIBILITY_STAGE,
            "Holds a SubRep certificate, but the runtime admissibility check did not "
            "admit it for the current objective weights. This is not a certification failure.",
        ),
        "RISK_BUDGET_EXCEEDED": ExclusionReason(
            RISK_BUDGET_STAGE,
            "Holds a SubRep certificate and is runtime admissible, but its certificate "
            "epsilon exceeds the local risk budget of this request. This is not a "
            "certification failure.",
        ),
        "INSUFFICIENT_ADMISSION_MARGIN": ExclusionReason(
            RISK_BUDGET_STAGE,
            "Holds a SubRep certificate and is runtime admissible, but its admission "
            "margin is below the local minimum of this request. This is not a "
            "certification failure.",
        ),
        "OPERATOR_EXCLUSION": ExclusionReason(
            OPERATOR_STAGE,
            "Removed by an explicit operator instruction for this request. This is not a "
            "certification, runtime admissibility, or risk budget result.",
        ),
        "OBJECTIVE_DIMENSION_MISMATCH": ExclusionReason(
            OBJECTIVE_COMPATIBILITY_STAGE,
            "Holds a SubRep certificate for a different number of objectives than this "
            "request, so its evidence cannot be scored here. This is not a certification "
            "failure.",
        ),
    }
)
EXCLUSION_STAGES = frozenset(reason.stage for reason in EXCLUSION_REASONS.values())

# The facts a claim can state, each identified by the shape of its evidence path.
CLAIM_PATHS: Mapping[str, re.Pattern[str]] = MappingProxyType(
    {
        "score": re.compile(
            r"admitted_skills\.[^.]+\.final_selection_score"
            r"|subrep_decision\.(?:selected|runner_up)_score"
        ),
        "score_gap": re.compile(r"subrep_decision\.score_gap"),
        "objective_weight": re.compile(r"explanation_evidence\.objective_weights\.[^.]+"),
        "objective_delta": re.compile(
            r"explanation_evidence\.skills\.[^.]+\.objective_deltas\.[^.]+"
        ),
        "delta_r": re.compile(r"explanation_evidence\.skills\.[^.]+\.delta_r"),
        "certificate": re.compile(
            r"audit_only_certificate_evidence\.skills\.[^.]+\."
            r"(?:gate_type|epsilon|certificate_admission_margin|weight_region_type)"
        ),
        "risk_budget": re.compile(
            r"risk_budget\.(?:max_certificate_epsilon|minimum_admission_margin)"
        ),
        "exclusion_stage": re.compile(r"exclusions\.\d+\.stage"),
    }
)
CLAIM_TYPES = frozenset(CLAIM_PATHS)


def claim_type_of(reference: str) -> str | None:
    """The claim type whose facts live at this evidence path, or None if no claim can cite it."""

    for claim_type, pattern in CLAIM_PATHS.items():
        if pattern.fullmatch(reference):
            return claim_type
    return None


def canonical_evidence_ref(reference: str) -> str:
    """Write an evidence path in dotted form: "exclusions[0].stage" is "exclusions.0.stage".

    Models often index lists with brackets. Both forms name the same field, so the
    bracket form is accepted and stored dotted; a path that does not exist is still rejected.
    """

    return re.sub(r"\[(\d+)\]", r".\1", reference)


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
        reason_code = _non_empty(self.reason_code, "reason_code").upper()
        if reason_code not in EXCLUSION_REASONS:
            raise ValueError(
                f"reason_code must be one of {sorted(EXCLUSION_REASONS)}, got {reason_code!r}"
            )
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "reason", _non_empty(self.reason, "reason"))
        label = _non_empty(self.evidence_label, "evidence_label").upper()
        if label not in VALID_EVIDENCE_LABELS:
            raise ValueError(f"evidence_label must be one of {VALID_EVIDENCE_LABELS}")
        object.__setattr__(self, "evidence_label", label)

    @property
    def stage(self) -> str:
        return EXCLUSION_REASONS[self.reason_code].stage

    @property
    def stage_description(self) -> str:
        return EXCLUSION_REASONS[self.reason_code].description

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "stage": self.stage,
            "stage_description": self.stage_description,
        }


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
            "objective_order": [item.objective_id for item in self.objective_weights],
            "admitted_skills": {
                skill.skill_id: {
                    "skill_id": skill.skill_id,
                    "final_selection_score": skill.score,
                }
                for skill in self.admitted_skills
            },
            "risk_budget": asdict(self.risk_budget),
            "exclusions": [item.to_dict() for item in self.exclusions],
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
        """Return every path OmegaClaw may cite: each value and each section holding them.

        Sections count too, so an explanation can cite a whole list such as
        objective_order, or an empty one such as admitted_skills in an abstention.
        """

        payload = self.to_dict()
        references: set[str] = set()

        def visit(value: Any, prefix: str) -> None:
            references.add(prefix)
            if isinstance(value, Mapping):
                for key, child in value.items():
                    visit(child, f"{prefix}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    visit(child, f"{prefix}.{index}")

        for root in EVIDENCE_ROOTS:
            visit(payload[root], root)
        return references

    def evidence_value(self, reference: str) -> Any:
        """The value at an evidence path, such as "exclusions.0.stage"; KeyError if absent."""

        if reference.split(".", 1)[0] not in EVIDENCE_ROOTS:
            raise KeyError(reference)
        value: Any = self.to_dict()
        for part in reference.split("."):
            if isinstance(value, Mapping) and part in value:
                value = value[part]
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                raise KeyError(reference)
        return value

    def claimable_facts(self) -> dict[str, tuple[str, Any]]:
        """Every path a claim may cite, with its claim type and value, in request order."""

        facts = {}
        for reference in sorted(self.valid_evidence_refs()):
            claim_type = claim_type_of(reference)
            if claim_type is not None:
                facts[reference] = (claim_type, self.evidence_value(reference))
        return facts


# The request sections a response may cite.
EVIDENCE_ROOTS = (
    "subrep_decision",
    "task_context",
    "objective_order",
    "risk_budget",
    "admitted_skills",
    "exclusions",
    "explanation_evidence",
    "audit_only_certificate_evidence",
    "evidence_label",
)


def _exact_fields(payload: Any, fields: set[str], name: str) -> Mapping[str, Any]:
    """Require an object with exactly these fields, naming what is missing or extra."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} items must be objects")
    missing = fields.difference(payload)
    unexpected = set(payload).difference(fields)
    if missing or unexpected:
        raise ValueError(
            f"{name} items must have exactly the fields {sorted(fields)}"
            + (f"; missing {sorted(missing)}" if missing else "")
            + (f"; unexpected {sorted(unexpected)}" if unexpected else "")
        )
    return payload


@dataclass(frozen=True)
class Claim:
    """One fact Omega states about the request, with the evidence path of that fact.

    SubRep checks the value at the path and writes the explanation from verified claims,
    so a claim carries no prose of its own.
    """

    type: str
    evidence_ref: str
    value: Any

    @classmethod
    def from_dict(cls, payload: Any) -> "Claim":
        item = _exact_fields(payload, {"type", "evidence_ref", "value"}, "claims")
        return cls(item["type"], item["evidence_ref"], item["value"])

    def __post_init__(self) -> None:
        claim_type = _non_empty(self.type, "claim type").lower()
        if claim_type not in CLAIM_TYPES:
            raise ValueError(f"claim type must be one of {sorted(CLAIM_TYPES)}, got {claim_type!r}")
        object.__setattr__(self, "type", claim_type)
        reference = _non_empty(self.evidence_ref, "claim evidence_ref")
        object.__setattr__(self, "evidence_ref", canonical_evidence_ref(reference))
        if isinstance(self.value, (Mapping, list, tuple)):
            raise ValueError("claim value must be one value: a number, text, true, false, or null")
        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("claim value must be finite")


@dataclass(frozen=True)
class AdvisoryConcern:
    """A non-binding concern about the decision, given as a code and the evidence it rests on.

    SubRep checks that the evidence supports the code and writes the sentence itself.
    """

    code: str
    evidence_refs: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: Any) -> "AdvisoryConcern":
        item = _exact_fields(payload, {"code", "evidence_refs"}, "advisory_concerns")
        return cls(item["code"], item["evidence_refs"])

    def __post_init__(self) -> None:
        code = _non_empty(self.code, "advisory concern code").upper()
        if code not in VALID_ADVISORY_CONCERN_CODES:
            raise ValueError(
                f"advisory concern code must be one of {sorted(VALID_ADVISORY_CONCERN_CODES)}"
            )
        object.__setattr__(self, "code", code)
        # A string would be read one character at a time, giving a misleading error.
        if not isinstance(self.evidence_refs, (list, tuple)):
            raise ValueError("advisory concern evidence_refs must be an array")
        refs = tuple(
            canonical_evidence_ref(_non_empty(item, "advisory evidence reference"))
            for item in self.evidence_refs
        )
        if not refs:
            raise ValueError("advisory concerns require at least one evidence reference")
        if len(refs) != len(set(refs)):
            raise ValueError("advisory evidence references must not contain duplicates")
        object.__setattr__(self, "evidence_refs", refs)


@dataclass(frozen=True)
class OmegaRecommendationResponse:
    """Structured response expected from OmegaClaw: the decision echo and claims, no prose."""

    request_id: str
    selected_skill_id: str | None
    abstain: bool
    selection_basis: str
    claims: tuple[Claim, ...]
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
            "claims",
            "advisory_concerns",
            "mode",
        }
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"response is missing required fields: {sorted(missing)}")
        unexpected = set(payload).difference(required)
        if unexpected:
            raise ValueError(f"response contains unexpected fields: {sorted(unexpected)}")
        for name in ("claims", "advisory_concerns"):
            if not isinstance(payload[name], (list, tuple)):
                raise ValueError(f"{name} must be an array")
        return cls(
            schema_version=payload["schema_version"],
            request_id=payload["request_id"],
            selected_skill_id=payload["selected_skill_id"],
            abstain=payload["abstain"],
            selection_basis=payload["selection_basis"],
            claims=tuple(
                item if isinstance(item, Claim) else Claim.from_dict(item)
                for item in payload["claims"]
            ),
            advisory_concerns=tuple(
                item if isinstance(item, AdvisoryConcern) else AdvisoryConcern.from_dict(item)
                for item in payload["advisory_concerns"]
            ),
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
        if self.abstain:
            if self.selected_skill_id is not None:
                raise ValueError("selected_skill_id must be null when abstain is true")
        else:
            object.__setattr__(
                self,
                "selected_skill_id",
                _non_empty(self.selected_skill_id, "selected_skill_id"),
            )
        claims = tuple(self.claims)
        if not all(isinstance(item, Claim) for item in claims):
            raise ValueError("claims must contain Claim values")
        if len(claims) > MAX_CLAIMS:
            raise ValueError(f"claims must contain at most {MAX_CLAIMS} items")
        object.__setattr__(self, "claims", claims)
        concerns = tuple(self.advisory_concerns)
        if not all(isinstance(item, AdvisoryConcern) for item in concerns):
            raise ValueError("advisory_concerns must contain AdvisoryConcern values")
        object.__setattr__(self, "advisory_concerns", concerns)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FORMAT_FINDING = "format"
FACTUAL_FINDING = "factual"
VALID_FINDING_CATEGORIES = {FORMAT_FINDING, FACTUAL_FINDING}


@dataclass(frozen=True)
class Finding:
    """One reason an Omega response fails the contract or misstates the evidence.

    ``format`` findings concern the response shape and citations; ``factual``
    findings concern whether what the response says matches the request.
    """

    code: str
    category: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _non_empty(self.code, "finding code").upper())
        if self.category not in VALID_FINDING_CATEGORIES:
            raise ValueError(f"finding category must be one of {VALID_FINDING_CATEGORIES}")
        object.__setattr__(self, "message", _non_empty(self.message, "finding message"))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RecommendationOutcome:
    """Adapter outcome; it never represents skill execution.

    SubRep's decision is always present, whatever happened to the explanation: ``status``
    says only whether an explanation was accepted, and ``explanation`` is None otherwise.
    """

    request_id: str
    status: str
    selected_skill_id: str | None = None
    explanation: str | None = None
    error: str | None = None
    raw_response: str | None = field(default=None, repr=False)
    response: Mapping[str, Any] | None = None
    attempts: tuple[Mapping[str, Any], ...] = ()
    abstain: bool | None = None
    subrep_decision: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _non_empty(self.request_id, "request_id"))
        if self.status not in VALID_OUTCOME_STATUSES:
            raise ValueError(f"status must be one of {VALID_OUTCOME_STATUSES}")
        attempts = tuple(dict(item) for item in self.attempts)
        _json_safe(attempts, "attempts")
        object.__setattr__(self, "attempts", attempts)
        if self.subrep_decision is not None:
            decision = dict(self.subrep_decision)
            _json_safe(decision, "subrep_decision")
            object.__setattr__(self, "subrep_decision", decision)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

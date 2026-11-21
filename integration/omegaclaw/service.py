"""SubRep-owned orchestration before the Omega explanation boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from library.skill_library import SkillLibrary
from library.skill_metadata import SkillEntry

from .adapter import OmegaRecommendationAdapter
from .contracts import (
    CERTIFICATION_STAGE,
    OPERATOR_STAGE,
    ObjectiveWeight,
    RecommendationOutcome,
    RecommendationRequest,
    RiskBudget,
    SkillEvidence,
    SkillExclusion,
)

CALLER_EXCLUSION_STAGES = frozenset({OPERATOR_STAGE, CERTIFICATION_STAGE})


class SubRepOmegaRecommendationService:
    """Create an authoritative decision and request an explanation without execution."""

    def __init__(self, adapter: OmegaRecommendationAdapter) -> None:
        self.adapter = adapter

    def recommend_from_library(
        self,
        *,
        task_context: Mapping[str, Any],
        objective_weights: Mapping[str, float],
        objective_order: Sequence[str],
        risk_budget: RiskBudget,
        skill_library: SkillLibrary,
        exclusions: Sequence[SkillExclusion] = (),
        support_directions: np.ndarray | None = None,
        support_values: np.ndarray | None = None,
        evidence_label: str = "OBSERVED",
        request_id: str | None = None,
    ) -> RecommendationOutcome:
        request = self.build_request_from_library(
            task_context=task_context,
            objective_weights=objective_weights,
            objective_order=objective_order,
            risk_budget=risk_budget,
            skill_library=skill_library,
            exclusions=exclusions,
            support_directions=support_directions,
            support_values=support_values,
            evidence_label=evidence_label,
            request_id=request_id,
        )
        return self.adapter.recommend(request)

    def build_request_from_library(
        self,
        *,
        task_context: Mapping[str, Any],
        objective_weights: Mapping[str, float],
        objective_order: Sequence[str],
        risk_budget: RiskBudget,
        skill_library: SkillLibrary,
        exclusions: Sequence[SkillExclusion] = (),
        support_directions: np.ndarray | None = None,
        support_values: np.ndarray | None = None,
        evidence_label: str = "OBSERVED",
        request_id: str | None = None,
    ) -> RecommendationRequest:
        """Build the exact validated snapshot used by every recommendation path."""

        ordered_ids = tuple(objective_order)
        if len(ordered_ids) < 2:
            raise ValueError("objective_order must contain at least two objective IDs")
        if any(not isinstance(name, str) or not name.strip() for name in ordered_ids):
            raise ValueError("objective_order must contain non-empty string IDs")
        if len(ordered_ids) != len(set(ordered_ids)):
            raise ValueError("objective_order must contain unique objective IDs")
        if set(ordered_ids) != set(objective_weights):
            raise ValueError("objective_order must contain exactly the objective_weights keys")

        objectives = tuple(
            ObjectiveWeight(objective_id=name, weight=objective_weights[name])
            for name in ordered_ids
        )
        weights_array = np.asarray([item.weight for item in objectives], dtype=np.float64)
        explicit_exclusions = {item.skill_id: item for item in exclusions}
        _check_explicit_exclusions(explicit_exclusions.values(), skill_library)
        runtime_ids = {
            entry.skill_id
            for entry in skill_library.query_admissible(
                current_weight=weights_array,
                support_directions=support_directions,
                support_values=support_values,
            )
        }

        final_exclusions = dict(explicit_exclusions)
        candidates: list[SkillEvidence] = []
        objective_ids = tuple(item.objective_id for item in objectives)
        for entry in skill_library.get_admitted_skills():
            if entry.skill_id in explicit_exclusions:
                continue
            exclusion = _library_exclusion(
                entry=entry,
                runtime_admissible=entry.skill_id in runtime_ids,
                risk_budget=risk_budget,
                objective_count=len(objectives),
                evidence_label=evidence_label,
            )
            if exclusion is not None:
                final_exclusions[entry.skill_id] = exclusion
                continue
            candidates.append(
                _skill_evidence(
                    entry=entry,
                    objective_ids=objective_ids,
                    weights=weights_array,
                    evidence_label=evidence_label,
                )
            )

        return RecommendationRequest(
            request_id=request_id or f"SR{uuid4().hex[:14].upper()}",
            created_at=datetime.now(timezone.utc).isoformat(),
            task_context=dict(task_context),
            objective_weights=objectives,
            risk_budget=risk_budget,
            admitted_skills=tuple(sorted(candidates, key=lambda item: item.skill_id)),
            exclusions=tuple(sorted(final_exclusions.values(), key=lambda item: item.skill_id)),
            evidence_label=evidence_label,
        )


def _check_explicit_exclusions(
    exclusions: Iterable[SkillExclusion],
    skill_library: SkillLibrary,
) -> None:
    """Allow callers only the stages SubRep cannot compute from the library itself."""

    for exclusion in exclusions:
        if exclusion.stage not in CALLER_EXCLUSION_STAGES:
            raise ValueError(
                f"exclusion {exclusion.skill_id!r} uses {exclusion.reason_code!r}, which "
                "SubRep computes itself; callers may only supply operator exclusions "
                "and certification failures"
            )
        if (
            exclusion.stage == CERTIFICATION_STAGE
            and skill_library.get_skill(exclusion.skill_id) is not None
        ):
            raise ValueError(
                f"exclusion {exclusion.skill_id!r} reports a certification failure, but "
                "the skill library holds a certificate for it"
            )


def _library_exclusion(
    *,
    entry: SkillEntry,
    runtime_admissible: bool,
    risk_budget: RiskBudget,
    objective_count: int,
    evidence_label: str,
) -> SkillExclusion | None:
    # The dimension check comes first: the runtime query also drops a skill whose
    # deltas do not fit the weights, and that must not read as an admissibility result.
    if len(entry.delta_n) != objective_count:
        return SkillExclusion(
            skill_id=entry.skill_id,
            reason_code="OBJECTIVE_DIMENSION_MISMATCH",
            reason=(
                f"Skill has {len(entry.delta_n)} objective deltas but the request has "
                f"{objective_count} objectives."
            ),
            evidence_label=evidence_label,
        )
    if not runtime_admissible:
        return SkillExclusion(
            skill_id=entry.skill_id,
            reason_code="NOT_RUNTIME_ADMISSIBLE",
            reason=(
                "The SubRep runtime admissibility query did not admit this skill for the "
                "current objective weights."
            ),
            evidence_label=evidence_label,
        )
    if (
        risk_budget.max_certificate_epsilon is not None
        and entry.epsilon > risk_budget.max_certificate_epsilon
    ):
        return SkillExclusion(
            skill_id=entry.skill_id,
            reason_code="RISK_BUDGET_EXCEEDED",
            reason=(
                f"Certificate epsilon {entry.epsilon:.6g} exceeds the local maximum "
                f"{risk_budget.max_certificate_epsilon:.6g}."
            ),
            evidence_label=evidence_label,
        )
    if (
        risk_budget.minimum_admission_margin is not None
        and entry.admission_margin < risk_budget.minimum_admission_margin
    ):
        return SkillExclusion(
            skill_id=entry.skill_id,
            reason_code="INSUFFICIENT_ADMISSION_MARGIN",
            reason=(
                f"Admission margin {entry.admission_margin:.6g} is below the local minimum "
                f"{risk_budget.minimum_admission_margin:.6g}."
            ),
            evidence_label=evidence_label,
        )
    return None


def _skill_evidence(
    *,
    entry: SkillEntry,
    objective_ids: tuple[str, ...],
    weights: np.ndarray,
    evidence_label: str,
) -> SkillEvidence:
    deltas = {
        objective_id: float(delta)
        for objective_id, delta in zip(objective_ids, entry.delta_n)
    }
    score = float(entry.delta_r + np.dot(weights, np.asarray(entry.delta_n, dtype=np.float64)))
    return SkillEvidence(
        skill_id=entry.skill_id,
        score=score,
        delta_r=float(entry.delta_r),
        objective_deltas=deltas,
        gate_type=entry.gate_type,
        admission_margin=float(entry.admission_margin),
        epsilon=float(entry.epsilon),
        weight_region_type=entry.weight_region_type,
        evidence_label=evidence_label,
    )

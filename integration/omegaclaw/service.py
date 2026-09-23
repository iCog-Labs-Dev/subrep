"""SubRep-owned orchestration before the Omega recommendation boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from library.skill_library import SkillLibrary
from library.skill_metadata import SkillEntry

from .adapter import OmegaRecommendationAdapter
from .contracts import (
    ObjectiveWeight,
    RecommendationOutcome,
    RecommendationRequest,
    RiskBudget,
    SkillEvidence,
    SkillExclusion,
)


class SubRepOmegaRecommendationService:
    """Create an eligible snapshot and request advice without executing a skill."""

    def __init__(self, adapter: OmegaRecommendationAdapter) -> None:
        self.adapter = adapter

    def recommend_from_library(
        self,
        *,
        task_context: Mapping[str, Any],
        objective_weights: Mapping[str, float],
        risk_budget: RiskBudget,
        skill_library: SkillLibrary,
        exclusions: Sequence[SkillExclusion] = (),
        support_directions: np.ndarray | None = None,
        support_values: np.ndarray | None = None,
        evidence_label: str = "OBSERVED",
        request_id: str | None = None,
    ) -> RecommendationOutcome:
        objectives = tuple(
            ObjectiveWeight(objective_id=name, weight=weight)
            for name, weight in objective_weights.items()
        )
        weights_array = np.asarray([item.weight for item in objectives], dtype=np.float64)
        runtime_admissible = skill_library.query_admissible(
            current_weight=weights_array,
            support_directions=support_directions,
            support_values=support_values,
        )

        explicit_exclusions = {item.skill_id: item for item in exclusions}
        final_exclusions = dict(explicit_exclusions)
        runtime_ids = {entry.skill_id for entry in runtime_admissible}
        for entry in skill_library.get_admitted_skills():
            if entry.skill_id not in runtime_ids and entry.skill_id not in final_exclusions:
                final_exclusions[entry.skill_id] = SkillExclusion(
                    skill_id=entry.skill_id,
                    reason_code="NOT_RUNTIME_ADMISSIBLE",
                    reason="SubRep's runtime admissibility query excluded this skill.",
                    evidence_label=evidence_label,
                )

        candidates: list[SkillEvidence] = []
        objective_ids = tuple(item.objective_id for item in objectives)
        for entry in runtime_admissible:
            if entry.skill_id in explicit_exclusions:
                continue
            exclusion = _risk_or_shape_exclusion(
                entry=entry,
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

        request = RecommendationRequest(
            request_id=request_id or f"subrep-{uuid4().hex}",
            created_at=datetime.now(timezone.utc).isoformat(),
            task_context=dict(task_context),
            objective_weights=objectives,
            risk_budget=risk_budget,
            admitted_skills=tuple(sorted(candidates, key=lambda item: item.skill_id)),
            exclusions=tuple(sorted(final_exclusions.values(), key=lambda item: item.skill_id)),
            allow_abstain=True,
            evidence_label=evidence_label,
        )
        return self.adapter.recommend(request)


def _risk_or_shape_exclusion(
    *,
    entry: SkillEntry,
    risk_budget: RiskBudget,
    objective_count: int,
    evidence_label: str,
) -> SkillExclusion | None:
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

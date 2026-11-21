"""Predefined, clearly synthetic scenarios for repeatable review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from certification.certificate_schema import Certificate
from library.skill_library import SkillLibrary

from .contracts import RiskBudget, SkillExclusion

STRICT_RISK_BUDGET = RiskBudget(max_certificate_epsilon=0.0, minimum_admission_margin=0.0)


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    description: str
    task_context: dict
    objective_weights: dict[str, float]
    objective_order: tuple[str, ...]
    risk_budget: RiskBudget
    skill_library: SkillLibrary
    expected_selected_skill_id: str | None
    exclusions: tuple[SkillExclusion, ...] = ()
    support_values: tuple[float, ...] | None = None

    def service_arguments(self) -> dict[str, Any]:
        """Keyword arguments for ``SubRepOmegaRecommendationService`` request building."""

        support = None if self.support_values is None else np.asarray(self.support_values)
        return {
            "task_context": self.task_context,
            "objective_weights": self.objective_weights,
            "objective_order": self.objective_order,
            "risk_budget": self.risk_budget,
            "skill_library": self.skill_library,
            "exclusions": self.exclusions,
            "support_directions": None if support is None else np.eye(support.size),
            "support_values": support,
        }


def predefined_scenarios() -> tuple[SyntheticScenario, ...]:
    """Return the acceptance scenarios, all built on synthetic evidence only."""

    return (
        SyntheticScenario(
            name="clear_preferred_skill",
            description="Safety is dominant, so the safety-focused skill has a clear lead.",
            task_context={"scenario": "clear preferred skill", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.8, "efficiency": 0.2},
            objective_order=("safety", "efficiency"),
            risk_budget=STRICT_RISK_BUDGET,
            skill_library=_library("skill_safety", "skill_efficiency"),
            expected_selected_skill_id="skill_safety",
        ),
        SyntheticScenario(
            name="changed_task_priorities",
            description="Efficiency becomes dominant and changes the preferred skill.",
            task_context={"scenario": "changed task priorities", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.2, "efficiency": 0.8},
            objective_order=("safety", "efficiency"),
            risk_budget=STRICT_RISK_BUDGET,
            skill_library=_library("skill_safety", "skill_efficiency"),
            expected_selected_skill_id="skill_efficiency",
        ),
        SyntheticScenario(
            name="excluded_skill",
            description="The otherwise preferred efficiency skill is excluded by the operator.",
            task_context={"scenario": "excluded skill", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.1, "efficiency": 0.9},
            objective_order=("safety", "efficiency"),
            risk_budget=STRICT_RISK_BUDGET,
            skill_library=_library("skill_safety", "skill_efficiency"),
            expected_selected_skill_id="skill_safety",
            exclusions=(_operator_exclusion("skill_efficiency"),),
        ),
        SyntheticScenario(
            name="no_admissible_options",
            description="No skill is admitted, so the backend must abstain.",
            task_context={"scenario": "no admissible options", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.5, "efficiency": 0.5},
            objective_order=("safety", "efficiency"),
            risk_budget=STRICT_RISK_BUDGET,
            skill_library=SkillLibrary(),
            expected_selected_skill_id=None,
            exclusions=(_certification_failure("rejected_candidate"),),
        ),
        SyntheticScenario(
            name="risk_budget_exclusion",
            description=(
                "The highest-scoring skill is certified with a PDS epsilon, which the "
                "zero-epsilon risk budget excludes."
            ),
            task_context={"scenario": "risk budget exclusion", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.1, "efficiency": 0.9},
            objective_order=("safety", "efficiency"),
            risk_budget=STRICT_RISK_BUDGET,
            skill_library=_library("skill_safety", "skill_efficiency", "skill_fast_pds"),
            expected_selected_skill_id="skill_efficiency",
        ),
        SyntheticScenario(
            name="runtime_inadmissible",
            description=(
                "One skill is certified only for low efficiency weights, so the current "
                "weight region does not admit it at runtime."
            ),
            task_context={"scenario": "runtime inadmissible", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.3, "efficiency": 0.7},
            objective_order=("safety", "efficiency"),
            risk_budget=STRICT_RISK_BUDGET,
            skill_library=_library("skill_safety", "skill_efficiency", "skill_regional"),
            expected_selected_skill_id="skill_efficiency",
            support_values=(0.4, 1.0),
        ),
        SyntheticScenario(
            name="mixed_exclusions",
            description="One request excludes skills at every stage, leaving one admitted skill.",
            task_context={"scenario": "mixed exclusions", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.6, "efficiency": 0.4},
            objective_order=("safety", "efficiency"),
            risk_budget=STRICT_RISK_BUDGET,
            skill_library=_library(
                "skill_safety",
                "skill_efficiency",
                "skill_fast_pds",
                "skill_regional",
                "skill_three_objective",
            ),
            expected_selected_skill_id="skill_safety",
            exclusions=(
                _operator_exclusion("skill_efficiency"),
                _certification_failure("rejected_candidate"),
            ),
            support_values=(0.7, 1.0),
        ),
        SyntheticScenario(
            name="abstention_mixed_reasons",
            description=(
                "Every option is excluded, by the operator, the risk budget, or failed "
                "certification, so SubRep abstains."
            ),
            task_context={"scenario": "abstention with mixed reasons", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.5, "efficiency": 0.5},
            objective_order=("safety", "efficiency"),
            risk_budget=STRICT_RISK_BUDGET,
            skill_library=_library("skill_efficiency", "skill_fast_pds"),
            expected_selected_skill_id=None,
            exclusions=(
                _operator_exclusion("skill_efficiency"),
                _certification_failure("rejected_candidate"),
            ),
        ),
    )


def _operator_exclusion(skill_id: str) -> SkillExclusion:
    return SkillExclusion(
        skill_id=skill_id,
        reason_code="OPERATOR_EXCLUSION",
        reason="An operator removed this skill for the synthetic review scenario.",
        evidence_label="SYNTHETIC",
    )


def _certification_failure(skill_id: str) -> SkillExclusion:
    return SkillExclusion(
        skill_id=skill_id,
        reason_code="GATE_FAILED",
        reason="This synthetic candidate failed both the CDS and the PDS gate.",
        evidence_label="SYNTHETIC",
    )


@dataclass(frozen=True)
class _SyntheticSkill:
    gate_type: str
    delta_r: float
    delta_n: tuple[float, ...]
    admission_margin: float
    epsilon: float = 0.0
    certified_support_values: tuple[float, ...] | None = None


_SKILLS = {
    "skill_safety": _SyntheticSkill("CDS", 0.2, (0.8, -0.1), admission_margin=0.1),
    "skill_efficiency": _SyntheticSkill("CDS", 0.2, (-0.1, 0.8), admission_margin=0.1),
    # Admitted only under PDS: 0.15 - 0.17 = -0.02 >= -0.05.
    "skill_fast_pds": _SyntheticSkill(
        "PDS", 0.15, (-0.17, 1.0), admission_margin=0.03, epsilon=0.05
    ),
    # Certified for efficiency weight <= 0.3 only, where the worst case is -0.3.
    "skill_regional": _SyntheticSkill(
        "CDS", 0.1, (0.6, -0.4), admission_margin=0.4, certified_support_values=(1.0, 0.3)
    ),
    "skill_three_objective": _SyntheticSkill("CDS", 0.3, (0.2, 0.2, 0.2), admission_margin=0.5),
}


def _library(*skill_ids: str) -> SkillLibrary:
    library = SkillLibrary()
    for skill_id in skill_ids:
        skill = _SKILLS[skill_id]
        support_values = skill.certified_support_values
        region: dict[str, Any] = {}
        if support_values is not None:
            size = len(support_values)
            region = {
                "weight_region_type": "MDN_WX",
                "certification_context": (0.0,) * size,
                "mdn_alpha": (1.0,) * size,
                "wx_support_directions": tuple(
                    tuple(1.0 if row == column else 0.0 for column in range(size))
                    for row in range(size)
                ),
                "wx_support_values": support_values,
            }
        certificate = Certificate(
            skill_id=skill_id,
            gate_type=skill.gate_type,
            delta_r=skill.delta_r,
            delta_n=skill.delta_n,
            admission_margin=skill.admission_margin,
            epsilon=skill.epsilon,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=2026,
            gamma=1.0,
            baseline_id="synthetic_baseline",
            environment="synthetic_recommendation_review",
            episode_length=1,
            version="1.0",
            **region,
        )

        def recommendation_only_policy(_observation, sid=skill_id):
            raise AssertionError(f"recommendation-only flow executed {sid}")

        added = library.add_skill(skill_id, certificate, recommendation_only_policy, **region)
        if not added:
            raise RuntimeError(f"could not construct synthetic skill {skill_id}")
    return library

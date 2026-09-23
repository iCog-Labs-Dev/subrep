"""Predefined, clearly synthetic scenarios for repeatable review."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from certification.certificate_schema import Certificate
from library.skill_library import SkillLibrary

from .contracts import RiskBudget, SkillExclusion


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    description: str
    task_context: dict
    objective_weights: dict[str, float]
    risk_budget: RiskBudget
    skill_library: SkillLibrary
    exclusions: tuple[SkillExclusion, ...] = ()


def predefined_scenarios() -> tuple[SyntheticScenario, ...]:
    """Return the four acceptance scenarios using synthetic evidence only."""

    return (
        SyntheticScenario(
            name="clear_preferred_skill",
            description="Safety is dominant, so the safety-focused skill has a clear lead.",
            task_context={"scenario": "clear preferred skill", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.8, "efficiency": 0.2},
            risk_budget=RiskBudget(max_certificate_epsilon=0.0, minimum_admission_margin=0.0),
            skill_library=_balanced_library(),
        ),
        SyntheticScenario(
            name="changed_task_priorities",
            description="Efficiency becomes dominant and changes the preferred skill.",
            task_context={"scenario": "changed task priorities", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.2, "efficiency": 0.8},
            risk_budget=RiskBudget(max_certificate_epsilon=0.0, minimum_admission_margin=0.0),
            skill_library=_balanced_library(),
        ),
        SyntheticScenario(
            name="excluded_skill",
            description="The otherwise preferred efficiency skill is explicitly excluded.",
            task_context={"scenario": "excluded skill", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.1, "efficiency": 0.9},
            risk_budget=RiskBudget(max_certificate_epsilon=0.0, minimum_admission_margin=0.0),
            skill_library=_balanced_library(),
            exclusions=(
                SkillExclusion(
                    skill_id="skill_efficiency",
                    reason_code="OPERATOR_EXCLUSION",
                    reason="Synthetic scenario exclusion used to verify hard filtering.",
                    evidence_label="SYNTHETIC",
                ),
            ),
        ),
        SyntheticScenario(
            name="no_admissible_options",
            description="No skill is admitted, so the backend must abstain.",
            task_context={"scenario": "no admissible options", "evidence": "SYNTHETIC"},
            objective_weights={"safety": 0.5, "efficiency": 0.5},
            risk_budget=RiskBudget(max_certificate_epsilon=0.0, minimum_admission_margin=0.0),
            skill_library=SkillLibrary(),
            exclusions=(
                SkillExclusion(
                    skill_id="rejected_candidate",
                    reason_code="GATE_FAILED",
                    reason="Synthetic candidate failed SubRep certification.",
                    evidence_label="SYNTHETIC",
                ),
            ),
        ),
    )


def _balanced_library() -> SkillLibrary:
    library = SkillLibrary()
    entries = (
        ("skill_safety", 0.2, (0.8, -0.1)),
        ("skill_efficiency", 0.2, (-0.1, 0.8)),
    )
    for skill_id, delta_r, delta_n in entries:
        certificate = Certificate(
            skill_id=skill_id,
            gate_type="CDS",
            delta_r=delta_r,
            delta_n=delta_n,
            admission_margin=delta_r + min(delta_n),
            epsilon=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=2026,
            gamma=1.0,
            baseline_id="synthetic_baseline",
            environment="synthetic_recommendation_review",
            episode_length=1,
            version="1.0",
        )

        def recommendation_only_policy(_observation, sid=skill_id):
            raise AssertionError(f"recommendation-only flow executed {sid}")

        added = library.add_skill(skill_id, certificate, recommendation_only_policy)
        if not added:
            raise RuntimeError(f"could not construct synthetic skill {skill_id}")
    return library

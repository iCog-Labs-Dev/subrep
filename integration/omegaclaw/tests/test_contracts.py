from __future__ import annotations

from datetime import datetime, timezone

import pytest

from integration.omegaclaw.contracts import (
    ObjectiveWeight,
    RecommendationRequest,
    RiskBudget,
    SkillEvidence,
)


def _skill(**changes) -> SkillEvidence:
    values = {
        "skill_id": "safe_skill",
        "score": 1.5,
        "delta_r": 1.0,
        "objective_deltas": {"safety": 0.6, "fuel": 0.2},
        "gate_type": "CDS",
        "admission_margin": 1.2,
        "epsilon": 0.0,
        "weight_region_type": "FULL_SIMPLEX",
        "evidence_label": "OBSERVED",
    }
    values.update(changes)
    return SkillEvidence(**values)


def _request(skill: SkillEvidence | None = None, **changes) -> RecommendationRequest:
    values = {
        "request_id": "request-1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_context": {"task": "land safely"},
        "objective_weights": (
            ObjectiveWeight("safety", 0.75),
            ObjectiveWeight("fuel", 0.25),
        ),
        "risk_budget": RiskBudget(max_certificate_epsilon=0.0),
        "admitted_skills": (skill or _skill(),),
    }
    values.update(changes)
    return RecommendationRequest(**values)


def test_request_rejects_candidate_that_exceeds_risk_budget():
    skill = _skill(
        gate_type="PDS",
        epsilon=0.2,
        score=1.5,
    )

    with pytest.raises(ValueError, match="max_certificate_epsilon"):
        _request(skill)


def test_request_rejects_mixed_synthetic_and_observed_labels():
    with pytest.raises(ValueError, match="evidence_label"):
        _request(evidence_label="SYNTHETIC")


def test_request_requires_timezone_aware_timestamp():
    with pytest.raises(ValueError, match="timezone"):
        _request(created_at="2026-09-20T12:00:00")


def test_skill_rejects_unknown_gate_and_weight_region():
    with pytest.raises(ValueError, match="gate_type"):
        _skill(gate_type="UNKNOWN")
    with pytest.raises(ValueError, match="weight_region_type"):
        _skill(weight_region_type="UNKNOWN")


def test_risk_budget_rejects_negative_minimum_margin():
    with pytest.raises(ValueError, match="minimum_admission_margin"):
        RiskBudget(minimum_admission_margin=-0.1)

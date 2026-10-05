from __future__ import annotations

from datetime import datetime, timezone

import pytest

from integration.omegaclaw.contracts import (
    ObjectiveWeight,
    RecommendationRequest,
    RiskBudget,
    SkillEvidence,
    SubRepDecision,
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


def test_expected_selection_uses_lexical_tie_breaker():
    tied = _skill(skill_id="zeta_skill")
    lexical_winner = _skill(skill_id="alpha_skill")
    request = _request(admitted_skills=(tied, lexical_winner))

    assert request.subrep_decision.selected_skill_id == "alpha_skill"


def test_serialized_request_separates_selection_from_audit_evidence():
    payload = _request().to_dict()

    assert payload["mode"] == "EXPLANATION_ONLY"
    assert payload["objective_order"] == ["safety", "fuel"]
    assert payload["subrep_decision"]["selected_skill_id"] == "safe_skill"
    assert payload["subrep_decision"]["selected_score"] == 1.5
    assert payload["admitted_skills"]["safe_skill"] == {
        "skill_id": "safe_skill",
        "final_selection_score": 1.5,
    }
    assert "certificate_admission_margin" not in payload["admitted_skills"]["safe_skill"]
    assert (
        payload["audit_only_certificate_evidence"]["skills"]["safe_skill"]
        ["certificate_admission_margin"]
        == 1.2
    )


def test_subrep_decision_precomputes_runner_up_and_gap():
    runner_up = _skill(skill_id="runner_skill", score=1.0, delta_r=0.5)
    request = _request(admitted_skills=(_skill(), runner_up))

    assert request.subrep_decision.selected_skill_id == "safe_skill"
    assert request.subrep_decision.runner_up_skill_id == "runner_skill"
    assert request.subrep_decision.runner_up_score == 1.0
    assert request.subrep_decision.score_gap == 0.5


def test_valid_evidence_refs_include_decision_and_objective_paths():
    refs = _request().valid_evidence_refs()

    assert "subrep_decision.selected_score" in refs
    assert "explanation_evidence.objective_weights.safety" in refs
    assert "explanation_evidence.skills.safe_skill.objective_deltas.safety" in refs


def test_request_rejects_caller_supplied_decision_that_changes_winner():
    tampered_decision = SubRepDecision(
        selected_skill_id="other_skill",
        selected_score=1.5,
        runner_up_skill_id=None,
        runner_up_score=None,
        score_gap=None,
        abstain=False,
    )

    with pytest.raises(ValueError, match="does not match"):
        _request(subrep_decision=tampered_decision)


def test_request_copies_and_freezes_nested_evidence():
    task_context = {"mission": {"phases": ["approach", "land"]}}
    objective_deltas = {"safety": 0.6, "fuel": 0.2}
    skill = _skill(objective_deltas=objective_deltas)
    request = _request(skill=skill, task_context=task_context)

    task_context["mission"]["phases"].append("mutated")
    objective_deltas["safety"] = -10.0

    assert request.to_dict()["task_context"] == {
        "mission": {"phases": ["approach", "land"]}
    }
    assert request.admitted_skills[0].objective_deltas["safety"] == 0.6
    with pytest.raises(TypeError):
        request.task_context["mission"] = {}
    with pytest.raises(TypeError):
        request.admitted_skills[0].objective_deltas["safety"] = -10.0

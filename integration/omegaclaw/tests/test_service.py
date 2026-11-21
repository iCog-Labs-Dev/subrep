from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from certification.certificate_schema import Certificate
from integration.omegaclaw.adapter import OmegaRecommendationAdapter
from integration.omegaclaw.audit import JsonlAuditStore
from integration.omegaclaw.contracts import RiskBudget, SkillExclusion
from integration.omegaclaw.scenarios import predefined_scenarios
from integration.omegaclaw.service import SubRepOmegaRecommendationService
from integration.omegaclaw.synthetic_backend import SyntheticOmegaBackend
from library.skill_library import SkillLibrary


def test_risk_budget_filters_skill_before_backend_request(tmp_path):
    certificate = Certificate(
        skill_id="bounded_tradeoff",
        gate_type="PDS",
        delta_r=0.2,
        delta_n=(-0.5, -0.5),
        admission_margin=0.2,
        epsilon=0.5,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=1,
        gamma=1.0,
        baseline_id="synthetic",
        environment="synthetic",
        episode_length=1,
        version="1.0",
    )
    library = SkillLibrary()
    assert library.add_skill("bounded_tradeoff", certificate, lambda _obs: None)
    audit_path = tmp_path / "risk.jsonl"
    service = SubRepOmegaRecommendationService(
        OmegaRecommendationAdapter(
            SyntheticOmegaBackend(),
            JsonlAuditStore(audit_path),
        )
    )

    outcome = service.recommend_from_library(
        task_context={"evidence": "SYNTHETIC"},
        objective_weights={"safety": 0.5, "efficiency": 0.5},
        objective_order=("safety", "efficiency"),
        risk_budget=RiskBudget(max_certificate_epsilon=0.1),
        skill_library=library,
        evidence_label="SYNTHETIC",
    )

    assert outcome.status == "abstained"
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    assert record["request"]["admitted_skills"] == {}
    assert record["request"]["subrep_decision"]["abstain"] is True
    assert record["request"]["subrep_decision"]["reason_code"] == "NO_ADMISSIBLE_SKILLS"
    assert record["request"]["exclusions"][0]["reason_code"] == "RISK_BUDGET_EXCEEDED"


def test_reordered_named_weights_preserve_scores_and_selection(tmp_path):
    scenario = predefined_scenarios()[0]
    service = SubRepOmegaRecommendationService(
        OmegaRecommendationAdapter(
            SyntheticOmegaBackend(),
            JsonlAuditStore(tmp_path / "order.jsonl"),
        )
    )
    common = {
        "task_context": scenario.task_context,
        "objective_order": scenario.objective_order,
        "risk_budget": scenario.risk_budget,
        "skill_library": scenario.skill_library,
        "evidence_label": "SYNTHETIC",
    }

    original = service.build_request_from_library(
        objective_weights={"safety": 0.8, "efficiency": 0.2},
        **common,
    )
    reordered = service.build_request_from_library(
        objective_weights={"efficiency": 0.2, "safety": 0.8},
        **common,
    )

    original_scores = {skill.skill_id: skill.score for skill in original.admitted_skills}
    reordered_scores = {skill.skill_id: skill.score for skill in reordered.admitted_skills}
    assert original_scores == reordered_scores
    assert original.subrep_decision == reordered.subrep_decision
    assert reordered.to_dict()["objective_order"] == ["safety", "efficiency"]


def test_objective_order_must_match_named_weights(tmp_path):
    scenario = predefined_scenarios()[0]
    service = SubRepOmegaRecommendationService(
        OmegaRecommendationAdapter(
            SyntheticOmegaBackend(),
            JsonlAuditStore(tmp_path / "order.jsonl"),
        )
    )

    with pytest.raises(ValueError, match="exactly"):
        service.build_request_from_library(
            task_context=scenario.task_context,
            objective_weights=scenario.objective_weights,
            objective_order=("safety", "fuel"),
            risk_budget=scenario.risk_budget,
            skill_library=scenario.skill_library,
            evidence_label="SYNTHETIC",
        )


def _certificate(skill_id, delta_r, delta_n, **changes) -> Certificate:
    values = {
        "skill_id": skill_id,
        "gate_type": "CDS",
        "delta_r": delta_r,
        "delta_n": delta_n,
        "admission_margin": 0.1,
        "epsilon": 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": 1,
        "gamma": 1.0,
        "baseline_id": "synthetic",
        "environment": "synthetic",
        "episode_length": 1,
        "version": "1.0",
    }
    values.update(changes)
    return Certificate(**values)


def _add_wx_skill(library, skill_id, delta_r, delta_n, support_values) -> None:
    size = len(delta_n)
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
    certificate = _certificate(skill_id, delta_r, delta_n, **region)
    assert library.add_skill(skill_id, certificate, lambda _obs: None, **region)


def _build(service, library, *, exclusions=(), support_values=(1.0, 1.0)):
    return service.build_request_from_library(
        task_context={"evidence": "SYNTHETIC"},
        objective_weights={"safety": 0.5, "efficiency": 0.5},
        objective_order=("safety", "efficiency"),
        risk_budget=RiskBudget(max_certificate_epsilon=0.0),
        skill_library=library,
        exclusions=exclusions,
        support_directions=np.eye(2),
        support_values=np.asarray(support_values),
        evidence_label="SYNTHETIC",
    )


def _service(tmp_path) -> SubRepOmegaRecommendationService:
    return SubRepOmegaRecommendationService(
        OmegaRecommendationAdapter(
            SyntheticOmegaBackend(),
            JsonlAuditStore(tmp_path / "stages.jsonl"),
        )
    )


def test_uncovered_weight_region_is_reported_as_runtime_inadmissible(tmp_path):
    library = SkillLibrary()
    # Certified while efficiency weight <= 0.3; the current region allows up to 1.0.
    _add_wx_skill(library, "narrow_region", 0.1, (0.6, -0.4), (1.0, 0.3))

    request = _build(_service(tmp_path), library, support_values=(0.4, 1.0))

    (exclusion,) = request.exclusions
    assert exclusion.reason_code == "NOT_RUNTIME_ADMISSIBLE"
    assert exclusion.stage == "RUNTIME_ADMISSIBILITY"
    assert "not a certification failure" in exclusion.stage_description


def test_wrong_objective_count_is_not_reported_as_runtime_inadmissible(tmp_path):
    library = SkillLibrary()
    _add_wx_skill(library, "three_objectives", 0.5, (0.2, 0.2, 0.2), (1.0, 1.0, 1.0))

    request = _build(_service(tmp_path), library)

    (exclusion,) = request.exclusions
    assert exclusion.reason_code == "OBJECTIVE_DIMENSION_MISMATCH"
    assert exclusion.stage == "OBJECTIVE_COMPATIBILITY"


def test_caller_cannot_report_certification_failure_for_a_certified_skill(tmp_path):
    library = SkillLibrary()
    assert library.add_skill("certified", _certificate("certified", 0.2, (0.1, 0.1)), lambda _obs: None)
    exclusion = SkillExclusion(
        skill_id="certified",
        reason_code="GATE_FAILED",
        reason="Claimed gate failure.",
        evidence_label="SYNTHETIC",
    )

    with pytest.raises(ValueError, match="holds a certificate"):
        _build(_service(tmp_path), library, exclusions=(exclusion,))


@pytest.mark.parametrize(
    "reason_code",
    [
        "NOT_RUNTIME_ADMISSIBLE",
        "RISK_BUDGET_EXCEEDED",
        "INSUFFICIENT_ADMISSION_MARGIN",
        "OBJECTIVE_DIMENSION_MISMATCH",
    ],
)
def test_caller_cannot_supply_a_stage_that_subrep_computes(tmp_path, reason_code):
    exclusion = SkillExclusion(
        skill_id="anything",
        reason_code=reason_code,
        reason="Caller supplied.",
        evidence_label="SYNTHETIC",
    )

    with pytest.raises(ValueError, match="SubRep computes itself"):
        _build(_service(tmp_path), SkillLibrary(), exclusions=(exclusion,))


def test_caller_may_report_certification_failure_for_an_unadmitted_candidate(tmp_path):
    exclusion = SkillExclusion(
        skill_id="rejected_candidate",
        reason_code="GATE_FAILED",
        reason="Failed the CDS and PDS gates.",
        evidence_label="SYNTHETIC",
    )

    request = _build(_service(tmp_path), SkillLibrary(), exclusions=(exclusion,))

    payload = request.to_dict()["exclusions"][0]
    assert payload["stage"] == "CERTIFICATION"
    assert payload["stage_description"].startswith("Failed the SubRep CDS/PDS")

from __future__ import annotations

import json

from integration.omegaclaw.adapter import OmegaRecommendationAdapter
from integration.omegaclaw.audit import JsonlAuditStore
from integration.omegaclaw.contracts import EXCLUSION_STAGES
from integration.omegaclaw.scenarios import predefined_scenarios
from integration.omegaclaw.service import SubRepOmegaRecommendationService
from integration.omegaclaw.synthetic_backend import SyntheticOmegaBackend


def test_predefined_scenarios_cover_acceptance_cases(tmp_path):
    audit_path = tmp_path / "scenarios.jsonl"
    service = SubRepOmegaRecommendationService(
        OmegaRecommendationAdapter(
            SyntheticOmegaBackend(),
            JsonlAuditStore(audit_path),
        )
    )
    outcomes = {}
    for scenario in predefined_scenarios():
        outcomes[scenario.name] = service.recommend_from_library(
            **scenario.service_arguments(),
            evidence_label="SYNTHETIC",
            request_id=f"test-{scenario.name}",
        )

    assert outcomes["clear_preferred_skill"].selected_skill_id == "skill_safety"
    assert outcomes["changed_task_priorities"].selected_skill_id == "skill_efficiency"
    assert outcomes["excluded_skill"].selected_skill_id == "skill_safety"
    assert outcomes["no_admissible_options"].status == "abstained"
    assert "final selection score (0.82)" in outcomes["clear_preferred_skill"].explanation
    assert outcomes["no_admissible_options"].explanation.startswith("SubRep abstained")

    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(predefined_scenarios())
    excluded_record = next(
        record for record in records if record["request"]["request_id"] == "test-excluded_skill"
    )
    assert excluded_record["request"]["exclusions"][0]["skill_id"] == "skill_efficiency"
    assert all(
        record["request"]["evidence_label"] == "SYNTHETIC" for record in records
    )


def _service(tmp_path) -> SubRepOmegaRecommendationService:
    return SubRepOmegaRecommendationService(
        OmegaRecommendationAdapter(SyntheticOmegaBackend(), JsonlAuditStore(tmp_path / "a.jsonl"))
    )


def test_every_scenario_reaches_its_expected_decision(tmp_path):
    service = _service(tmp_path)

    for scenario in predefined_scenarios():
        request = service.build_request_from_library(
            **scenario.service_arguments(), evidence_label="SYNTHETIC"
        )
        assert request.subrep_decision.selected_skill_id == scenario.expected_selected_skill_id, (
            scenario.name
        )


def test_scenarios_cover_every_exclusion_stage_and_abstention(tmp_path):
    service = _service(tmp_path)
    requests = {
        scenario.name: service.build_request_from_library(
            **scenario.service_arguments(), evidence_label="SYNTHETIC"
        )
        for scenario in predefined_scenarios()
    }

    def stages(name):
        return [exclusion.stage for exclusion in requests[name].exclusions]

    assert set(stages("mixed_exclusions")) == EXCLUSION_STAGES
    assert set(stages("abstention_mixed_reasons")) == {"CERTIFICATION", "OPERATOR", "RISK_BUDGET"}
    assert requests["abstention_mixed_reasons"].subrep_decision.abstain
    assert stages("risk_budget_exclusion") == ["RISK_BUDGET"]
    assert stages("runtime_inadmissible") == ["RUNTIME_ADMISSIBILITY"]


def test_synthetic_backend_explains_every_scenario_without_findings(tmp_path):
    service = _service(tmp_path)

    for scenario in predefined_scenarios():
        outcome = service.recommend_from_library(
            **scenario.service_arguments(), evidence_label="SYNTHETIC"
        )
        assert outcome.status in {"accepted", "abstained"}, scenario.name
        assert [attempt["findings"] for attempt in outcome.attempts] == [[]], scenario.name

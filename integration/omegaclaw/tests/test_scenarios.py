from __future__ import annotations

import json

from integration.omegaclaw.adapter import OmegaRecommendationAdapter
from integration.omegaclaw.audit import JsonlAuditStore
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
            task_context=scenario.task_context,
            objective_weights=scenario.objective_weights,
            risk_budget=scenario.risk_budget,
            skill_library=scenario.skill_library,
            exclusions=scenario.exclusions,
            evidence_label="SYNTHETIC",
            request_id=f"test-{scenario.name}",
        )

    assert outcomes["clear_preferred_skill"].selected_skill_id == "skill_safety"
    assert outcomes["changed_task_priorities"].selected_skill_id == "skill_efficiency"
    assert outcomes["excluded_skill"].selected_skill_id == "skill_safety"
    assert outcomes["no_admissible_options"].status == "abstained"
    assert "0.820" in outcomes["clear_preferred_skill"].explanation
    assert "No admitted skills" in outcomes["no_admissible_options"].explanation

    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    excluded_record = next(
        record for record in records if record["request"]["request_id"] == "test-excluded_skill"
    )
    assert excluded_record["request"]["exclusions"][0]["skill_id"] == "skill_efficiency"
    assert all(
        record["request"]["evidence_label"] == "SYNTHETIC" for record in records
    )

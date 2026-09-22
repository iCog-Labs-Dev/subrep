from __future__ import annotations

import json
from datetime import datetime, timezone

from certification.certificate_schema import Certificate
from integration.omegaclaw.adapter import OmegaRecommendationAdapter
from integration.omegaclaw.audit import JsonlAuditStore
from integration.omegaclaw.contracts import RiskBudget
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
        risk_budget=RiskBudget(max_certificate_epsilon=0.1),
        skill_library=library,
        evidence_label="SYNTHETIC",
    )

    assert outcome.status == "abstained"
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    assert record["request"]["admitted_skills"] == []
    assert record["request"]["exclusions"][0]["reason_code"] == "RISK_BUDGET_EXCEEDED"

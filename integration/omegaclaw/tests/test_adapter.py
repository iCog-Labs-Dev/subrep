from __future__ import annotations

import json
from datetime import datetime, timezone

from integration.omegaclaw.adapter import OmegaRecommendationAdapter
from integration.omegaclaw.audit import JsonlAuditStore
from integration.omegaclaw.contracts import (
    RESPONSE_SCHEMA_VERSION,
    ObjectiveWeight,
    RecommendationRequest,
    RiskBudget,
    SkillEvidence,
)


class StaticBackend:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def complete(self, prompt, request_id, timeout_seconds):
        assert request_id in prompt
        assert timeout_seconds > 0
        if self.error is not None:
            raise self.error
        return self.response


def _request() -> RecommendationRequest:
    return RecommendationRequest(
        request_id="request-1",
        created_at=datetime.now(timezone.utc).isoformat(),
        task_context={"task": "land safely"},
        objective_weights=(
            ObjectiveWeight("safety", 0.75),
            ObjectiveWeight("fuel", 0.25),
        ),
        risk_budget=RiskBudget(max_certificate_epsilon=0.0),
        admitted_skills=(
            SkillEvidence(
                skill_id="safe_skill",
                score=1.5,
                delta_r=1.0,
                objective_deltas={"safety": 0.6, "fuel": 0.2},
                gate_type="CDS",
                admission_margin=1.2,
                epsilon=0.0,
                weight_region_type="FULL_SIMPLEX",
            ),
        ),
    )


def _response(**changes) -> str:
    payload = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "request_id": "request-1",
        "selected_skill_id": "safe_skill",
        "abstain": False,
        "explanation": "safe_skill has the supplied highest score of 1.5.",
        "cited_skill_ids": ["safe_skill"],
    }
    payload.update(changes)
    return json.dumps(payload)


def test_accepts_admitted_skill_and_saves_audit(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=_response()),
        JsonlAuditStore(audit_path),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "accepted"
    assert outcome.selected_skill_id == "safe_skill"
    saved = json.loads(audit_path.read_text(encoding="utf-8"))
    assert saved["request"]["admitted_skills"][0]["skill_id"] == "safe_skill"
    assert saved["outcome"]["status"] == "accepted"


def test_rejects_skill_outside_admitted_set_and_records_raw_response(tmp_path):
    raw = _response(
        selected_skill_id="excluded_skill",
        cited_skill_ids=["excluded_skill"],
    )
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=raw),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert "admitted set" in outcome.error or "unknown skills" in outcome.error
    assert outcome.raw_response == raw


def test_records_backend_error(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    adapter = OmegaRecommendationAdapter(
        StaticBackend(error=TimeoutError("backend unavailable")),
        JsonlAuditStore(audit_path),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "backend_error"
    assert "backend unavailable" in outcome.error
    assert json.loads(audit_path.read_text(encoding="utf-8"))["outcome"]["status"] == "backend_error"


def test_backend_value_error_is_not_misclassified_as_invalid_response(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(error=ValueError("transport configuration failed")),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "backend_error"
    assert "transport configuration failed" in outcome.error


def test_accepts_explicit_abstention(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                selected_skill_id=None,
                abstain=True,
                explanation="The supplied evidence is insufficient.",
                cited_skill_ids=[],
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "abstained"
    assert outcome.selected_skill_id is None


def test_rejects_malformed_json(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response="not-json"),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert outcome.raw_response == "not-json"


def test_rejects_wrong_request_id(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=_response(request_id="stale-request")),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert "request_id" in outcome.error


def test_rejects_wrong_schema_version(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=_response(schema_version="unsupported.v2")),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert "schema_version" in outcome.error


def test_rejects_unknown_evidence_citation(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=_response(cited_skill_ids=["safe_skill", "invented_skill"])),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert "unknown skills" in outcome.error


def test_prompt_contains_all_required_decision_inputs(tmp_path):
    class InspectingBackend:
        def complete(self, prompt, request_id, timeout_seconds):
            del timeout_seconds
            request_json = prompt.split("SUBREP_REQUEST_JSON_BEGIN\n", 1)[1].split(
                "\nSUBREP_REQUEST_JSON_END", 1
            )[0]
            payload = json.loads(request_json)
            assert payload["task_context"] == {"task": "land safely"}
            assert payload["objective_weights"]
            assert payload["risk_budget"] == {
                "max_certificate_epsilon": 0.0,
                "minimum_admission_margin": None,
            }
            assert payload["admitted_skills"][0]["score"] == 1.5
            assert payload["exclusions"] == []
            return _response(request_id=request_id)

    adapter = OmegaRecommendationAdapter(
        InspectingBackend(),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    assert adapter.recommend(_request()).status == "accepted"

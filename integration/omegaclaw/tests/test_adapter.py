from __future__ import annotations

import json
from datetime import datetime, timezone

from integration.omegaclaw.adapter import (
    OmegaRecommendationAdapter,
    build_recommendation_prompt,
)
from integration.omegaclaw.audit import JsonlAuditStore
from integration.omegaclaw.contracts import (
    INTEGRATION_MODE,
    RESPONSE_SCHEMA_VERSION,
    SELECTION_BASIS,
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


def _request_with_runner() -> RecommendationRequest:
    request = _request()
    runner = SkillEvidence(
        skill_id="runner_skill",
        score=1.0,
        delta_r=0.5,
        objective_deltas={"safety": 0.6, "fuel": 0.2},
        gate_type="CDS",
        admission_margin=0.8,
        epsilon=0.0,
        weight_region_type="FULL_SIMPLEX",
    )
    return RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=request.risk_budget,
        admitted_skills=request.admitted_skills + (runner,),
    )


def _request_with_selected_skill(**skill_changes) -> RecommendationRequest:
    request = _request()
    values = {
        "skill_id": "safe_skill",
        "score": 1.5,
        "delta_r": 1.0,
        "objective_deltas": {"safety": 0.6, "fuel": 0.2},
        "gate_type": "CDS",
        "admission_margin": 1.2,
        "epsilon": 0.0,
        "weight_region_type": "FULL_SIMPLEX",
    }
    values.update(skill_changes)
    skill = SkillEvidence(**values)
    return RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=RiskBudget(max_certificate_epsilon=max(0.0, skill.epsilon)),
        admitted_skills=(skill,),
    )


def _response(**changes) -> str:
    payload = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "request_id": "request-1",
        "selected_skill_id": "safe_skill",
        "abstain": False,
        "mode": INTEGRATION_MODE,
        "selection_basis": SELECTION_BASIS,
        "explanation": "safe_skill was selected by SubRep with a score of 1.5.",
        "evidence_refs": ["subrep_decision.selected_score"],
        "advisory_concerns": [],
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
    assert saved["schema_version"] == "subrep.omegaclaw.explanation.audit.v1"
    assert saved["request"]["subrep_decision"]["selected_skill_id"] == "safe_skill"
    assert saved["request"]["admitted_skills"]["safe_skill"]["final_selection_score"] == 1.5
    assert saved["outcome"]["status"] == "accepted"


def test_rejects_skill_outside_admitted_set_and_records_raw_response(tmp_path):
    raw = _response(
        selected_skill_id="excluded_skill",
        explanation="excluded_skill was selected.",
    )
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=raw),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert "authoritative SubRep decision" in outcome.error
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
    request = _request()
    request = RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=request.risk_budget,
        admitted_skills=(),
    )
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                selected_skill_id=None,
                abstain=True,
                explanation="No recommendation was made because no admissible skills exist.",
                evidence_refs=["subrep_decision.reason_code"],
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(request)

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


def test_rejects_wrong_mode_and_selection_basis(tmp_path):
    for changes, expected_error in (
        ({"mode": "SELECTION"}, "mode"),
        ({"selection_basis": "OMEGA_DECISION"}, "selection_basis"),
    ):
        adapter = OmegaRecommendationAdapter(
            StaticBackend(response=_response(**changes)),
            JsonlAuditStore(tmp_path / f"{expected_error}.jsonl"),
            invalid_response_retries=0,
        )

        outcome = adapter.recommend(_request())

        assert outcome.status == "invalid_response"
        assert expected_error in outcome.error


def test_rejects_unexpected_response_fields(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=_response(reconsidered_skill_id="other_skill")),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert "unexpected fields" in outcome.error


def test_rejects_unknown_evidence_reference(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                evidence_refs=[
                    "subrep_decision.selected_score",
                    "explanation_evidence.skills.invented_skill.delta_r",
                ]
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert "unknown evidence_refs" in outcome.error


def test_requires_runner_up_score_reference_when_runner_exists(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=_response()),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(_request_with_runner())

    assert outcome.status == "invalid_response"
    assert "runner_up_score" in outcome.error


def test_accepts_evidence_grounded_advisory_concern(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                evidence_refs=[
                    "subrep_decision.selected_score",
                    "subrep_decision.runner_up_score",
                ],
                advisory_concerns=[
                    {
                        "code": "SMALL_SCORE_GAP",
                        "message": "The two admitted scores are close.",
                        "evidence_refs": ["subrep_decision.score_gap"],
                    }
                ],
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(_request_with_runner())

    assert outcome.status == "accepted"
    assert outcome.response["advisory_concerns"][0]["code"] == "SMALL_SCORE_GAP"


def test_rejects_advisory_concern_with_irrelevant_valid_evidence(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                advisory_concerns=[
                    {
                        "code": "LOW_CERTIFICATE_MARGIN",
                        "message": "The margin may be low.",
                        "evidence_refs": ["subrep_decision.selected_score"],
                    }
                ]
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert "certificate margin" in outcome.error


def test_accepts_certificate_advisories_with_relevant_evidence(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                advisory_concerns=[
                    {
                        "code": "PDS_EPSILON_USED",
                        "message": "The selected skill uses a positive PDS epsilon.",
                        "evidence_refs": [
                            "audit_only_certificate_evidence.skills.safe_skill.gate_type",
                            "audit_only_certificate_evidence.skills.safe_skill.epsilon",
                        ],
                    },
                    {
                        "code": "LOW_CERTIFICATE_MARGIN",
                        "message": "Review the selected skill's certificate margin.",
                        "evidence_refs": [
                            "audit_only_certificate_evidence.skills.safe_skill.certificate_admission_margin"
                        ],
                    },
                ]
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )
    request = _request_with_selected_skill(gate_type="PDS", epsilon=0.1)

    outcome = adapter.recommend(request)

    assert outcome.status == "accepted"


def test_accepts_negative_high_priority_objective_advisories(tmp_path):
    objective_ref = "explanation_evidence.skills.safe_skill.objective_deltas.safety"
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                advisory_concerns=[
                    {
                        "code": "NEGATIVE_OBJECTIVE_TRADEOFF",
                        "message": "The selected skill reduces the safety objective.",
                        "evidence_refs": [objective_ref],
                    },
                    {
                        "code": "HIGH_PRIORITY_OBJECTIVE_DECLINE",
                        "message": "The highest-weight objective declines.",
                        "evidence_refs": [
                            "explanation_evidence.objective_weights.safety",
                            objective_ref,
                        ],
                    },
                ]
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )
    request = _request_with_selected_skill(
        delta_r=1.6,
        objective_deltas={"safety": -0.2, "fuel": 0.2},
    )

    outcome = adapter.recommend(request)

    assert outcome.status == "accepted"


def test_rejects_advisory_concern_with_unknown_evidence(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                advisory_concerns=[
                    {
                        "code": "LOW_CERTIFICATE_MARGIN",
                        "message": "The margin may be low.",
                        "evidence_refs": [
                            "audit_only_certificate_evidence.skills.other_skill.certificate_admission_margin"
                        ],
                    }
                ]
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert "advisory concern contains unknown evidence_refs" in outcome.error


def test_prompt_contains_all_required_decision_inputs(tmp_path):
    class InspectingBackend:
        def complete(self, prompt, request_id, timeout_seconds):
            del timeout_seconds
            request_json = prompt.split("SUBREP_REQUEST_JSON_BEGIN\n", 1)[1].split(
                "\nSUBREP_REQUEST_JSON_END", 1
            )[0]
            payload = json.loads(request_json)
            assert payload["task_context"] == {"task": "land safely"}
            assert payload["explanation_evidence"]["objective_weights"]
            assert payload["risk_budget"] == {
                "max_certificate_epsilon": 0.0,
                "minimum_admission_margin": None,
            }
            assert payload["mode"] == "EXPLANATION_ONLY"
            assert payload["subrep_decision"]["selected_skill_id"] == "safe_skill"
            assert payload["admitted_skills"]["safe_skill"]["final_selection_score"] == 1.5
            assert payload["exclusions"] == []
            response_shape = json.loads(
                prompt.split("Required response shape: ", 1)[1].split("\n", 1)[0]
            )
            assert response_shape["selected_skill_id"] == "safe_skill"
            assert response_shape["abstain"] is False
            assert response_shape["evidence_refs"] == [
                "subrep_decision.selected_score"
            ]
            assert "LOW_CERTIFICATE_MARGIN" in prompt
            return _response(request_id=request_id)

    adapter = OmegaRecommendationAdapter(
        InspectingBackend(),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    assert adapter.recommend(_request()).status == "accepted"


def test_abstention_prompt_uses_the_authoritative_abstention_shape():
    request = _request()
    request = RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=request.risk_budget,
        admitted_skills=(),
    )

    prompt = build_recommendation_prompt(request)
    response_shape = json.loads(
        prompt.split("Required response shape: ", 1)[1].split("\n", 1)[0]
    )

    assert response_shape["selected_skill_id"] is None
    assert response_shape["abstain"] is True
    assert response_shape["evidence_refs"] == ["subrep_decision.reason_code"]


def test_rejects_suboptimal_admitted_skill(tmp_path):
    request = _request()
    lower = SkillEvidence(
        skill_id="lower_skill",
        score=0.3,
        delta_r=0.1,
        objective_deltas={"safety": 0.2, "fuel": 0.2},
        gate_type="CDS",
        admission_margin=0.3,
        epsilon=0.0,
        weight_region_type="FULL_SIMPLEX",
    )
    request = RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=request.risk_budget,
        admitted_skills=request.admitted_skills + (lower,),
    )
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                selected_skill_id="lower_skill",
                explanation="lower_skill was selected.",
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(request)

    assert outcome.status == "invalid_response"
    assert "authoritative SubRep decision" in outcome.error


def test_rejects_abstention_when_candidates_exist(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                selected_skill_id=None,
                abstain=True,
                explanation="No recommendation was made.",
                evidence_refs=["subrep_decision.selected_score"],
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert "does not echo" in outcome.error


def test_retries_one_invalid_response_and_saves_both_attempts(tmp_path):
    class SequenceBackend:
        def __init__(self):
            self.responses = iter(("not-json", _response()))

        def complete(self, prompt, request_id, timeout_seconds):
            assert request_id in prompt
            assert timeout_seconds > 0
            return next(self.responses)

    audit_path = tmp_path / "audit.jsonl"
    adapter = OmegaRecommendationAdapter(
        SequenceBackend(),
        JsonlAuditStore(audit_path),
        invalid_response_retries=1,
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "accepted"
    assert [attempt["status"] for attempt in outcome.attempts] == [
        "invalid_response",
        "valid",
    ]
    saved = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(saved["outcome"]["attempts"]) == 2


def test_rejects_wrong_lexical_tie_break_selection(tmp_path):
    request = _request()
    tied = SkillEvidence(
        skill_id="zeta_skill",
        score=1.5,
        delta_r=1.0,
        objective_deltas={"safety": 0.6, "fuel": 0.2},
        gate_type="CDS",
        admission_margin=1.2,
        epsilon=0.0,
        weight_region_type="FULL_SIMPLEX",
    )
    request = RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=request.risk_budget,
        admitted_skills=request.admitted_skills + (tied,),
    )
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                selected_skill_id="zeta_skill",
                explanation="zeta_skill was selected.",
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(request)

    assert outcome.status == "invalid_response"
    assert "authoritative SubRep decision" in outcome.error


def test_rejects_selection_when_admitted_set_is_empty(tmp_path):
    request = _request()
    request = RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=request.risk_budget,
        admitted_skills=(),
    )
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=_response(evidence_refs=["subrep_decision.reason_code"])),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(request)

    assert outcome.status == "invalid_response"
    assert "authoritative SubRep decision" in outcome.error

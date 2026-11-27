from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from integration.omegaclaw.adapter import (
    OmegaRecommendationAdapter,
    build_correction_prompt,
    build_recommendation_prompt,
)
from integration.omegaclaw.audit import JsonlAuditStore
from integration.omegaclaw.contracts import (
    AUDIT_SCHEMA_VERSION,
    INTEGRATION_MODE,
    MAX_CLAIMS,
    RESPONSE_SCHEMA_VERSION,
    SELECTION_BASIS,
    ObjectiveWeight,
    RecommendationRequest,
    RiskBudget,
    SkillEvidence,
    SkillExclusion,
)
from integration.omegaclaw.websocket_gateway import OmegaGatewayError, OmegaGatewayTimeout


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


def _request_with_excluded_skill() -> RecommendationRequest:
    request = _request()
    return RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=request.risk_budget,
        admitted_skills=request.admitted_skills,
        exclusions=(
            SkillExclusion(
                skill_id="excluded_skill",
                reason_code="OPERATOR_EXCLUSION",
                reason="Excluded before ranking.",
            ),
        ),
    )


def _synthetic_request() -> RecommendationRequest:
    request = _request()
    skill = request.admitted_skills[0]
    return RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=request.risk_budget,
        admitted_skills=(
            SkillEvidence(
                skill_id=skill.skill_id,
                score=skill.score,
                delta_r=skill.delta_r,
                objective_deltas=skill.objective_deltas,
                gate_type=skill.gate_type,
                admission_margin=skill.admission_margin,
                epsilon=skill.epsilon,
                weight_region_type=skill.weight_region_type,
                evidence_label="SYNTHETIC",
            ),
        ),
        evidence_label="SYNTHETIC",
    )


def _abstaining_request() -> RecommendationRequest:
    request = _request()
    return RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=request.risk_budget,
        admitted_skills=(),
    )


SAFETY_WEIGHT = {
    "type": "objective_weight",
    "evidence_ref": "explanation_evidence.objective_weights.safety",
    "value": 0.75,
}
SAFETY_DELTA = {
    "type": "objective_delta",
    "evidence_ref": "explanation_evidence.skills.safe_skill.objective_deltas.safety",
    "value": 0.6,
}


def _response(**changes) -> str:
    payload = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "request_id": "request-1",
        "selected_skill_id": "safe_skill",
        "abstain": False,
        "mode": INTEGRATION_MODE,
        "selection_basis": SELECTION_BASIS,
        "claims": [SAFETY_WEIGHT, SAFETY_DELTA],
        "advisory_concerns": [],
    }
    payload.update(changes)
    return json.dumps(payload)


def _adapter(backend, tmp_path, **settings) -> OmegaRecommendationAdapter:
    return OmegaRecommendationAdapter(
        backend, JsonlAuditStore(tmp_path / "audit.jsonl"), **settings
    )


def test_accepts_verified_claims_writes_the_explanation_and_saves_audit(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=_response()), JsonlAuditStore(audit_path)
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "accepted"
    assert outcome.selected_skill_id == "safe_skill"
    assert outcome.explanation.startswith("SubRep selected safe_skill")
    assert "The objective weights are safety 0.75." in outcome.explanation
    assert "safe_skill has objective deltas safety +0.6." in outcome.explanation
    saved = json.loads(audit_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == AUDIT_SCHEMA_VERSION
    assert saved["request"]["subrep_decision"]["selected_skill_id"] == "safe_skill"
    assert saved["request"]["admitted_skills"]["safe_skill"]["final_selection_score"] == 1.5
    assert saved["outcome"]["status"] == "accepted"
    assert saved["outcome"]["attempts"][0]["explanation"] == outcome.explanation


def test_rejects_skill_outside_admitted_set_and_records_raw_response(tmp_path):
    raw = _response(selected_skill_id="excluded_skill")
    outcome = _adapter(StaticBackend(response=raw), tmp_path).recommend(_request())

    assert outcome.status == "invalid_response"
    assert "authoritative SubRep decision" in outcome.error
    assert outcome.raw_response == raw


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(StaticBackend(error=TimeoutError("backend unavailable")), id="timeout"),
        pytest.param(StaticBackend(response="not-json"), id="invalid-twice"),
    ],
)
def test_the_outcome_keeps_subrep_decision_when_no_explanation_is_accepted(backend, tmp_path):
    outcome = _adapter(backend, tmp_path).recommend(_request())

    assert outcome.status in {"backend_error", "invalid_response"}
    assert outcome.explanation is None
    assert outcome.selected_skill_id == "safe_skill"
    assert outcome.abstain is False
    assert outcome.subrep_decision["selected_skill_id"] == "safe_skill"
    assert outcome.subrep_decision["selected_score"] == 1.5


def test_a_timeout_is_recorded_and_not_retried(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    adapter = OmegaRecommendationAdapter(
        StaticBackend(error=OmegaGatewayTimeout("Omega did not answer")),
        JsonlAuditStore(audit_path),
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "backend_error"
    assert "Omega did not answer" in outcome.error
    assert len(outcome.attempts) == 1
    assert (
        json.loads(audit_path.read_text(encoding="utf-8"))["outcome"]["status"] == "backend_error"
    )


def test_a_transport_error_is_retried_once(tmp_path):
    class FlakyBackend:
        calls = 0

        def complete(self, prompt, request_id, timeout_seconds):
            FlakyBackend.calls += 1
            if FlakyBackend.calls == 1:
                raise OmegaGatewayError("Omega disconnected before the request was sent")
            return _response()

    outcome = _adapter(FlakyBackend(), tmp_path).recommend(_request())

    assert outcome.status == "accepted"
    assert [attempt["status"] for attempt in outcome.attempts] == ["backend_error", "valid"]


def test_a_backend_value_error_is_not_misclassified_as_invalid_response(tmp_path):
    outcome = _adapter(
        StaticBackend(error=ValueError("transport configuration failed")), tmp_path
    ).recommend(_request())

    assert outcome.status == "backend_error"
    assert "transport configuration failed" in outcome.error
    assert len(outcome.attempts) == 2  # one retry, then the error stands


def test_accepts_explicit_abstention_without_claims(tmp_path):
    response = _response(selected_skill_id=None, abstain=True, claims=[])

    outcome = _adapter(StaticBackend(response=response), tmp_path).recommend(_abstaining_request())

    assert outcome.status == "abstained"
    assert outcome.selected_skill_id is None
    assert outcome.abstain is True
    assert outcome.explanation.startswith("SubRep abstained (NO_ADMISSIBLE_SKILLS)")


@pytest.mark.parametrize(
    ("changes", "expected_error"),
    [
        pytest.param({"request_id": "stale-request"}, "request_id", id="wrong-request-id"),
        pytest.param(
            {"schema_version": "subrep.omegaclaw.explanation.response.v1"},
            "schema_version",
            id="old-schema",
        ),
        pytest.param({"mode": "SELECTION"}, "mode", id="wrong-mode"),
        pytest.param({"selection_basis": "OMEGA_DECISION"}, "selection_basis", id="wrong-basis"),
        pytest.param(
            {"reconsidered_skill_id": "other_skill"}, "unexpected fields", id="unexpected-field"
        ),
        pytest.param(
            {"explanation": "Free text from the model."},
            "unexpected fields",
            id="free-text-explanation",
        ),
        pytest.param(
            {"claims": "safe_skill was selected"},
            "claims must be an array",
            id="claims-not-an-array",
        ),
        pytest.param(
            {"claims": [{"type": "score", "value": 1.5}]},
            "missing ['evidence_ref']",
            id="claim-without-a-path",
        ),
        pytest.param(
            {"claims": [SAFETY_DELTA] * (MAX_CLAIMS + 1)},
            f"at most {MAX_CLAIMS}",
            id="too-many-claims",
        ),
        pytest.param(
            {"claims": []},
            "at least one objective_weight or objective_delta claim",
            id="selection-without-claims",
        ),
        pytest.param(
            {"advisory_concerns": ["NEGATIVE_OBJECTIVE_TRADEOFF"]},
            "items must be objects",
            id="bare-advisory-code",
        ),
    ],
)
def test_rejects_a_response_that_breaks_the_contract(changes, expected_error, tmp_path):
    outcome = _adapter(
        StaticBackend(response=_response(**changes)), tmp_path, invalid_response_retries=0
    ).recommend(_request())

    assert outcome.status == "invalid_response"
    assert expected_error in outcome.error


def test_rejects_malformed_json(tmp_path):
    outcome = _adapter(StaticBackend(response="not-json"), tmp_path).recommend(_request())

    assert outcome.status == "invalid_response"
    assert outcome.raw_response == "not-json"


def test_rejects_a_claim_about_a_skill_that_is_not_in_the_request(tmp_path):
    claim = {
        "type": "delta_r",
        "evidence_ref": "explanation_evidence.skills.invented_skill.delta_r",
        "value": 1.0,
    }
    outcome = _adapter(
        StaticBackend(response=_response(claims=[SAFETY_DELTA, claim])), tmp_path
    ).recommend(_request())

    assert outcome.status == "invalid_response"
    assert "which is not in the request" in outcome.error


def test_rejects_a_claim_with_the_wrong_value(tmp_path):
    wrong = {**SAFETY_DELTA, "value": -0.6}
    outcome = _adapter(
        StaticBackend(response=_response(claims=[wrong])), tmp_path, invalid_response_retries=0
    ).recommend(_request())

    assert outcome.status == "invalid_response"
    assert "but the request has 0.6" in outcome.error


def _close_runner_request() -> RecommendationRequest:
    # 0.95 + 0.75 * 0.6 + 0.25 * 0.2 = 1.45: a gap of 0.05, 3% of the selected 1.5.
    request = _request()
    runner = SkillEvidence(
        skill_id="runner_skill",
        score=1.45,
        delta_r=0.95,
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


GAP_CONCERN = {"code": "SMALL_SCORE_GAP", "evidence_refs": ["subrep_decision.score_gap"]}


def test_accepts_a_small_score_gap_concern_and_states_it(tmp_path):
    outcome = _adapter(
        StaticBackend(response=_response(advisory_concerns=[GAP_CONCERN])),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(_close_runner_request())

    assert outcome.status == "accepted"
    assert outcome.response["advisory_concerns"][0]["code"] == "SMALL_SCORE_GAP"
    assert (
        "Advisory note: the gap of 0.05 to the runner-up is small, 3% of the selected score."
        in outcome.explanation
    )


def test_rejects_a_small_score_gap_concern_for_a_large_gap(tmp_path):
    # runner_skill scores 1.0, a gap of 0.5: a third of the selected score.
    outcome = _adapter(
        StaticBackend(response=_response(advisory_concerns=[GAP_CONCERN])),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(_request_with_runner())

    assert outcome.status == "invalid_response"
    assert "at most 10% of the selected score" in outcome.error


def test_rejects_advisory_concern_with_irrelevant_valid_evidence(tmp_path):
    concern = {
        "code": "LOW_CERTIFICATE_MARGIN",
        "evidence_refs": ["subrep_decision.selected_score"],
    }
    outcome = _adapter(
        StaticBackend(response=_response(advisory_concerns=[concern])),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(_request())

    assert outcome.status == "invalid_response"
    assert "certificate margin" in outcome.error


def _margin_request(margin: float, minimum: float | None) -> RecommendationRequest:
    request = _request()
    skill = SkillEvidence(
        skill_id="safe_skill",
        score=1.5,
        delta_r=1.0,
        objective_deltas={"safety": 0.6, "fuel": 0.2},
        gate_type="CDS",
        admission_margin=margin,
        epsilon=0.0,
        weight_region_type="FULL_SIMPLEX",
    )
    return RecommendationRequest(
        request_id=request.request_id,
        created_at=request.created_at,
        task_context=request.task_context,
        objective_weights=request.objective_weights,
        risk_budget=RiskBudget(max_certificate_epsilon=0.0, minimum_admission_margin=minimum),
        admitted_skills=(skill,),
    )


MARGIN_CONCERN = {
    "code": "LOW_CERTIFICATE_MARGIN",
    "evidence_refs": [
        "audit_only_certificate_evidence.skills.safe_skill.certificate_admission_margin"
    ],
}


@pytest.mark.parametrize(
    ("margin", "minimum", "expected_status", "expected_text"),
    [
        pytest.param(
            1.05, 1.0, "accepted", "is close to the local minimum of 1", id="within-10-percent"
        ),
        pytest.param(
            1.2, 1.0, "invalid_response", "needs a margin of at most 1.1", id="20-percent-above"
        ),
        pytest.param(
            1.2, None, "invalid_response", "needs a minimum admission margin", id="no-minimum"
        ),
    ],
)
def test_low_certificate_margin_needs_a_margin_close_to_the_minimum(
    margin, minimum, expected_status, expected_text, tmp_path
):
    outcome = _adapter(
        StaticBackend(response=_response(advisory_concerns=[MARGIN_CONCERN])),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(_margin_request(margin, minimum))

    assert outcome.status == expected_status
    assert expected_text in (outcome.explanation or outcome.error)


def test_accepts_a_pds_epsilon_concern_with_relevant_evidence(tmp_path):
    concern = {
        "code": "PDS_EPSILON_USED",
        "evidence_refs": [
            "audit_only_certificate_evidence.skills.safe_skill.gate_type",
            "audit_only_certificate_evidence.skills.safe_skill.epsilon",
        ],
    }
    outcome = _adapter(
        StaticBackend(response=_response(advisory_concerns=[concern])),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(_request_with_selected_skill(gate_type="PDS", epsilon=0.1))

    assert outcome.status == "accepted"
    assert "certified by the PDS gate with a positive epsilon of 0.1" in outcome.explanation


def test_accepts_negative_and_high_priority_objective_concerns(tmp_path):
    delta_ref = "explanation_evidence.skills.safe_skill.objective_deltas.safety"
    concerns = [
        {"code": "NEGATIVE_OBJECTIVE_TRADEOFF", "evidence_refs": [delta_ref]},
        {
            "code": "HIGH_PRIORITY_OBJECTIVE_DECLINE",
            "evidence_refs": ["explanation_evidence.objective_weights.safety", delta_ref],
        },
    ]
    claims = [{"type": "objective_delta", "evidence_ref": delta_ref, "value": -0.2}]
    outcome = _adapter(
        StaticBackend(response=_response(claims=claims, advisory_concerns=concerns)),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(
        _request_with_selected_skill(delta_r=1.6, objective_deltas={"safety": -0.2, "fuel": 0.2})
    )

    assert outcome.status == "accepted"
    # Both concerns cite the same decline of the highest-weight objective: it is stated once.
    assert outcome.explanation.count("Advisory note:") == 1
    assert (
        "safe_skill lowers safety (delta -0.2), the objective with the highest weight (0.75)."
        in (outcome.explanation)
    )


def test_rejects_advisory_concern_with_unknown_evidence(tmp_path):
    concern = {
        "code": "LOW_CERTIFICATE_MARGIN",
        "evidence_refs": [
            "audit_only_certificate_evidence.skills.other_skill.certificate_admission_margin"
        ],
    }
    outcome = _adapter(
        StaticBackend(response=_response(advisory_concerns=[concern])),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(_request())

    assert outcome.status == "invalid_response"
    assert "advisory concern contains unknown evidence_refs" in outcome.error


def test_rejects_an_advisory_message_written_by_the_model(tmp_path):
    concern = {**MARGIN_CONCERN, "message": "The margin is low."}
    outcome = _adapter(
        StaticBackend(response=_response(advisory_concerns=[concern])),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(_request())

    assert outcome.status == "invalid_response"
    assert "unexpected ['message']" in outcome.error


def _prompt_parts(prompt: str) -> tuple[str, dict, dict, dict]:
    instructions = prompt.split("Required response shape: ", 1)[0]
    shape = json.loads(prompt.split("Required response shape: ", 1)[1].split("\n", 1)[0])
    facts = json.loads(prompt.split("CLAIMABLE_FACTS: ", 1)[1].split("\n", 1)[0])
    request_json = prompt.split("SUBREP_REQUEST_JSON_BEGIN\n", 1)[1].split(
        "\nSUBREP_REQUEST_JSON_END", 1
    )[0]
    return instructions, shape, facts, json.loads(request_json)


def test_prompt_asks_for_claims_and_lists_every_claimable_fact():
    instructions, shape, facts, payload = _prompt_parts(
        build_recommendation_prompt(_request_with_excluded_skill())
    )

    assert "Do not write any explanation text" in instructions
    assert "Only the paths listed in CLAIMABLE_FACTS can be claimed" in instructions
    assert "keeping the sign of every number" in instructions
    assert "starts with the word send, then a space" in instructions
    assert "each double quote in this message as _quote_" in instructions
    assert "LOW_CERTIFICATE_MARGIN" in instructions
    assert shape["selected_skill_id"] == "safe_skill"
    assert shape["abstain"] is False
    assert set(shape) == {
        "schema_version",
        "request_id",
        "selected_skill_id",
        "abstain",
        "mode",
        "selection_basis",
        "claims",
        "advisory_concerns",
    }
    assert facts["explanation_evidence.skills.safe_skill.objective_deltas.safety"] == {
        "type": "objective_delta",
        "value": 0.6,
    }
    assert facts["exclusions.0.stage"] == {"type": "exclusion_stage", "value": "OPERATOR"}
    assert "subrep_decision.selected_skill_id" not in facts
    assert payload["task_context"] == {"task": "land safely"}
    assert payload["risk_budget"] == {
        "max_certificate_epsilon": 0.0,
        "minimum_admission_margin": None,
    }
    assert payload["objective_order"] == ["safety", "fuel"]


def test_prompt_shows_numbers_without_float_noise():
    # The clear_preferred_skill numbers: 0.2 + 0.8 * 0.8 + 0.2 * -0.1 is 0.8200000000000001.
    skill = SkillEvidence(
        skill_id="safe_skill",
        score=0.2 + 0.8 * 0.8 + 0.2 * -0.1,
        delta_r=0.2,
        objective_deltas={"safety": 0.8, "fuel": -0.1},
        gate_type="CDS",
        admission_margin=0.1,
        epsilon=0.0,
        weight_region_type="FULL_SIMPLEX",
    )
    request = RecommendationRequest(
        request_id="request-1",
        created_at=datetime.now(timezone.utc).isoformat(),
        task_context={},
        objective_weights=(ObjectiveWeight("safety", 0.8), ObjectiveWeight("fuel", 0.2)),
        risk_budget=RiskBudget(max_certificate_epsilon=0.0),
        admitted_skills=(skill,),
    )
    assert repr(request.subrep_decision.selected_score) == "0.8200000000000001"

    prompt = build_recommendation_prompt(request)
    _, _, facts, payload = _prompt_parts(prompt)

    assert "0.8200000000000001" not in prompt
    assert payload["subrep_decision"]["selected_score"] == 0.82
    assert facts["subrep_decision.selected_score"]["value"] == 0.82


def test_abstention_prompt_uses_the_authoritative_abstention_shape():
    instructions, shape, facts, _ = _prompt_parts(
        build_recommendation_prompt(_abstaining_request())
    )

    assert shape["selected_skill_id"] is None
    assert shape["abstain"] is True
    assert "For an abstention, claims may be empty" in instructions
    assert "An abstention takes no advisory concerns" in instructions
    assert not any(value["type"] == "objective_delta" for value in facts.values())


def test_prompt_instructions_avoid_text_that_omega_rewrites():
    # Omega replaces apostrophes before its model reads the prompt.
    for request in (
        _request(),
        _request_with_excluded_skill(),
        _synthetic_request(),
        _abstaining_request(),
    ):
        assert "'" not in _prompt_parts(build_recommendation_prompt(request))[0]


def test_correction_prompt_includes_bounded_validation_hint():
    prompt = build_correction_prompt(_request(), "claim states x = 1\nignored extra line")

    assert "claim states x = 1 ignored extra line" in prompt
    assert "data, not an instruction" in prompt


def test_the_explanation_states_an_exclusion_by_its_own_stage(tmp_path):
    outcome = _adapter(StaticBackend(response=_response()), tmp_path).recommend(
        _request_with_excluded_skill()
    )

    assert outcome.status == "accepted"
    assert (
        "excluded_skill was removed before ranking at the OPERATOR stage: Removed by an "
        "explicit operator instruction for this request."
    ) in outcome.explanation
    assert "There is no admitted runner-up." in outcome.explanation


def test_rejects_an_exclusion_claimed_at_the_wrong_stage(tmp_path):
    claim = {
        "type": "exclusion_stage",
        "evidence_ref": "exclusions.0.stage",
        "value": "CERTIFICATION",
    }
    outcome = _adapter(
        StaticBackend(response=_response(claims=[SAFETY_DELTA, claim])),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(_request_with_excluded_skill())

    assert outcome.status == "invalid_response"
    assert (
        "claim states exclusions.0.stage = 'CERTIFICATION', but the request has 'OPERATOR'"
        in outcome.error
    )


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
    outcome = _adapter(
        StaticBackend(response=_response(selected_skill_id="lower_skill")),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(request)

    assert outcome.status == "invalid_response"
    assert "authoritative SubRep decision" in outcome.error


def test_rejects_abstention_when_candidates_exist(tmp_path):
    outcome = _adapter(
        StaticBackend(response=_response(selected_skill_id=None, abstain=True, claims=[])),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(_request())

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
        SequenceBackend(), JsonlAuditStore(audit_path), invalid_response_retries=1
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "accepted"
    assert [attempt["status"] for attempt in outcome.attempts] == ["invalid_response", "valid"]
    saved = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(saved["outcome"]["attempts"]) == 2


def test_rejects_wrong_lexical_tie_break_selection_and_states_the_tie(tmp_path):
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
    wrong = _adapter(
        StaticBackend(response=_response(selected_skill_id="zeta_skill")),
        tmp_path,
        invalid_response_retries=0,
    ).recommend(request)
    right = _adapter(StaticBackend(response=_response()), tmp_path).recommend(request)

    assert wrong.status == "invalid_response"
    assert "authoritative SubRep decision" in wrong.error
    assert (
        "zeta_skill has the same score; SubRep breaks exact ties by skill ID order"
        in right.explanation
    )


def test_rejects_selection_when_admitted_set_is_empty(tmp_path):
    outcome = _adapter(
        StaticBackend(response=_response()), tmp_path, invalid_response_retries=0
    ).recommend(_abstaining_request())

    assert outcome.status == "invalid_response"
    assert "authoritative SubRep decision" in outcome.error


def test_an_error_inside_the_checks_is_recorded_and_the_decision_still_returned(
    tmp_path, monkeypatch
):
    import integration.omegaclaw.adapter as adapter_module

    def broken_check(request, raw_response):
        raise ZeroDivisionError("float division by zero")

    monkeypatch.setattr(adapter_module, "assess_raw_response", broken_check)
    audit_path = tmp_path / "audit.jsonl"
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=_response()), JsonlAuditStore(audit_path)
    )

    outcome = adapter.recommend(_request())

    assert outcome.status == "invalid_response"
    assert outcome.selected_skill_id == "safe_skill"
    assert outcome.attempts[0]["findings"][0]["code"] == "INTERNAL_VALIDATION_ERROR"
    assert "ZeroDivisionError" in outcome.error
    assert (
        json.loads(audit_path.read_text(encoding="utf-8"))["outcome"]["status"]
        == "invalid_response"
    )

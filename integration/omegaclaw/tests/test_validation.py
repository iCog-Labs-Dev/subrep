from __future__ import annotations

import json

import pytest

from integration.omegaclaw.adapter import OmegaRecommendationAdapter
from integration.omegaclaw.audit import JsonlAuditStore
from integration.omegaclaw.contracts import (
    FACTUAL_FINDING,
    FORMAT_FINDING,
    Finding,
    OmegaRecommendationResponse,
)
from integration.omegaclaw.tests.test_adapter import (
    SAFETY_DELTA,
    StaticBackend,
    _request,
    _request_with_excluded_skill,
    _response,
)
from integration.omegaclaw.validation import (
    ResponseValidationError,
    assess_raw_response,
    validate_response_for_request,
)

WRONG_STAGE = {
    "type": "exclusion_stage",
    "evidence_ref": "exclusions.0.stage",
    "value": "CERTIFICATION",
}


def _codes(assessment) -> dict[str, str]:
    return {finding.code: finding.category for finding in assessment.findings}


def test_valid_response_has_no_findings_and_an_explanation():
    assessment = assess_raw_response(_request(), _response())

    assert assessment.valid
    assert assessment.response.selected_skill_id == "safe_skill"
    assert assessment.error is None
    assert assessment.claims_checked
    assert assessment.explanation.startswith("SubRep selected safe_skill")


def test_malformed_response_is_a_format_finding_only():
    assessment = assess_raw_response(_request(), "safe_skill was selected.")

    assert assessment.response is None
    assert _codes(assessment) == {"MALFORMED_RESPONSE": FORMAT_FINDING}
    assert not assessment.claims_checked
    assert assessment.explanation is None


def test_schema_violation_still_has_its_claims_verified():
    raw = json.loads(_response(claims=[SAFETY_DELTA, WRONG_STAGE]))
    raw["unexpected"] = True

    assessment = assess_raw_response(_request_with_excluded_skill(), json.dumps(raw))

    assert assessment.response is None
    assert assessment.claims_checked
    assert _codes(assessment) == {
        "SCHEMA_VIOLATION": FORMAT_FINDING,
        "CLAIM_VALUE_MISMATCH": FACTUAL_FINDING,
    }


def test_schema_violation_without_claims_is_not_verified():
    raw = json.loads(_response())
    del raw["claims"]

    assessment = assess_raw_response(_request(), json.dumps(raw))

    assert not assessment.claims_checked
    assert _codes(assessment) == {"SCHEMA_VIOLATION": FORMAT_FINDING}


def test_every_problem_is_reported_not_only_the_first():
    claims = [
        {**SAFETY_DELTA, "value": 9.5},
        {"type": "delta_r", "evidence_ref": "not.a.path", "value": 1.0},
    ]
    raw = _response(request_id="other-request", claims=claims)

    assessment = assess_raw_response(_request(), raw)

    assert _codes(assessment) == {
        "REQUEST_ID_MISMATCH": FORMAT_FINDING,
        "UNKNOWN_EVIDENCE_REF": FORMAT_FINDING,
        "CLAIM_VALUE_MISMATCH": FACTUAL_FINDING,
    }
    assert assessment.explanation is None


def test_every_problem_with_an_advisory_concern_is_reported():
    concern = {"code": "NEGATIVE_OBJECTIVE_TRADEOFF", "evidence_refs": ["not.a.path"]}

    assessment = assess_raw_response(_request(), _response(advisory_concerns=[concern]))

    assert _codes(assessment) == {
        "UNKNOWN_EVIDENCE_REF": FORMAT_FINDING,
        "UNSUPPORTED_ADVISORY_CONCERN": FACTUAL_FINDING,
    }


@pytest.mark.parametrize(
    ("reference", "stored", "expected"),
    [
        pytest.param("exclusions[0].stage", "exclusions.0.stage", {}, id="bracket-index"),
        pytest.param("exclusions.0.stage", "exclusions.0.stage", {}, id="dotted-index"),
        pytest.param(
            "exclusions[3].stage",
            "exclusions.3.stage",
            {"UNKNOWN_EVIDENCE_REF": FORMAT_FINDING},
            id="bracket-index-that-does-not-exist",
        ),
    ],
)
def test_bracket_and_dotted_indexes_name_the_same_evidence_path(reference, stored, expected):
    claim = {"type": "exclusion_stage", "evidence_ref": reference, "value": "OPERATOR"}
    raw = _response(claims=[SAFETY_DELTA, claim])

    assessment = assess_raw_response(_request_with_excluded_skill(), raw)

    assert _codes(assessment) == expected
    claims = OmegaRecommendationResponse.from_dict(json.loads(raw)).claims
    assert claims[1].evidence_ref == stored


def test_wrong_decision_echo_is_factual():
    request = _request()
    raw = _response(selected_skill_id="other_skill")
    response = OmegaRecommendationResponse.from_dict(json.loads(raw))

    with pytest.raises(ResponseValidationError) as raised:
        validate_response_for_request(request, response)

    categories = {finding.code: finding.category for finding in raised.value.findings}
    assert categories["DECISION_MISMATCH"] == FACTUAL_FINDING
    assert "does not echo the authoritative SubRep decision" in str(raised.value)


def test_adapter_records_findings_for_each_attempt(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(response=_response(claims=[SAFETY_DELTA, WRONG_STAGE])),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(_request_with_excluded_skill())

    assert outcome.status == "invalid_response"
    (attempt,) = outcome.attempts
    assert attempt["findings"] == [
        {
            "code": "CLAIM_VALUE_MISMATCH",
            "category": FACTUAL_FINDING,
            "message": (
                "claim states exclusions.0.stage = 'CERTIFICATION', but the request has "
                "'OPERATOR'"
            ),
        }
    ]
    assert attempt["claims_checked"] is True
    assert attempt["explanation"] is None


def test_correction_prompt_quotes_the_findings(tmp_path):
    prompts = []

    class RecordingBackend:
        def __init__(self):
            self.responses = [
                _response(claims=[SAFETY_DELTA, WRONG_STAGE]),
                _response(claims=[SAFETY_DELTA, {**WRONG_STAGE, "value": "OPERATOR"}]),
            ]

        def complete(self, prompt, request_id, timeout_seconds):
            del request_id, timeout_seconds
            prompts.append(prompt)
            return self.responses.pop(0)

    adapter = OmegaRecommendationAdapter(
        RecordingBackend(), JsonlAuditStore(tmp_path / "audit.jsonl")
    )

    outcome = adapter.recommend(_request_with_excluded_skill())

    assert outcome.status == "accepted"
    assert "but the request has 'OPERATOR'" in prompts[1]
    assert [attempt["status"] for attempt in outcome.attempts] == ["invalid_response", "valid"]


def test_finding_rejects_an_unknown_category():
    with pytest.raises(ValueError, match="category"):
        Finding("SOMETHING", "style", "not a category")


def test_an_abstention_takes_no_advisory_concerns():
    from integration.omegaclaw.tests.test_adapter import _abstaining_request

    concern = {
        "code": "NEGATIVE_OBJECTIVE_TRADEOFF",
        "evidence_refs": ["subrep_decision.reason_code"],
    }
    raw = _response(selected_skill_id=None, abstain=True, claims=[], advisory_concerns=[concern])

    assessment = assess_raw_response(_abstaining_request(), raw)

    assert _codes(assessment) == {"ADVISORY_NOT_ALLOWED": FORMAT_FINDING}

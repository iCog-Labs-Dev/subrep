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


def _codes(assessment) -> dict[str, str]:
    return {finding.code: finding.category for finding in assessment.findings}


def test_valid_response_has_no_findings():
    assessment = assess_raw_response(_request(), _response())

    assert assessment.valid
    assert assessment.response.selected_skill_id == "safe_skill"
    assert assessment.error is None


def test_malformed_response_is_a_format_finding_only():
    assessment = assess_raw_response(_request(), "safe_skill was selected.")

    assert assessment.response is None
    assert _codes(assessment) == {"MALFORMED_RESPONSE": FORMAT_FINDING}


def test_schema_violation_still_has_its_explanation_fact_checked():
    raw = json.loads(
        _response(explanation="excluded_skill was excluded during certification; safe_skill won.")
    )
    raw["unexpected"] = True

    assessment = assess_raw_response(_request_with_excluded_skill(), json.dumps(raw))

    assert assessment.response is None
    assert _codes(assessment) == {
        "SCHEMA_VIOLATION": FORMAT_FINDING,
        "WRONG_EXCLUSION_STAGE": FACTUAL_FINDING,
    }


def test_every_problem_is_reported_not_only_the_first():
    raw = _response(
        request_id="other-request",
        explanation="safe_skill was executed with a score of 9.5.",
        evidence_refs=["subrep_decision.selected_score", "not.a.path"],
    )

    assessment = assess_raw_response(_request(), raw)

    assert _codes(assessment) == {
        "REQUEST_ID_MISMATCH": FORMAT_FINDING,
        "UNKNOWN_EVIDENCE_REF": FORMAT_FINDING,
        "SCORE_MISMATCH": FACTUAL_FINDING,
        "UNSUPPORTED_NUMBER": FACTUAL_FINDING,
        "EXECUTION_CLAIM": FACTUAL_FINDING,
    }


def test_wrong_decision_echo_is_factual():
    request = _request()
    response = OmegaRecommendationResponse.from_dict(
        json.loads(_response(selected_skill_id="other_skill"))
    )

    with pytest.raises(ResponseValidationError) as raised:
        validate_response_for_request(request, response)

    categories = {finding.code: finding.category for finding in raised.value.findings}
    assert categories["DECISION_MISMATCH"] == FACTUAL_FINDING
    assert "does not echo the authoritative SubRep decision" in str(raised.value)


def test_adapter_records_findings_for_each_attempt(tmp_path):
    adapter = OmegaRecommendationAdapter(
        StaticBackend(
            response=_response(
                explanation="safe_skill was selected; excluded_skill failed certification."
            )
        ),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
        invalid_response_retries=0,
    )

    outcome = adapter.recommend(_request_with_excluded_skill())

    assert outcome.status == "invalid_response"
    (attempt,) = outcome.attempts
    assert attempt["findings"] == [
        {
            "code": "WRONG_EXCLUSION_STAGE",
            "category": FACTUAL_FINDING,
            "message": attempt["findings"][0]["message"],
        }
    ]
    assert "OPERATOR" in attempt["findings"][0]["message"]


def test_correction_prompt_quotes_the_findings(tmp_path):
    prompts = []

    class RecordingBackend:
        def __init__(self):
            self.responses = [
                _response(explanation="safe_skill was selected; excluded_skill failed certification."),
                _response(explanation="safe_skill was selected; the operator excluded excluded_skill."),
            ]

        def complete(self, prompt, request_id, timeout_seconds):
            del request_id, timeout_seconds
            prompts.append(prompt)
            return self.responses.pop(0)

    adapter = OmegaRecommendationAdapter(
        RecordingBackend(),
        JsonlAuditStore(tmp_path / "audit.jsonl"),
    )

    outcome = adapter.recommend(_request_with_excluded_skill())

    assert outcome.status == "accepted"
    assert "excluded it at stage OPERATOR" in prompts[1]
    assert [attempt["status"] for attempt in outcome.attempts] == ["invalid_response", "valid"]


def test_finding_rejects_an_unknown_category():
    with pytest.raises(ValueError, match="category"):
        Finding("SOMETHING", "style", "not a category")

"""Check an Omega response against the SubRep request it explains."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .claims import explanation_findings
from .contracts import (
    FACTUAL_FINDING,
    FORMAT_FINDING,
    AdvisoryConcern,
    Finding,
    OmegaRecommendationResponse,
    RecommendationRequest,
)


class ResponseValidationError(ValueError):
    """Raised with every finding when a response cannot be accepted."""

    def __init__(self, findings: Iterable[Finding]) -> None:
        self.findings = tuple(findings)
        if not self.findings:
            raise ValueError("ResponseValidationError requires at least one finding")
        super().__init__("; ".join(finding.message for finding in self.findings))


@dataclass(frozen=True)
class ResponseAssessment:
    """Every finding for one raw response, plus the parsed response when it parses.

    ``explanation_checked`` is false when the response had no explanation to check,
    so its factual accuracy is unknown rather than perfect.
    """

    response: OmegaRecommendationResponse | None
    findings: tuple[Finding, ...]
    explanation_checked: bool = True

    @property
    def valid(self) -> bool:
        return not self.findings

    @property
    def error(self) -> str | None:
        return "; ".join(finding.message for finding in self.findings) or None


def assess_raw_response(
    request: RecommendationRequest,
    raw_response: Any,
) -> ResponseAssessment:
    """Run every format and factual check that the raw response allows.

    A response that breaks the schema still has its explanation checked, so a
    report can tell factual errors apart from format errors.
    """

    try:
        payload = parse_structured_response(raw_response)
    except (ValueError, TypeError) as exc:
        return ResponseAssessment(
            None,
            (Finding("MALFORMED_RESPONSE", FORMAT_FINDING, str(exc)),),
            explanation_checked=False,
        )
    try:
        response = OmegaRecommendationResponse.from_dict(payload)
    except (ValueError, TypeError) as exc:
        findings = [Finding("SCHEMA_VIOLATION", FORMAT_FINDING, str(exc))]
        explanation = payload.get("explanation")
        checked = isinstance(explanation, str) and bool(explanation.strip())
        if checked:
            findings.extend(explanation_findings(request, explanation))
        return ResponseAssessment(None, tuple(findings), explanation_checked=checked)
    return ResponseAssessment(response, tuple(response_findings(request, response)))


def validate_response_for_request(
    request: RecommendationRequest,
    response: OmegaRecommendationResponse,
) -> None:
    """Raise with every finding unless the response may be accepted."""

    findings = response_findings(request, response)
    if findings:
        raise ResponseValidationError(findings)


def response_findings(
    request: RecommendationRequest,
    response: OmegaRecommendationResponse,
) -> list[Finding]:
    """Validate identity, decision echo, citations, concerns, and explanation claims."""

    findings: list[Finding] = []
    if response.request_id != request.request_id:
        findings.append(
            Finding(
                "REQUEST_ID_MISMATCH",
                FORMAT_FINDING,
                "response request_id does not match the request",
            )
        )

    decision = request.subrep_decision
    if response.selected_skill_id != decision.selected_skill_id:
        findings.append(
            Finding(
                "DECISION_MISMATCH",
                FACTUAL_FINDING,
                "selected_skill_id does not echo the authoritative SubRep decision",
            )
        )
    if response.abstain != decision.abstain:
        findings.append(
            Finding(
                "DECISION_MISMATCH",
                FACTUAL_FINDING,
                "response abstain does not echo the authoritative SubRep decision",
            )
        )

    valid_refs = request.valid_evidence_refs()
    unknown_refs = set(response.evidence_refs).difference(valid_refs)
    if unknown_refs:
        findings.append(
            Finding(
                "UNKNOWN_EVIDENCE_REF",
                FORMAT_FINDING,
                f"response contains unknown evidence_refs: {sorted(unknown_refs)}",
            )
        )
    required_refs = (
        ["subrep_decision.reason_code"]
        if decision.abstain
        else ["subrep_decision.selected_score"]
    )
    if decision.runner_up_skill_id is not None and not decision.abstain:
        required_refs.append("subrep_decision.runner_up_score")
    for required_ref in required_refs:
        if required_ref not in response.evidence_refs:
            findings.append(
                Finding(
                    "MISSING_EVIDENCE_REF",
                    FORMAT_FINDING,
                    f"response must cite {required_ref!r}",
                )
            )

    for concern in response.advisory_concerns:
        findings.extend(_advisory_concern_findings(request, concern, valid_refs))

    findings.extend(explanation_findings(request, response.explanation))
    return findings


def parse_structured_response(raw_response: Any) -> dict:
    """Parse a raw JSON object, accepting only an optional single JSON code fence."""

    if not isinstance(raw_response, str) or not raw_response.strip():
        raise ValueError("backend returned an empty response")
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("response contains an incomplete code fence")
        opening = lines[0].strip().lower()
        if opening not in {"```", "```json"}:
            raise ValueError("only a JSON code fence is accepted")
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("response must be a JSON object")
    return payload


def _advisory_concern_findings(
    request: RecommendationRequest,
    concern: AdvisoryConcern,
    valid_refs: set[str],
) -> list[Finding]:
    unknown_refs = set(concern.evidence_refs).difference(valid_refs)
    if unknown_refs:
        return [
            Finding(
                "UNKNOWN_EVIDENCE_REF",
                FORMAT_FINDING,
                f"advisory concern contains unknown evidence_refs: {sorted(unknown_refs)}",
            )
        ]
    decision = request.subrep_decision
    if decision.abstain:
        return [
            Finding(
                "ADVISORY_NOT_ALLOWED",
                FORMAT_FINDING,
                "advisory concerns are not allowed for an abstaining decision",
            )
        ]
    problem = _unsupported_concern(request, concern)
    if problem is None:
        return []
    return [Finding("UNSUPPORTED_ADVISORY_CONCERN", FACTUAL_FINDING, problem)]


def _unsupported_concern(
    request: RecommendationRequest,
    concern: AdvisoryConcern,
) -> str | None:
    """Return why a concern is not supported by the evidence it cites, if it is not."""

    decision = request.subrep_decision
    selected = next(
        skill for skill in request.admitted_skills if skill.skill_id == decision.selected_skill_id
    )
    refs = set(concern.evidence_refs)
    explanation_prefix = f"explanation_evidence.skills.{selected.skill_id}"
    certificate_prefix = f"audit_only_certificate_evidence.skills.{selected.skill_id}"

    if concern.code == "SMALL_SCORE_GAP":
        if decision.runner_up_skill_id is None or "subrep_decision.score_gap" not in refs:
            return "SMALL_SCORE_GAP must cite subrep_decision.score_gap"
        return None

    if concern.code == "NEGATIVE_OBJECTIVE_TRADEOFF":
        negative_refs = {
            f"{explanation_prefix}.objective_deltas.{objective_id}"
            for objective_id, delta in selected.objective_deltas.items()
            if delta < 0.0
        }
        if not refs.intersection(negative_refs):
            return "NEGATIVE_OBJECTIVE_TRADEOFF must cite a negative selected-skill objective delta"
        return None

    if concern.code == "PDS_EPSILON_USED":
        required = {f"{certificate_prefix}.gate_type", f"{certificate_prefix}.epsilon"}
        if selected.gate_type != "PDS" or selected.epsilon <= 0.0 or not required.issubset(refs):
            return (
                "PDS_EPSILON_USED requires a positive PDS epsilon and its gate_type and epsilon refs"
            )
        return None

    if concern.code == "LOW_CERTIFICATE_MARGIN":
        if f"{certificate_prefix}.certificate_admission_margin" not in refs:
            return "LOW_CERTIFICATE_MARGIN must cite the selected skill's certificate margin"
        return None

    if concern.code == "HIGH_PRIORITY_OBJECTIVE_DECLINE":
        weights = {item.objective_id: item.weight for item in request.objective_weights}
        maximum_weight = max(weights.values())
        supported = any(
            delta < 0.0
            and weights[objective_id] == maximum_weight
            and {
                f"explanation_evidence.objective_weights.{objective_id}",
                f"{explanation_prefix}.objective_deltas.{objective_id}",
            }.issubset(refs)
            for objective_id, delta in selected.objective_deltas.items()
        )
        if not supported:
            return (
                "HIGH_PRIORITY_OBJECTIVE_DECLINE must cite the weight and negative delta of a "
                "highest-weight objective"
            )
    return None

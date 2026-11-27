"""Check an Omega response against the SubRep request it explains."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from .contracts import (
    FACTUAL_FINDING,
    FORMAT_FINDING,
    AdvisoryConcern,
    Claim,
    Finding,
    OmegaRecommendationResponse,
    RecommendationRequest,
)
from .facts import claim_findings, missing_claim_findings, render_explanation, unsupported_concern


class ResponseValidationError(ValueError):
    """Raised with every finding when a response cannot be accepted."""

    def __init__(self, findings: Iterable[Finding]) -> None:
        self.findings = tuple(findings)
        if not self.findings:
            raise ValueError("ResponseValidationError requires at least one finding")
        super().__init__("; ".join(finding.message for finding in self.findings))


@dataclass(frozen=True)
class ResponseAssessment:
    """Every finding for one raw response, plus what could be built from it.

    ``claims_checked`` is false when no claim could be read, so none was verified, which is
    not the same as every claim passing. ``explanation`` is SubRep's explanation, written
    from the verified claims, and is set only for a valid response.
    """

    response: OmegaRecommendationResponse | None
    findings: tuple[Finding, ...]
    claims_checked: bool = True
    explanation: str | None = None

    @property
    def valid(self) -> bool:
        return not self.findings

    @property
    def error(self) -> str | None:
        return "; ".join(finding.message for finding in self.findings) or None


def assess_raw_response(request: RecommendationRequest, raw_response: Any) -> ResponseAssessment:
    """Run every check the raw response allows, and write the explanation if it is valid.

    A response that breaks the schema still has its readable claims verified, so a report
    can tell wrong facts apart from format errors.
    """

    try:
        payload = parse_structured_response(raw_response)
    except (ValueError, TypeError) as exc:
        return ResponseAssessment(
            None,
            (Finding("MALFORMED_RESPONSE", FORMAT_FINDING, str(exc)),),
            claims_checked=False,
        )
    try:
        response = OmegaRecommendationResponse.from_dict(payload)
    except (ValueError, TypeError) as exc:
        findings = [Finding("SCHEMA_VIOLATION", FORMAT_FINDING, str(exc))]
        claims = _readable_claims(payload.get("claims"))
        findings.extend(claim_findings(request, claims))
        return ResponseAssessment(None, tuple(findings), claims_checked=bool(claims))
    findings = tuple(response_findings(request, response))
    explanation = None if findings else render_explanation(request, response)
    return ResponseAssessment(response, findings, claims_checked=True, explanation=explanation)


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
    """Validate identity, the decision echo, every claim, and every advisory concern."""

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

    findings.extend(claim_findings(request, response.claims))
    findings.extend(missing_claim_findings(request, response))

    valid_refs = request.valid_evidence_refs()
    for concern in response.advisory_concerns:
        findings.extend(_advisory_concern_findings(request, concern, valid_refs))
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


def _readable_claims(payload: Any) -> list[Claim]:
    if not isinstance(payload, (list, tuple)):
        return []
    claims = []
    for item in payload:
        try:
            claims.append(Claim.from_dict(item))
        except (ValueError, TypeError):
            continue
    return claims


def _advisory_concern_findings(
    request: RecommendationRequest,
    concern: AdvisoryConcern,
    valid_refs: set[str],
) -> list[Finding]:
    """Every problem with one concern, so a correction prompt can name them all."""

    findings: list[Finding] = []
    unknown_refs = set(concern.evidence_refs).difference(valid_refs)
    if unknown_refs:
        findings.append(
            Finding(
                "UNKNOWN_EVIDENCE_REF",
                FORMAT_FINDING,
                f"advisory concern contains unknown evidence_refs: {sorted(unknown_refs)}",
            )
        )
    if request.subrep_decision.abstain:
        findings.append(
            Finding(
                "ADVISORY_NOT_ALLOWED",
                FORMAT_FINDING,
                "advisory concerns are not allowed for an abstaining decision",
            )
        )
        return findings
    problem = unsupported_concern(request, concern)
    if problem is not None:
        findings.append(Finding("UNSUPPORTED_ADVISORY_CONCERN", FACTUAL_FINDING, problem))
    return findings

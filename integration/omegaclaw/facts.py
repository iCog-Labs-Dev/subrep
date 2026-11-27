"""Verify Omega's claims against the request and write the explanation from verified facts.

Omega returns claims, not prose. Each claim names a fact by its evidence path and states
its value, and SubRep checks the value at that path. The explanation is then written here,
from SubRep's own decision, runner-up, exclusions, and evidence label, plus the verified
claims. Nothing the model writes reaches the reader as text, so a wrong claim cannot be
worded past the checks, and nothing SubRep must state can be left out.
"""

from __future__ import annotations

from math import isclose
from typing import Any, Iterable

from .contracts import (
    CLAIM_ABSOLUTE_TOLERANCE,
    CLAIM_RELATIVE_TOLERANCE,
    FACTUAL_FINDING,
    FORMAT_FINDING,
    LOW_CERTIFICATE_MARGIN_FRACTION,
    SMALL_SCORE_GAP_FRACTION,
    AdvisoryConcern,
    Claim,
    Finding,
    OmegaRecommendationResponse,
    RecommendationRequest,
    claim_type_of,
)

_CERTIFICATE_FIELDS = {
    "gate_type": "gate type",
    "epsilon": "epsilon",
    "certificate_admission_margin": "admission margin",
    "weight_region_type": "weight region",
}
_RISK_BUDGET_FIELDS = {
    "max_certificate_epsilon": "maximum certificate epsilon",
    "minimum_admission_margin": "minimum admission margin",
}


# Verification


def claim_findings(request: RecommendationRequest, claims: Iterable[Claim]) -> list[Finding]:
    """A finding for every claim that cites no claimable fact or states the wrong value."""

    valid_refs = request.valid_evidence_refs()
    findings = []
    for claim in claims:
        reference = claim.evidence_ref
        if reference not in valid_refs:
            findings.append(
                Finding(
                    "UNKNOWN_EVIDENCE_REF",
                    FORMAT_FINDING,
                    f"claim cites {reference!r}, which is not in the request",
                )
            )
            continue
        cited_type = claim_type_of(reference)
        if cited_type is None:
            findings.append(
                Finding(
                    "UNCLAIMABLE_EVIDENCE_REF",
                    FORMAT_FINDING,
                    f"claim cites {reference!r}, which is not a fact a claim can state; "
                    "cite a path from CLAIMABLE_FACTS",
                )
            )
            continue
        if cited_type != claim.type:
            findings.append(
                Finding(
                    "CLAIM_TYPE_MISMATCH",
                    FORMAT_FINDING,
                    f"claim of type {claim.type!r} cites {reference!r}, which holds a "
                    f"{cited_type!r} fact",
                )
            )
            continue
        actual = request.evidence_value(reference)
        if not values_match(claim.value, actual):
            findings.append(
                Finding(
                    "CLAIM_VALUE_MISMATCH",
                    FACTUAL_FINDING,
                    f"claim states {reference} = {_show(claim.value)}, but the request has "
                    f"{_show(actual)}",
                )
            )
    return findings


# The facts that show why one skill scores higher than another.
EXPLAINING_CLAIM_TYPES = frozenset({"objective_weight", "objective_delta"})


def missing_claim_findings(
    request: RecommendationRequest, response: OmegaRecommendationResponse
) -> list[Finding]:
    """A selection must be explained by a weight or delta claim; an abstention needs none.

    Other claims, such as the risk budget alone, would leave the explanation without the
    reason the selected skill scores highest.
    """

    if request.subrep_decision.abstain:
        return []
    if any(claim.type in EXPLAINING_CLAIM_TYPES for claim in response.claims):
        return []
    return [
        Finding(
            "MISSING_CLAIMS",
            FORMAT_FINDING,
            "a selection needs at least one objective_weight or objective_delta claim, which "
            "shows why the selected skill scores highest",
        )
    ]


def values_match(claimed: Any, actual: Any) -> bool:
    """Numbers match up to float rounding, sign included; text matches ignoring case."""

    if isinstance(claimed, bool) or isinstance(actual, bool):
        return claimed is actual
    if isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
        return isclose(
            float(claimed),
            float(actual),
            rel_tol=CLAIM_RELATIVE_TOLERANCE,
            abs_tol=CLAIM_ABSOLUTE_TOLERANCE,
        )
    if isinstance(claimed, str) and isinstance(actual, str):
        return claimed.strip().lower() == actual.strip().lower()
    return claimed is None and actual is None


def unsupported_concern(request: RecommendationRequest, concern: AdvisoryConcern) -> str | None:
    """Why the cited evidence does not support a concern's code, or None when it does."""

    decision = request.subrep_decision
    selected = _selected_skill(request)
    refs = set(concern.evidence_refs)
    explanation_prefix = f"explanation_evidence.skills.{selected.skill_id}"
    certificate_prefix = f"audit_only_certificate_evidence.skills.{selected.skill_id}"

    if concern.code == "SMALL_SCORE_GAP":
        if decision.runner_up_skill_id is None or "subrep_decision.score_gap" not in refs:
            return "SMALL_SCORE_GAP must cite subrep_decision.score_gap"
        limit = SMALL_SCORE_GAP_FRACTION * abs(decision.selected_score)
        if decision.score_gap > limit:
            return (
                f"SMALL_SCORE_GAP needs a gap of at most {SMALL_SCORE_GAP_FRACTION:.0%} of the "
                f"selected score ({_number(limit)}); the gap is {_number(decision.score_gap)}"
            )
        return None

    if concern.code == "NEGATIVE_OBJECTIVE_TRADEOFF":
        if not refs.intersection(_negative_delta_refs(request)):
            return "NEGATIVE_OBJECTIVE_TRADEOFF must cite a negative selected-skill objective delta"
        return None

    if concern.code == "PDS_EPSILON_USED":
        required = {f"{certificate_prefix}.gate_type", f"{certificate_prefix}.epsilon"}
        if selected.gate_type != "PDS" or selected.epsilon <= 0.0 or not required.issubset(refs):
            return (
                "PDS_EPSILON_USED requires a positive PDS epsilon and its gate_type and "
                "epsilon refs"
            )
        return None

    if concern.code == "LOW_CERTIFICATE_MARGIN":
        if f"{certificate_prefix}.certificate_admission_margin" not in refs:
            return "LOW_CERTIFICATE_MARGIN must cite the selected skill's certificate margin"
        minimum = request.risk_budget.minimum_admission_margin
        if minimum is None:
            return "LOW_CERTIFICATE_MARGIN needs a minimum admission margin in the risk budget"
        limit = minimum * (1.0 + LOW_CERTIFICATE_MARGIN_FRACTION)
        if selected.admission_margin > limit:
            return (
                f"LOW_CERTIFICATE_MARGIN needs a margin of at most {_number(limit)} "
                f"({LOW_CERTIFICATE_MARGIN_FRACTION:.0%} above the minimum {_number(minimum)}); "
                f"the margin is {_number(selected.admission_margin)}"
            )
        return None

    # HIGH_PRIORITY_OBJECTIVE_DECLINE
    if _high_priority_declines(request, refs, explanation_prefix):
        return None
    return (
        "HIGH_PRIORITY_OBJECTIVE_DECLINE must cite the weight and negative delta of a "
        "highest-weight objective"
    )


# Writing the explanation


def render_explanation(
    request: RecommendationRequest, response: OmegaRecommendationResponse
) -> str:
    """The explanation for a verified response, written only from verified facts."""

    sentences = _decision_sentences(request)
    sentences.extend(_claim_sentences(request, response.claims))
    sentences.extend(
        f"{exclusion.skill_id} was removed before ranking at the {exclusion.stage} stage: "
        f"{exclusion.stage_description} Recorded reason: {_sentence(exclusion.reason)}"
        for exclusion in request.exclusions
    )
    sentences.extend(_concern_sentences(request, response.advisory_concerns))
    if request.evidence_label == "SYNTHETIC":
        sentences.append("All evidence in this request is synthetic, and no skill was executed.")
    else:
        sentences.append(
            "The values are SubRep certificate evidence, not results of this request, and no "
            "skill was executed."
        )
    return " ".join(sentences)


def _decision_sentences(request: RecommendationRequest) -> list[str]:
    decision = request.subrep_decision
    if decision.abstain:
        return [
            f"SubRep abstained ({decision.reason_code}): no skill was admitted, so it "
            "recommends none."
        ]
    sentences = [
        f"SubRep selected {decision.selected_skill_id}, the admitted skill with the highest "
        f"final selection score ({_number(decision.selected_score)}), where each score is "
        "delta_r plus every objective weight times its objective delta."
    ]
    if decision.runner_up_skill_id is None:
        sentences.append("There is no admitted runner-up.")
    elif decision.score_gap == 0:
        sentences.append(
            f"{decision.runner_up_skill_id} has the same score; SubRep breaks exact ties by "
            "skill ID order, so it is the runner-up."
        )
    else:
        sentences.append(
            f"The admitted runner-up is {decision.runner_up_skill_id} at "
            f"{_number(decision.runner_up_score)}, {_number(decision.score_gap)} lower."
        )
    return sentences


def _claim_sentences(request: RecommendationRequest, claims: Iterable[Claim]) -> list[str]:
    """Sentences for the verified claims, grouped by fact, in the request's own order."""

    claimed = {claim.evidence_ref for claim in claims}
    decision = request.subrep_decision
    sentences = []

    weights = [
        f"{item.objective_id} {_number(item.weight)}"
        for item in request.objective_weights
        if f"explanation_evidence.objective_weights.{item.objective_id}" in claimed
    ]
    if weights:
        sentences.append(f"The objective weights are {_listed(weights)}.")

    ranked = sorted(request.admitted_skills, key=lambda skill: (-skill.score, skill.skill_id))
    for skill in ranked:
        prefix = f"explanation_evidence.skills.{skill.skill_id}"
        deltas = [
            f"{objective} {_signed(delta)}"
            for objective, delta in skill.objective_deltas.items()
            if f"{prefix}.objective_deltas.{objective}" in claimed
        ]
        extras = []
        if f"{prefix}.delta_r" in claimed:
            extras.append(f"delta_r {_number(skill.delta_r)}")
        stated_score = skill.skill_id in (decision.selected_skill_id, decision.runner_up_skill_id)
        if (
            f"admitted_skills.{skill.skill_id}.final_selection_score" in claimed
            and not stated_score
        ):
            extras.append(f"final selection score {_number(skill.score)}")
        if deltas:
            with_extras = f", with {_listed(extras)}" if extras else ""
            sentences.append(
                f"{skill.skill_id} has objective deltas {_listed(deltas)}{with_extras}."
            )
        elif extras:
            sentences.append(f"{skill.skill_id} has {_listed(extras)}.")

    certificates = []
    for skill in ranked:
        prefix = f"audit_only_certificate_evidence.skills.{skill.skill_id}"
        values = {
            "gate_type": skill.gate_type,
            "epsilon": _number(skill.epsilon),
            "certificate_admission_margin": _number(skill.admission_margin),
            "weight_region_type": skill.weight_region_type,
        }
        fields = [
            f"{label} {values[name]}"
            for name, label in _CERTIFICATE_FIELDS.items()
            if f"{prefix}.{name}" in claimed
        ]
        if fields:
            certificates.append(f"{skill.skill_id} has {_listed(fields)}")
    if certificates:
        sentences.append(
            "Certificate evidence, recorded for audit and not used to rank: "
            f"{'; '.join(certificates)}."
        )

    limits = []
    for name, label in _RISK_BUDGET_FIELDS.items():
        if f"risk_budget.{name}" in claimed:
            value = getattr(request.risk_budget, name)
            limits.append(f"no {label}" if value is None else f"a {label} of {_number(value)}")
    if limits:
        sentences.append(f"The local risk budget sets {_listed(limits)}.")
    return sentences


def _concern_sentences(
    request: RecommendationRequest, concerns: Iterable[AdvisoryConcern]
) -> list[str]:
    """One sentence per distinct concern; a decline stated as high priority is not repeated."""

    concerns = list(concerns)
    stated_high = {
        objective
        for concern in concerns
        if concern.code == "HIGH_PRIORITY_OBJECTIVE_DECLINE"
        for objective in _cited_declines(request, concern, highest_only=True)
    }
    sentences: list[str] = []
    for concern in concerns:
        sentence = _concern_sentence(request, concern, skip=stated_high)
        if sentence and sentence not in sentences:
            sentences.append(sentence)
    return sentences


def _cited_declines(
    request: RecommendationRequest, concern: AdvisoryConcern, *, highest_only: bool
) -> list[str]:
    selected = _selected_skill(request)
    weights = {item.objective_id: item.weight for item in request.objective_weights}
    top = max(weights.values())
    cited = set(concern.evidence_refs)
    prefix = f"explanation_evidence.skills.{selected.skill_id}.objective_deltas"
    return [
        objective
        for objective, delta in selected.objective_deltas.items()
        if delta < 0
        and f"{prefix}.{objective}" in cited
        and (not highest_only or weights[objective] == top)
    ]


def _concern_sentence(
    request: RecommendationRequest, concern: AdvisoryConcern, *, skip: set[str]
) -> str | None:
    decision = request.subrep_decision
    selected = _selected_skill(request)
    weights = {item.objective_id: item.weight for item in request.objective_weights}
    if concern.code == "SMALL_SCORE_GAP":
        sentence = (
            f"Advisory note: the gap of {_number(decision.score_gap)} to the runner-up is small"
        )
        if decision.selected_score == 0:
            return sentence + "."
        share = decision.score_gap / abs(decision.selected_score)
        return sentence + f", {share:.0%} of the selected score."
    if concern.code == "PDS_EPSILON_USED":
        return (
            f"Advisory note: {selected.skill_id} is certified by the PDS gate with a positive "
            f"epsilon of {_number(selected.epsilon)}."
        )
    if concern.code == "LOW_CERTIFICATE_MARGIN":
        return (
            f"Advisory note: the certificate admission margin of {selected.skill_id}, "
            f"{_number(selected.admission_margin)}, is close to the local minimum of "
            f"{_number(request.risk_budget.minimum_admission_margin)}."
        )
    if concern.code == "HIGH_PRIORITY_OBJECTIVE_DECLINE":
        declines = _cited_declines(request, concern, highest_only=True)
        return (
            f"Advisory note: {selected.skill_id} lowers "
            + _listed(
                f"{objective} (delta {_signed(selected.objective_deltas[objective])})"
                for objective in declines
            )
            + f", the objective with the highest weight ({_number(max(weights.values()))})."
        )
    # NEGATIVE_OBJECTIVE_TRADEOFF, leaving out declines a high-priority note already states.
    declines = [
        objective
        for objective in _cited_declines(request, concern, highest_only=False)
        if objective not in skip
    ]
    if not declines:
        return None
    return (
        f"Advisory note: {selected.skill_id} lowers "
        + _listed(
            f"{objective} (delta {_signed(selected.objective_deltas[objective])}, weight "
            f"{_number(weights[objective])})"
            for objective in declines
        )
        + "."
    )


# Helpers


def _selected_skill(request: RecommendationRequest):
    return next(
        skill
        for skill in request.admitted_skills
        if skill.skill_id == request.subrep_decision.selected_skill_id
    )


def _negative_delta_refs(request: RecommendationRequest) -> set[str]:
    selected = _selected_skill(request)
    return {
        f"explanation_evidence.skills.{selected.skill_id}.objective_deltas.{objective}"
        for objective, delta in selected.objective_deltas.items()
        if delta < 0.0
    }


def _high_priority_declines(request: RecommendationRequest, refs: set[str], prefix: str) -> bool:
    weights = {item.objective_id: item.weight for item in request.objective_weights}
    top = max(weights.values())
    return any(
        delta < 0.0
        and weights[objective] == top
        and {
            f"explanation_evidence.objective_weights.{objective}",
            f"{prefix}.objective_deltas.{objective}",
        }.issubset(refs)
        for objective, delta in _selected_skill(request).objective_deltas.items()
    )


def _number(value: float) -> str:
    return f"{value:.6g}"


def _signed(value: float) -> str:
    return f"{value:+.6g}"


def _show(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return _number(value)
    return repr(value) if isinstance(value, str) else str(value)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def _listed(items: Iterable[str]) -> str:
    items = list(items)
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]

"""Claim verification and the explanation SubRep writes from verified claims."""

from __future__ import annotations

import json

import pytest

from integration.omegaclaw.contracts import (
    FACTUAL_FINDING,
    FORMAT_FINDING,
    INTEGRATION_MODE,
    RESPONSE_SCHEMA_VERSION,
    SELECTION_BASIS,
    RecommendationRequest,
)
from integration.omegaclaw.scenarios import predefined_scenarios
from integration.omegaclaw.service import SubRepOmegaRecommendationService
from integration.omegaclaw.validation import assess_raw_response

SAFETY_DELTA = "explanation_evidence.skills.skill_safety.objective_deltas.safety"
EFFICIENCY_DELTA = "explanation_evidence.skills.skill_safety.objective_deltas.efficiency"


def _scenario_request(name: str) -> RecommendationRequest:
    scenario = next(item for item in predefined_scenarios() if item.name == name)
    service = SubRepOmegaRecommendationService(adapter=None)
    return service.build_request_from_library(
        **scenario.service_arguments(), evidence_label="SYNTHETIC", request_id=f"facts-{name}"
    )


def _raw(request: RecommendationRequest, claims, **changes) -> str:
    decision = request.subrep_decision
    payload = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "request_id": request.request_id,
        "selected_skill_id": decision.selected_skill_id,
        "abstain": decision.abstain,
        "mode": INTEGRATION_MODE,
        "selection_basis": SELECTION_BASIS,
        "claims": claims,
        "advisory_concerns": [],
    }
    payload.update(changes)
    return json.dumps(payload)


def _claim(claim_type: str, evidence_ref: str, value) -> dict:
    return {"type": claim_type, "evidence_ref": evidence_ref, "value": value}


def _codes(request: RecommendationRequest, claims, **changes) -> dict[str, str]:
    assessment = assess_raw_response(request, _raw(request, claims, **changes))
    return {finding.code: finding.category for finding in assessment.findings}


def _stage_ref(request: RecommendationRequest, skill_id: str) -> str:
    index = next(i for i, item in enumerate(request.exclusions) if item.skill_id == skill_id)
    return f"exclusions.{index}.stage"


# Incorrect claims. Each was a wrong statement the old wording checks could miss or
# misread; as a claim it is compared with the value at its path, whatever the wording.

WRONG_CLAIMS = [
    pytest.param(
        "clear_preferred_skill",
        _claim("objective_delta", EFFICIENCY_DELTA, 0.1),
        id="delta-without-its-minus-sign",
    ),
    pytest.param(
        "clear_preferred_skill",
        _claim("objective_delta", EFFICIENCY_DELTA, 0.8),
        id="delta-of-the-other-objective",
    ),
    pytest.param(
        "clear_preferred_skill",
        _claim(
            "objective_delta",
            "explanation_evidence.skills.skill_efficiency.objective_deltas.safety",
            0.8,
        ),
        id="delta-of-the-other-skill",
    ),
    pytest.param(
        "clear_preferred_skill",
        _claim("objective_weight", "explanation_evidence.objective_weights.safety", 0.2),
        id="swapped-weight",
    ),
    pytest.param(
        "clear_preferred_skill",
        _claim("score", "admitted_skills.skill_safety.final_selection_score", 0.9),
        id="wrong-score",
    ),
    pytest.param(
        "clear_preferred_skill",
        _claim("score", "subrep_decision.runner_up_score", 0.82),
        id="selected-score-given-to-the-runner-up",
    ),
    pytest.param(
        "clear_preferred_skill",
        _claim("score_gap", "subrep_decision.score_gap", 0.6),
        id="wrong-gap",
    ),
    pytest.param(
        "mixed_exclusions",
        _claim("delta_r", "explanation_evidence.skills.skill_safety.delta_r", 0.8),
        id="wrong-delta-r",
    ),
    pytest.param(
        "mixed_exclusions",
        _claim(
            "certificate", "audit_only_certificate_evidence.skills.skill_safety.gate_type", "PDS"
        ),
        id="wrong-certificate-gate",
    ),
    pytest.param(
        "mixed_exclusions",
        _claim("certificate", "audit_only_certificate_evidence.skills.skill_safety.epsilon", 0.05),
        id="wrong-certificate-epsilon",
    ),
    pytest.param(
        "mixed_exclusions",
        _claim("risk_budget", "risk_budget.max_certificate_epsilon", 0.05),
        id="wrong-risk-budget",
    ),
]


@pytest.mark.parametrize(("scenario", "claim"), WRONG_CLAIMS)
def test_a_claim_with_the_wrong_value_is_rejected_as_a_factual_error(scenario, claim):
    valid = _claim("objective_delta", SAFETY_DELTA, 0.8)

    assert _codes(_scenario_request(scenario), [valid, claim]) == {
        "CLAIM_VALUE_MISMATCH": FACTUAL_FINDING
    }


@pytest.mark.parametrize(
    ("skill_id", "stage"),
    [
        pytest.param(
            "skill_efficiency",
            "CERTIFICATION",
            id="operator-exclusion-called-a-certification-failure",
        ),
        pytest.param(
            "rejected_candidate",
            "OPERATOR",
            id="certification-failure-called-an-operator-exclusion",
        ),
        pytest.param(
            "skill_fast_pds", "CERTIFICATION", id="risk-budget-called-a-certification-failure"
        ),
        pytest.param("skill_regional", "RISK_BUDGET", id="runtime-called-risk-budget"),
        pytest.param(
            "skill_three_objective", "RUNTIME_ADMISSIBILITY", id="objective-count-called-runtime"
        ),
    ],
)
def test_an_exclusion_claimed_at_the_wrong_stage_is_rejected(skill_id, stage):
    request = _scenario_request("mixed_exclusions")
    claim = _claim("exclusion_stage", _stage_ref(request, skill_id), stage)
    valid = _claim("objective_delta", SAFETY_DELTA, 0.8)

    assert _codes(request, [valid, claim]) == {"CLAIM_VALUE_MISMATCH": FACTUAL_FINDING}


@pytest.mark.parametrize(
    ("claim", "code"),
    [
        pytest.param(
            _claim(
                "objective_delta",
                "explanation_evidence.skills.ghost_skill.objective_deltas.safety",
                0.8,
            ),
            "UNKNOWN_EVIDENCE_REF",
            id="skill-that-is-not-in-the-request",
        ),
        pytest.param(
            _claim("objective_delta", "explanation_evidence.objective_weights.safety", 0.8),
            "CLAIM_TYPE_MISMATCH",
            id="type-does-not-match-the-path",
        ),
        pytest.param(
            _claim("score", "task_context.scenario", "clear preferred skill"),
            "UNCLAIMABLE_EVIDENCE_REF",
            id="path-that-holds-no-claimable-fact",
        ),
        pytest.param(
            _claim("score", "subrep_decision.selected_skill_id", "skill_safety"),
            "UNCLAIMABLE_EVIDENCE_REF",
            id="decision-field-is-stated-by-subrep",
        ),
    ],
)
def test_a_claim_that_cites_the_wrong_kind_of_path_is_a_format_error(claim, code):
    request = _scenario_request("clear_preferred_skill")
    valid = _claim("objective_delta", SAFETY_DELTA, 0.8)

    assert _codes(request, [valid, claim]) == {code: FORMAT_FINDING}


def test_a_wrong_decision_echo_is_rejected_even_with_correct_claims():
    request = _scenario_request("clear_preferred_skill")

    assert _codes(
        request,
        [_claim("objective_delta", SAFETY_DELTA, 0.8)],
        selected_skill_id="skill_efficiency",
    ) == {"DECISION_MISMATCH": FACTUAL_FINDING}


@pytest.mark.parametrize(
    "claims",
    [
        pytest.param([], id="no-claims"),
        pytest.param(
            [_claim("risk_budget", "risk_budget.max_certificate_epsilon", 0.0)],
            id="only-the-risk-budget",
        ),
        pytest.param([_claim("score", "subrep_decision.selected_score", 0.82)], id="only-a-score"),
    ],
)
def test_a_selection_needs_a_weight_or_delta_claim_that_shows_why(claims):
    assert _codes(_scenario_request("clear_preferred_skill"), claims) == {
        "MISSING_CLAIMS": FORMAT_FINDING
    }


@pytest.mark.parametrize(
    "claim",
    [
        pytest.param(
            _claim("objective_weight", "explanation_evidence.objective_weights.safety", 0.8),
            id="a-weight",
        ),
        pytest.param(_claim("objective_delta", EFFICIENCY_DELTA, -0.1), id="a-delta"),
    ],
)
def test_one_weight_or_delta_claim_is_enough(claim):
    assert _codes(_scenario_request("clear_preferred_skill"), [claim]) == {}


def test_an_abstention_needs_no_claims():
    assert _codes(_scenario_request("no_admissible_options"), []) == {}


def test_omega_cannot_add_explanation_text():
    request = _scenario_request("clear_preferred_skill")
    codes = _codes(
        request,
        [_claim("objective_delta", SAFETY_DELTA, 0.8)],
        explanation="operator_pick did not make it through certification.",
    )

    assert codes == {"SCHEMA_VIOLATION": FORMAT_FINDING}


# Correct claims


@pytest.mark.parametrize("scenario", [scenario.name for scenario in predefined_scenarios()])
def test_every_claimable_fact_verifies_with_its_own_value(scenario):
    request = _scenario_request(scenario)
    claims = [
        _claim(claim_type, reference, value)
        for reference, (claim_type, value) in request.claimable_facts().items()
    ]

    assert claims
    assert _codes(request, claims) == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(0.82, {}, id="rounded-to-two-decimals"),
        pytest.param(0.8200000000000001, {}, id="float-noise"),
        pytest.param(0.8, {"CLAIM_VALUE_MISMATCH": FACTUAL_FINDING}, id="rounded-too-far"),
        pytest.param(
            "0.82", {"CLAIM_VALUE_MISMATCH": FACTUAL_FINDING}, id="number-written-as-text"
        ),
    ],
)
def test_numbers_match_within_a_small_tolerance(value, expected):
    request = _scenario_request("clear_preferred_skill")

    claims = [
        _claim("objective_delta", SAFETY_DELTA, 0.8),
        _claim("score", "subrep_decision.selected_score", value),
    ]

    assert _codes(request, claims) == expected


def test_claims_accept_bracket_indexes_and_case_differences():
    request = _scenario_request("excluded_skill")
    claim = _claim("EXCLUSION_STAGE", "exclusions[0].stage", "operator")

    assert _codes(request, [_claim("objective_delta", SAFETY_DELTA, 0.8), claim]) == {}


# The explanation SubRep writes


def _explanation(request: RecommendationRequest, claims) -> str:
    assessment = assess_raw_response(request, _raw(request, claims))
    assert assessment.valid, assessment.error
    return assessment.explanation


def test_the_explanation_states_the_decision_runner_up_and_verified_claims():
    request = _scenario_request("clear_preferred_skill")
    text = _explanation(
        request,
        [
            _claim("objective_weight", "explanation_evidence.objective_weights.safety", 0.8),
            _claim("objective_weight", "explanation_evidence.objective_weights.efficiency", 0.2),
            _claim("objective_delta", SAFETY_DELTA, 0.8),
            _claim("objective_delta", EFFICIENCY_DELTA, -0.1),
        ],
    )

    assert "SubRep selected skill_safety" in text
    assert "(0.82)" in text and "0.8200000000000001" not in text
    assert "The admitted runner-up is skill_efficiency at 0.28, 0.54 lower." in text
    assert "The objective weights are safety 0.8 and efficiency 0.2." in text
    assert "skill_safety has objective deltas safety +0.8 and efficiency -0.1." in text
    assert "skill_efficiency has objective deltas" not in text  # not claimed, not stated
    assert text.endswith("All evidence in this request is synthetic, and no skill was executed.")


def test_the_explanation_states_every_exclusion_by_its_own_stage_without_being_asked():
    request = _scenario_request("mixed_exclusions")
    text = _explanation(request, [_claim("objective_delta", SAFETY_DELTA, 0.8)])

    for exclusion in request.exclusions:
        assert (
            f"{exclusion.skill_id} was removed before ranking at the {exclusion.stage} stage: "
            f"{exclusion.stage_description}"
        ) in text
    assert "skill_efficiency was removed before ranking at the OPERATOR stage" in text
    assert "Recorded reason: Certificate epsilon 0.05 exceeds the local maximum 0." in text
    assert "There is no admitted runner-up." in text


def test_the_explanation_of_an_abstention():
    request = _scenario_request("abstention_mixed_reasons")
    text = _explanation(request, [])

    assert text.startswith("SubRep abstained (NO_ADMISSIBLE_SKILLS): no skill was admitted")
    assert "runner-up" not in text
    assert text.count("was removed before ranking") == len(request.exclusions)


def test_delta_r_and_other_scores_join_the_skill_sentence():
    request = _scenario_request("clear_preferred_skill")
    text = _explanation(
        request,
        [
            _claim("objective_delta", SAFETY_DELTA, 0.8),
            _claim("objective_delta", EFFICIENCY_DELTA, -0.1),
            _claim("delta_r", "explanation_evidence.skills.skill_safety.delta_r", 0.2),
            _claim("delta_r", "explanation_evidence.skills.skill_efficiency.delta_r", 0.2),
        ],
    )

    assert (
        "skill_safety has objective deltas safety +0.8 and efficiency -0.1, with delta_r 0.2."
        in text
    )
    assert "skill_efficiency has delta_r 0.2." in text


def test_certificate_and_risk_budget_claims_are_stated_as_audit_evidence():
    request = _scenario_request("mixed_exclusions")
    prefix = "audit_only_certificate_evidence.skills.skill_safety"
    text = _explanation(
        request,
        [
            _claim("objective_delta", SAFETY_DELTA, 0.8),
            _claim("certificate", f"{prefix}.gate_type", "CDS"),
            _claim("certificate", f"{prefix}.epsilon", 0.0),
            _claim("risk_budget", "risk_budget.max_certificate_epsilon", 0.0),
        ],
    )

    assert (
        "Certificate evidence, recorded for audit and not used to rank: skill_safety has gate "
        "type CDS and epsilon 0."
    ) in text
    assert "The local risk budget sets a maximum certificate epsilon of 0." in text


# Edge cases found in review


def _two_objective_request(*skills) -> RecommendationRequest:
    from datetime import datetime, timezone

    from integration.omegaclaw.contracts import ObjectiveWeight, RiskBudget

    epsilon = max(skill.epsilon for skill in skills)
    return RecommendationRequest(
        request_id="facts-edge",
        created_at=datetime.now(timezone.utc).isoformat(),
        task_context={},
        objective_weights=(ObjectiveWeight("a", 0.5), ObjectiveWeight("b", 0.5)),
        risk_budget=RiskBudget(max_certificate_epsilon=epsilon),
        admitted_skills=skills,
        evidence_label="SYNTHETIC",
    )


def _skill(
    skill_id: str, delta_r: float, a: float, b: float, *, epsilon: float = 0.0, gate: str = "CDS"
):
    from integration.omegaclaw.contracts import SkillEvidence

    return SkillEvidence(
        skill_id,
        delta_r + 0.5 * a + 0.5 * b,
        delta_r,
        {"a": a, "b": b},
        gate,
        0.1,
        epsilon,
        "FULL_SIMPLEX",
        evidence_label="SYNTHETIC",
    )


def test_a_small_gap_note_on_a_tie_at_score_zero_does_not_divide_by_zero():
    request = _two_objective_request(_skill("alpha", 0.0, 0.0, 0.0), _skill("beta", 0.0, 0.0, 0.0))
    concern = {"code": "SMALL_SCORE_GAP", "evidence_refs": ["subrep_decision.score_gap"]}
    weight = _claim("objective_weight", "explanation_evidence.objective_weights.a", 0.5)

    assessment = assess_raw_response(request, _raw(request, [weight], advisory_concerns=[concern]))

    assert assessment.valid, assessment.error
    assert "Advisory note: the gap of 0 to the runner-up is small." in assessment.explanation


def test_a_small_value_cannot_pass_as_zero():
    request = _two_objective_request(_skill("alpha", 0.2, 0.4, 0.2, epsilon=5e-7, gate="PDS"))
    claim = _claim("certificate", "audit_only_certificate_evidence.skills.alpha.epsilon", 0.0)
    weight = _claim("objective_weight", "explanation_evidence.objective_weights.a", 0.5)

    assert _codes(request, [weight, claim]) == {"CLAIM_VALUE_MISMATCH": FACTUAL_FINDING}
    assert _codes(request, [weight, {**claim, "value": 5e-7}]) == {}


def test_an_advisory_decline_is_stated_once():
    request = _scenario_request("excluded_skill")
    concerns = [
        {"code": "NEGATIVE_OBJECTIVE_TRADEOFF", "evidence_refs": [EFFICIENCY_DELTA]},
        {
            "code": "HIGH_PRIORITY_OBJECTIVE_DECLINE",
            "evidence_refs": [
                "explanation_evidence.objective_weights.efficiency",
                EFFICIENCY_DELTA,
            ],
        },
        {
            "code": "HIGH_PRIORITY_OBJECTIVE_DECLINE",
            "evidence_refs": [
                "explanation_evidence.objective_weights.efficiency",
                EFFICIENCY_DELTA,
            ],
        },
    ]
    claims = [_claim("objective_delta", EFFICIENCY_DELTA, -0.1)]

    assessment = assess_raw_response(request, _raw(request, claims, advisory_concerns=concerns))

    assert assessment.valid, assessment.error
    assert assessment.explanation.count("Advisory note:") == 1
    assert (
        "skill_safety lowers efficiency (delta -0.1), the objective with the highest weight (0.9)."
        in (assessment.explanation)
    )


def test_the_explanation_for_observed_evidence_does_not_call_it_synthetic():
    from integration.omegaclaw.tests.test_adapter import _request, _response

    text = assess_raw_response(_request(), _response()).explanation

    assert text.endswith(
        "The values are SubRep certificate evidence, not results of this request, and no "
        "skill was executed."
    )
    assert "synthetic" not in text

"""Regression set: explanations that misstate SubRep evidence, and ones that must pass."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from integration.omegaclaw.claims import explanation_findings
from integration.omegaclaw.contracts import (
    EXCLUSION_REASONS,
    FACTUAL_FINDING,
    ObjectiveWeight,
    RecommendationRequest,
    RiskBudget,
    SkillEvidence,
    SkillExclusion,
)
from integration.omegaclaw.scenarios import predefined_scenarios
from integration.omegaclaw.service import SubRepOmegaRecommendationService

# One excluded skill per stage, so each case can name the stage it misstates.
EXCLUDED = {
    "operator_pick": "OPERATOR_EXCLUSION",
    "rejected_candidate": "GATE_FAILED",
    "narrow_region": "NOT_RUNTIME_ADMISSIBLE",
    "risky_pds": "RISK_BUDGET_EXCEEDED",
    "three_objective": "OBJECTIVE_DIMENSION_MISMATCH",
}
SELECTED = "Using synthetic evidence, skill_safety was selected with a score of 0.82"
RANKING = f"{SELECTED}, ahead of skill_efficiency at 0.28."


def _skill(skill_id: str, delta_r: float, safety: float, efficiency: float) -> SkillEvidence:
    score = delta_r + 0.8 * safety + 0.2 * efficiency
    return SkillEvidence(
        skill_id=skill_id,
        score=score,
        delta_r=delta_r,
        objective_deltas={"safety": safety, "efficiency": efficiency},
        gate_type="CDS",
        admission_margin=0.1,
        epsilon=0.0,
        weight_region_type="FULL_SIMPLEX",
        evidence_label="SYNTHETIC",
    )


def _request(*, admitted=None, label: str = "SYNTHETIC") -> RecommendationRequest:
    skills = (
        (_skill("skill_safety", 0.2, 0.8, -0.1), _skill("skill_efficiency", 0.2, -0.1, 0.8))
        if admitted is None
        else admitted
    )
    if label != "SYNTHETIC":
        skills = tuple(
            SkillEvidence(**{**skill.__dict__, "evidence_label": label}) for skill in skills
        )
    return RecommendationRequest(
        request_id="golden-1",
        created_at=datetime.now(timezone.utc).isoformat(),
        task_context={"scenario": "golden regression set"},
        objective_weights=(ObjectiveWeight("safety", 0.8), ObjectiveWeight("efficiency", 0.2)),
        risk_budget=RiskBudget(max_certificate_epsilon=0.0),
        admitted_skills=skills,
        exclusions=tuple(
            SkillExclusion(skill_id, code, "Golden regression exclusion.", evidence_label=label)
            for skill_id, code in EXCLUDED.items()
        ),
        evidence_label=label,
    )


def _codes(explanation: str, request: RecommendationRequest | None = None) -> set[str]:
    findings = explanation_findings(request or _request(), explanation)
    assert all(finding.category == FACTUAL_FINDING for finding in findings)
    return {finding.code for finding in findings}


WRONG_EXPLANATIONS = [
    pytest.param(
        f"{RANKING} operator_pick was excluded during certification.",
        {"WRONG_EXCLUSION_STAGE"},
        id="live-failure-operator-exclusion-during-certification",
    ),
    pytest.param(
        f"{RANKING} operator_pick failed certification.",
        {"WRONG_EXCLUSION_STAGE"},
        id="operator-described-as-failed-certification",
    ),
    pytest.param(
        f"{RANKING} operator_pick was not certified.",
        {"WRONG_EXCLUSION_STAGE"},
        id="operator-described-as-uncertified",
    ),
    pytest.param(
        f"{RANKING} operator_pick was rejected by the CDS gate.",
        {"WRONG_EXCLUSION_STAGE"},
        id="operator-described-as-gate-rejection",
    ),
    pytest.param(
        f"{RANKING} The certification process removed operator_pick.",
        {"WRONG_EXCLUSION_STAGE"},
        id="operator-described-as-removed-by-certification",
    ),
    pytest.param(
        f"{RANKING} rejected_candidate was excluded by the operator.",
        {"WRONG_EXCLUSION_STAGE"},
        id="certification-failure-described-as-operator",
    ),
    pytest.param(
        f"{RANKING} narrow_region failed certification.",
        {"WRONG_EXCLUSION_STAGE"},
        id="runtime-described-as-certification",
    ),
    pytest.param(
        f"{RANKING} narrow_region exceeds the local risk budget.",
        {"WRONG_EXCLUSION_STAGE"},
        id="runtime-described-as-risk-budget",
    ),
    pytest.param(
        f"{RANKING} risky_pds lacks a valid certificate.",
        {"WRONG_EXCLUSION_STAGE"},
        id="risk-budget-described-as-certification",
    ),
    pytest.param(
        f"{RANKING} risky_pds was not runtime admissible.",
        {"WRONG_EXCLUSION_STAGE"},
        id="risk-budget-described-as-runtime",
    ),
    pytest.param(
        f"{RANKING} three_objective failed the gates.",
        {"WRONG_EXCLUSION_STAGE"},
        id="objective-mismatch-described-as-certification",
    ),
    pytest.param(
        f"{RANKING} It was excluded. operator_pick was dropped. It failed certification.",
        {"WRONG_EXCLUSION_STAGE"},
        id="pronoun-carries-the-excluded-skill",
    ),
    pytest.param(
        "Using synthetic evidence, skill_safety was selected with a score of 0.28.",
        {"SCORE_MISMATCH"},
        id="selected-score-swapped-with-runner-up",
    ),
    pytest.param(
        f"{SELECTED}, ahead of skill_efficiency at 0.82.",
        {"SCORE_MISMATCH"},
        id="runner-up-score-swapped-with-selected",
    ),
    pytest.param(
        f"{RANKING} Its certificate margin is 0.37.",
        {"UNSUPPORTED_NUMBER"},
        id="invented-number",
    ),
    pytest.param(
        f"{RANKING} operator_pick scored 0.28 before it was excluded by the operator.",
        {"SCORE_MISMATCH"},
        id="excluded-skill-given-a-score",
    ),
    pytest.param(
        "Using synthetic evidence, SubRep selected skill_efficiency over skill_safety.",
        {"WRONG_SKILL_ROLE"},
        id="other-skill-described-as-selected",
    ),
    pytest.param(
        f"{RANKING} Previously SubRep selected skill_speed.",
        {"UNKNOWN_SKILL"},
        id="skill-from-another-request-described-as-selected",
    ),
    pytest.param(
        # From a live qwen3.5 answer: the prose is right, the pasted field is not.
        f"{SELECTED}, runner-up score 0.28 for efficiency. Response: "
        '{""reason_code"": null, ""runner_up_skill_id"": ""skill_safety""}',
        {"WRONG_SKILL_ROLE"},
        id="live-wrong-runner-up-in-a-pasted-decision-field",
    ),
    pytest.param(
        f"{RANKING} selected_skill_id: skill_efficiency.",
        {"WRONG_SKILL_ROLE"},
        id="wrong-selection-in-a-pasted-decision-field",
    ),
    pytest.param(
        f"{SELECTED}. risky_pds was the runner-up.",
        {"WRONG_SKILL_ROLE"},
        id="excluded-skill-described-as-runner-up",
    ),
    pytest.param(
        f"{RANKING} skill_safety was executed and landed safely.",
        {"EXECUTION_CLAIM"},
        id="invented-execution",
    ),
    pytest.param(
        f"{RANKING} After its deployment, skill_safety reduced risk.",
        {"EXECUTION_CLAIM"},
        id="invented-deployment-outcome",
    ),
    pytest.param(
        f"{RANKING} Real-world results confirm the safety gain.",
        {"SYNTHETIC_PRESENTED_AS_REAL"},
        id="synthetic-described-as-real-world",
    ),
    pytest.param(
        f"{RANKING} The safety gain was measured in deployment.",
        {"SYNTHETIC_PRESENTED_AS_REAL"},
        id="synthetic-described-as-measured",
    ),
    pytest.param(
        "skill_safety was selected with a score of 0.82, ahead of skill_efficiency at 0.28.",
        {"SYNTHETIC_NOT_DISCLOSED"},
        id="synthetic-not-disclosed",
    ),
]


@pytest.mark.parametrize(("explanation", "expected"), WRONG_EXPLANATIONS)
def test_wrong_explanation_is_rejected_for_the_right_reason(explanation, expected):
    assert _codes(explanation) == expected


CORRECT_EXPLANATIONS = [
    pytest.param(
        f"{RANKING} The score gap is 0.54.",
        id="selection-with-runner-up-and-gap",
    ),
    pytest.param(
        f"{RANKING} operator_pick was excluded by an explicit operator instruction, not "
        "because it failed certification.",
        id="operator-exclusion-with-negated-certification",
    ),
    pytest.param(
        f"{RANKING} operator_pick holds a certificate, but the operator excluded it.",
        id="operator-exclusion-of-a-certified-skill",
    ),
    pytest.param(
        f"{RANKING} rejected_candidate failed the SubRep CDS/PDS certification gates. "
        "narrow_region holds a certificate, but the runtime admissibility check did not "
        "admit it for the current weights. risky_pds is runtime admissible, but its "
        "epsilon exceeds the local risk budget. three_objective is certified for a "
        "different number of objectives.",
        id="every-stage-described-correctly",
    ),
    pytest.param(
        f"{RANKING} Safety carries 80% of the weight; its delta of 0.8 contributes 0.64.",
        id="percent-and-score-component",
    ),
    pytest.param(
        f"{RANKING} If efficiency dominated, SubRep would have selected skill_efficiency.",
        id="hypothetical-selection",
    ),
    pytest.param(
        f'{RANKING} "selected_skill_id": "skill_safety", "runner_up_skill_id": "skill_efficiency".',
        id="correct-pasted-decision-fields",
    ),
    pytest.param(
        f"{RANKING} subrep_decision.selected_skill_id is the recommendation field.",
        id="request-field-name-is-not-a-skill",
    ),
    pytest.param(
        f"{RANKING} Nothing has been executed; no skill was deployed.",
        id="negated-execution",
    ),
    pytest.param(
        f"{RANKING} The evidence is synthetic, not observed results or real-world data.",
        id="negated-real-evidence",
    ),
    pytest.param(
        f"{RANKING} risky_pds is above the minimum margin but over the risk budget.",
        id="within-one-limit-outside-another",
    ),
]


@pytest.mark.parametrize("explanation", CORRECT_EXPLANATIONS)
def test_correct_explanation_has_no_findings(explanation):
    assert _codes(explanation) == set()


def test_abstention_must_not_name_a_selection():
    request = _request(admitted=())

    assert _codes(
        "No recommendation in this synthetic scenario: rejected_candidate failed certification.",
        request,
    ) == set()
    assert _codes("In this synthetic scenario skill_safety was selected.", request) == {
        "ABSTENTION_NOT_STATED",
        "UNKNOWN_SKILL",
    }


def test_observed_evidence_needs_no_synthetic_disclosure():
    request = _request(label="OBSERVED")

    assert _codes(
        "skill_safety was selected with a score of 0.82, ahead of skill_efficiency at 0.28.",
        request,
    ) == set()


@pytest.mark.parametrize("described_code", sorted(EXCLUSION_REASONS))
@pytest.mark.parametrize("skill_id", sorted(EXCLUDED))
def test_stage_description_passes_only_for_its_own_stage(skill_id, described_code):
    """SubRep's own description of each stage must read as that stage and no other."""

    actual_stage = EXCLUSION_REASONS[EXCLUDED[skill_id]].stage
    described = EXCLUSION_REASONS[described_code]
    explanation = f"{RANKING} {skill_id} was excluded: {described.description}"

    codes = _codes(explanation)

    if described.stage == actual_stage:
        assert codes == set()
    else:
        assert codes == {"WRONG_EXCLUSION_STAGE"}


def _signed_request() -> RecommendationRequest:
    """skill_safety is selected at 0.82; the runner-up skill_negative scores -0.4."""

    return _request(
        admitted=(_skill("skill_safety", 0.2, 0.8, -0.1), _skill("skill_negative", -0.5, 0.1, 0.1))
    )


def _single_request() -> RecommendationRequest:
    """skill_safety is the only admitted skill, so there is no runner-up."""

    return _request(admitted=(_skill("skill_safety", 0.2, 0.8, -0.1),))


SIGNED_SCORES = [
    pytest.param(
        f"{SELECTED}, ahead of skill_negative at 0.40.",
        {"SCORE_MISMATCH"},
        id="negative-score-written-as-positive",
    ),
    pytest.param(
        "Using synthetic evidence, skill_safety was selected with a score of -0.82, ahead "
        "of skill_negative at -0.40.",
        {"SCORE_MISMATCH", "UNSUPPORTED_NUMBER"},
        id="positive-score-written-as-negative",
    ),
    pytest.param(
        f"{SELECTED}, ahead of skill_negative at -0.40.",
        set(),
        id="both-signs-correct",
    ),
    pytest.param(
        f'{SELECTED}. "selected_score": -0.82, "runner_up_score": -0.4.',
        {"SCORE_MISMATCH", "UNSUPPORTED_NUMBER"},
        id="pasted-selected-score-with-the-wrong-sign",
    ),
    pytest.param(
        f'{SELECTED}. "runner_up_score": 0.4.',
        {"SCORE_MISMATCH"},
        id="pasted-negative-runner-up-score-without-its-sign",
    ),
    pytest.param(
        f'{SELECTED}. "selected_score": 0.82, "runner_up_score": -0.4.',
        set(),
        id="pasted-scores-with-correct-signs",
    ),
]


@pytest.mark.parametrize(("explanation", "expected"), SIGNED_SCORES)
def test_score_signs_are_checked_in_both_directions(explanation, expected):
    assert _codes(explanation, _signed_request()) == expected


SIGNED_NUMBERS = [
    pytest.param(f"{RANKING} Its safety delta is -0.8.", {"UNSUPPORTED_NUMBER"}, id="flipped-delta"),
    pytest.param(f"{RANKING} Its efficiency delta is +0.1.", set(), id="plus-sign-matches-a-positive-value"),
    pytest.param(f"{RANKING} Its efficiency delta is -0.1.", set(), id="correct-negative-delta"),
    pytest.param(f"{RANKING} Efficiency fell by 0.1.", set(), id="unsigned-size-of-a-decline"),
]


@pytest.mark.parametrize(("explanation", "expected"), SIGNED_NUMBERS)
def test_an_explicit_sign_must_match_the_evidence(explanation, expected):
    # +0.1 is allowed only because the admission margin is 0.1 in this request.
    assert _codes(explanation) == expected


PASTED_DECISION_FIELDS = [
    pytest.param(
        f"{RANKING} selected_skill_id: skill_unknown.",
        _request,
        {"WRONG_SKILL_ROLE"},
        id="unknown-selected-skill-id",
    ),
    pytest.param(
        f"{RANKING} runner_up_skill_id: skill_unknown.",
        _request,
        {"WRONG_SKILL_ROLE"},
        id="unknown-runner-up-skill-id",
    ),
    pytest.param(
        f"{RANKING} runner_up_skill_id: null.",
        _request,
        {"WRONG_SKILL_ROLE"},
        id="null-runner-up-when-one-exists",
    ),
    pytest.param(
        f"{SELECTED} as the only admitted skill. runner_up_skill_id: skill_unknown.",
        _single_request,
        {"WRONG_SKILL_ROLE"},
        id="unknown-runner-up-when-there-is-none",
    ),
    pytest.param(
        f"{SELECTED} as the only admitted skill. runner_up_skill_id: operator_pick.",
        _single_request,
        {"WRONG_SKILL_ROLE"},
        id="excluded-skill-as-runner-up-when-there-is-none",
    ),
    pytest.param(
        f"{SELECTED} as the only admitted skill. runner_up_skill_id: null.",
        _single_request,
        set(),
        id="null-runner-up-when-there-is-none",
    ),
    pytest.param(
        f"{RANKING} abstain: true.",
        _request,
        {"DECISION_MISMATCH"},
        id="pasted-abstain-contradicts-the-selection",
    ),
    pytest.param(
        f"{RANKING} score_gap: 0.28.",
        _request,
        {"SCORE_MISMATCH"},
        id="pasted-score-gap-is-the-runner-up-score",
    ),
    pytest.param(
        f"{RANKING} reason_code: no_admissible_skills.",
        _request,
        {"DECISION_MISMATCH"},
        id="pasted-abstention-reason-on-a-selection",
    ),
    pytest.param(
        f'{RANKING} \\"runner_up_skill_id\\": \\"skill_efficiency\\", \\"abstain\\": false.',
        _request,
        set(),
        id="escaped-correct-pasted-fields",
    ),
    pytest.param(
        f"{RANKING} The selected_skill_id is the field SubRep fills.",
        _request,
        set(),
        id="field-named-in-prose-is-not-a-pasted-value",
    ),
]


@pytest.mark.parametrize(("explanation", "build", "expected"), PASTED_DECISION_FIELDS)
def test_pasted_decision_fields_are_compared_with_the_decision(explanation, build, expected):
    assert _codes(explanation, build()) == expected


def test_pasted_decision_fields_in_an_abstention():
    request = _request(admitted=())
    abstained = "No recommendation in this synthetic scenario."

    assert _codes(f"{abstained} selected_skill_id: null, reason_code: no_admissible_skills.", request) == set()
    assert _codes(f"{abstained} selected_skill_id: skill_safety.", request) == {"WRONG_SKILL_ROLE"}
    assert _codes(f"{abstained} abstain: false.", request) == {"DECISION_MISMATCH"}


UNKNOWN_SKILL_ROLES = [
    pytest.param(f"{RANKING} skill_speed was the runner-up.", {"UNKNOWN_SKILL"}, id="unknown-runner-up"),
    pytest.param(
        f"{RANKING} skill_speed was excluded by the operator.",
        {"UNKNOWN_SKILL"},
        id="unknown-excluded-skill",
    ),
    pytest.param(
        f"{RANKING} skill_speed scored 0.37.",
        {"UNKNOWN_SKILL", "UNSUPPORTED_NUMBER"},
        id="unknown-skill-given-a-score",
    ),
    pytest.param(
        f"{RANKING} objective_deltas at 0.8 are evidence fields, not skills.",
        set(),
        id="request-field-is-not-an-unknown-skill",
    ),
]


@pytest.mark.parametrize(("explanation", "expected"), UNKNOWN_SKILL_ROLES)
def test_skills_outside_the_request_are_flagged_in_every_role(explanation, expected):
    assert _codes(explanation) == expected


# Answers from the live minimax/minimax-m3 run that the checks flagged although they
# were correct. They must pass, and the real errors next to them must still be caught.


def _scenario_request(name: str) -> RecommendationRequest:
    scenario = next(item for item in predefined_scenarios() if item.name == name)
    service = SubRepOmegaRecommendationService(adapter=None)
    return service.build_request_from_library(
        **scenario.service_arguments(), evidence_label="SYNTHETIC", request_id="live-regression"
    )


LIVE_RISK_BUDGET_ANSWER = (
    "SubRep selected skill_efficiency with a final selection score of 0.9100000000000001 under "
    "MAX_FINAL_SELECTION_SCORE, ahead of the admitted runner-up skill_safety at 0.19 (score gap "
    "0.7200000000000002). skill_fast_pds was excluded at the RISK_BUDGET stage: it holds a SubRep "
    "certificate and is runtime admissible, but its certificate epsilon 0.05 exceeds the local "
    "maximum 0.0 of this request risk budget; this is not a certification failure. All evidence "
    "is synthetic, and no skill was executed, run, or deployed."
)
LIVE_NEGATED_LIST_ANSWER = (
    "Per the authoritative SubRep decision, skill_efficiency was selected with "
    "subrep_decision.selected_score 0.8200000000000001; the only admitted runner-up is "
    "skill_safety at subrep_decision.runner_up_score 0.28. The decision rests on synthetic "
    "evidence (evidence_label SYNTHETIC), not real, observed, or measured data: objective "
    "weights efficiency 0.8 and safety 0.2. Nothing has been executed."
)


def test_live_answer_quoting_a_number_from_an_exclusion_reason_passes():
    assert _codes(LIVE_RISK_BUDGET_ANSWER, _scenario_request("risk_budget_exclusion")) == set()


def test_a_number_absent_from_the_request_text_is_still_flagged():
    wrong = LIVE_RISK_BUDGET_ANSWER.replace("epsilon 0.05", "epsilon 0.06")

    assert _codes(wrong, _scenario_request("risk_budget_exclusion")) == {"UNSUPPORTED_NUMBER"}


def test_live_answer_negating_a_list_of_real_evidence_words_passes():
    assert _codes(LIVE_NEGATED_LIST_ANSWER, _scenario_request("changed_task_priorities")) == set()


@pytest.mark.parametrize(
    "claim",
    [
        "Observed results confirm the gain.",
        "The evidence is not real. Measured data confirm the gain.",
        "The evidence is synthetic, but real-world results agree.",
    ],
)
def test_synthetic_evidence_called_real_outside_a_negated_list_is_still_flagged(claim):
    explanation = f"{RANKING} {claim}"

    assert "SYNTHETIC_PRESENTED_AS_REAL" in _codes(explanation)

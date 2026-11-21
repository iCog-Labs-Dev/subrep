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
        {"WRONG_SKILL_ROLE"},
        id="skill-from-another-request-described-as-selected",
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
        "WRONG_SKILL_ROLE",
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

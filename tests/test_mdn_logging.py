from __future__ import annotations

import pytest

from utils.mdn_contracts import CandidateSkillRecord
from utils.mdn_logging import build_decision_record, serialize_candidate_skill, serialize_decision_record


def _candidate(skill_id: str, certified: bool = True) -> CandidateSkillRecord:
    return CandidateSkillRecord(
        skill_id=skill_id,
        delta_r=0.5,
        delta_n=(0.2, -0.1),
        is_certified=certified,
        gate_type="CDS",
        metadata={"source": "test"},
    )


def test_build_decision_record_returns_valid_record():
    record = build_decision_record(
        context=(0.1,) * 8,
        alpha=(2.0, 3.0),
        support_values=(0.7, 0.3),
        weights_used=(0.4, 0.6),
        candidate_skills=(_candidate("skill_a"), _candidate("skill_b", certified=False)),
        selected_skill_id="skill_a",
        selected_score=0.55,
        actual_payoff=1.2,
        actual_motives=(0.8, 0.1),
        utility=0.9,
    )

    assert record.selected_skill_id == "skill_a"
    assert record.utility == 0.9
    assert record.schema_version == "1.0"


def test_build_decision_record_rejects_invalid_payload():
    with pytest.raises(ValueError, match="selected_skill_id"):
        build_decision_record(
            context=(0.1,) * 8,
            alpha=(2.0, 3.0),
            support_values=(0.7, 0.3),
            weights_used=(0.4, 0.6),
            candidate_skills=(_candidate("skill_a"),),
            selected_skill_id="missing",
        )


def test_serialize_candidate_skill_preserves_core_fields():
    candidate = _candidate("skill_a")
    payload = serialize_candidate_skill(candidate)

    assert payload["skill_id"] == "skill_a"
    assert payload["gate_type"] == "CDS"
    assert payload["delta_n"] == [0.2, -0.1]


def test_serialize_decision_record_preserves_nested_fields():
    record = build_decision_record(
        context=(0.1,) * 8,
        alpha=(2.0, 3.0),
        support_values=(0.7, 0.3),
        weights_used=(0.4, 0.6),
        candidate_skills=(_candidate("skill_a"),),
        selected_skill_id="skill_a",
        selected_score=0.55,
        actual_payoff=1.2,
        actual_motives=(0.8, 0.1),
        utility=0.9,
    )
    payload = serialize_decision_record(record)

    assert payload["selected_skill_id"] == "skill_a"
    assert payload["schema_version"] == "1.0"
    assert payload["candidate_skills"][0]["skill_id"] == "skill_a"
    assert payload["actual_motives"] == [0.8, 0.1]

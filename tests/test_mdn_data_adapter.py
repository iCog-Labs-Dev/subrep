from __future__ import annotations

import numpy as np

from utils.mdn_data_adapter import record_to_prepared_candidate_outcome, records_to_prepared_candidate_outcomes


def _rollout_record(skill_id: str = "skill_a") -> dict[str, object]:
    return {
        "obs": np.array([0.1] * 8, dtype=np.float32),
        "payoff": 1.7,
        "motives": np.array([0.8, 0.4], dtype=np.float32),
        "skill_id": skill_id,
        "terminated": True,
    }


def test_record_to_prepared_candidate_outcome_maps_current_rollout_fields():
    outcome = record_to_prepared_candidate_outcome(_rollout_record())

    assert outcome.skill_id == "skill_a"
    assert outcome.gate_type == "CDS"
    assert len(outcome.context) == 8
    assert np.allclose(outcome.motives, (0.8, 0.4))


def test_record_to_prepared_candidate_outcome_supports_optional_gate_fields():
    record = _rollout_record()
    record["gate_type"] = "PDS"
    record["epsilon"] = 0.2

    outcome = record_to_prepared_candidate_outcome(record)

    assert outcome.gate_type == "PDS"
    assert outcome.epsilon == 0.2


def test_record_to_prepared_candidate_outcome_rejects_missing_fields():
    try:
        record_to_prepared_candidate_outcome({"obs": np.array([0.1] * 8, dtype=np.float32)})
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing rollout fields")


def test_records_to_prepared_candidate_outcomes_converts_multiple_records():
    outcomes = records_to_prepared_candidate_outcomes((_rollout_record("skill_a"), _rollout_record("skill_b")))

    assert len(outcomes) == 2
    assert outcomes[0].skill_id == "skill_a"
    assert outcomes[1].skill_id == "skill_b"

from __future__ import annotations

import numpy as np

from utils.mdn_data_adapter import (
    candidate_set_directory_to_prepared_candidate_outcomes,
    candidate_set_file_to_prepared_candidate_outcomes,
    record_to_prepared_candidate_outcome,
    records_to_prepared_candidate_outcomes,
)


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


def test_candidate_set_file_to_prepared_candidate_outcomes(tmp_path):
    path = tmp_path / "candidate_set_00001.npz"
    np.savez(
        path,
        context=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_ids=np.array(["ppo_deterministic", "random"]),
        candidate_payoffs=np.array([1.7, 0.4], dtype=np.float32),
        candidate_motives=np.array([[0.8, 0.4], [0.1, 0.2]], dtype=np.float32),
    )

    outcomes = candidate_set_file_to_prepared_candidate_outcomes(path)

    assert len(outcomes) == 2
    assert outcomes[0].context == outcomes[1].context
    assert outcomes[0].skill_id == "ppo_deterministic"
    assert outcomes[1].skill_id == "random"
    assert np.allclose(outcomes[0].motives, (0.8, 0.4))


def test_candidate_set_directory_to_prepared_candidate_outcomes(tmp_path):
    for index in range(2):
        np.savez(
            tmp_path / f"candidate_set_{index:05d}.npz",
            context=np.array([0.1 + index] * 8, dtype=np.float32),
            candidate_skill_ids=np.array(["ppo_deterministic", "random"]),
            candidate_payoffs=np.array([1.7, 0.4], dtype=np.float32),
            candidate_motives=np.array([[0.8, 0.4], [0.1, 0.2]], dtype=np.float32),
        )

    outcomes = candidate_set_directory_to_prepared_candidate_outcomes(tmp_path)

    assert len(outcomes) == 4
    assert len({outcome.context for outcome in outcomes}) == 2


def test_candidate_set_file_requires_multiple_candidates(tmp_path):
    path = tmp_path / "candidate_set_00001.npz"
    np.savez(
        path,
        context=np.array([0.1] * 8, dtype=np.float32),
        candidate_skill_ids=np.array(["only_one"]),
        candidate_payoffs=np.array([1.7], dtype=np.float32),
        candidate_motives=np.array([[0.8, 0.4]], dtype=np.float32),
    )

    try:
        candidate_set_file_to_prepared_candidate_outcomes(path)
    except ValueError as exc:
        assert "at least two candidates" in str(exc)
    else:
        raise AssertionError("Expected candidate-set loader to reject a single-candidate file")

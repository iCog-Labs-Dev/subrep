from __future__ import annotations

import numpy as np
import pytest

from generator.train_mdn_probability_aware_logs import probability_aware_logs_to_training_records
from utils.mdn_contracts import CandidateSkillRecord
from utils.probability_aware_logs import (
    SCHEMA_VERSION,
    epsilon_softmax_candidate_probabilities,
    load_probability_aware_log,
    sample_candidate_index,
    save_probability_aware_log,
    serialize_candidate_records,
    validate_probability_aware_log,
)


def _candidates() -> tuple[CandidateSkillRecord, ...]:
    return (
        CandidateSkillRecord(
            skill_id="ppo_deterministic",
            delta_r=4.0,
            delta_n=(1.0, 0.5),
            is_certified=True,
            gate_type="CDS",
            admission_margin=1.0,
        ),
        CandidateSkillRecord(
            skill_id="random",
            delta_r=-2.0,
            delta_n=(-1.0, -0.5),
            is_certified=False,
            gate_type="CDS",
            admission_margin=-1.0,
        ),
        CandidateSkillRecord(
            skill_id="main_engine",
            delta_r=2.0,
            delta_n=(0.2, 0.8),
            is_certified=True,
            gate_type="CDS",
            admission_margin=0.5,
        ),
    )


def _log_record() -> dict[str, object]:
    candidates = _candidates()
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "context": np.array([0.1] * 8, dtype=np.float32),
        "alpha": np.array([0.6, 0.4], dtype=np.float32),
        "support_values": np.array([1.0, 1.0], dtype=np.float32),
        "weights_used": np.array([0.6, 0.4], dtype=np.float32),
        "selected_candidate_index": np.asarray(2, dtype=np.int32),
        "selected_skill_id": np.asarray("main_engine"),
        "selected_score": np.asarray(2.44, dtype=np.float32),
        "behavior_probability": np.asarray(0.35, dtype=np.float32),
        "actual_payoff": np.asarray(7.0, dtype=np.float32),
        "actual_motives": np.array([3.0, 4.0], dtype=np.float32),
        "metadata": {"behavior_policy": "epsilon_softmax_certified_candidates"},
    }
    record.update(serialize_candidate_records(candidates))
    return record


def test_epsilon_softmax_probabilities_cover_only_certified_candidates():
    probabilities = epsilon_softmax_candidate_probabilities(
        _candidates(),
        np.array([0.5, 0.5], dtype=np.float32),
        epsilon=0.2,
        temperature=1.0,
    )

    assert set(probabilities) == {0, 2}
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert all(value > 0.0 for value in probabilities.values())


def test_sample_candidate_index_returns_logged_probability():
    selected_index, probability = sample_candidate_index(
        {0: 0.25, 2: 0.75},
        rng=np.random.default_rng(1),
    )

    assert selected_index in {0, 2}
    assert probability in {0.25, 0.75}


def test_probability_aware_log_roundtrip(tmp_path):
    path = tmp_path / "runtime_log_00001.npz"
    save_probability_aware_log(path, **_log_record())

    loaded = load_probability_aware_log(path)

    assert str(np.asarray(loaded["selected_skill_id"]).reshape(()).item()) == "main_engine"
    assert loaded["metadata"]["behavior_policy"] == "epsilon_softmax_certified_candidates"


def test_probability_aware_log_rejects_missing_behavior_probability():
    record = _log_record()
    del record["behavior_probability"]

    with pytest.raises(ValueError, match="behavior_probability"):
        validate_probability_aware_log(record)


def test_probability_aware_log_rejects_uncertified_selected_candidate():
    record = _log_record()
    record["selected_candidate_index"] = np.asarray(1, dtype=np.int32)
    record["selected_skill_id"] = np.asarray("random")

    with pytest.raises(ValueError, match="certified"):
        validate_probability_aware_log(record)


def test_probability_aware_logs_convert_to_ips_auxiliary_records():
    decision_records, auxiliary_records = probability_aware_logs_to_training_records([_log_record()])

    assert len(decision_records) == 1
    assert decision_records[0].behavior_probability == pytest.approx(0.35)
    assert len(auxiliary_records) == 3
    selected_records = [record for record in auxiliary_records if record.has_q_target]
    assert len(selected_records) == 1
    assert selected_records[0].behavior_probability == pytest.approx(0.35)
    assert selected_records[0].selected_candidate_index == 1
    assert selected_records[0].candidate_delta_r == pytest.approx((4.0, 2.0))

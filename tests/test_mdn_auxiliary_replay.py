from __future__ import annotations

import pytest

from generator.mdn_auxiliary_replay import (
    AuxiliaryReplayBuffer,
    AuxiliaryReplayEntry,
    replay_entry_to_selected_auxiliary_record,
)


def _entry() -> AuxiliaryReplayEntry:
    return AuxiliaryReplayEntry(
        context=(0.1,) * 8,
        selected_skill_id="skill_a",
        selected_candidate_index=0,
        behavior_probability=0.75,
        actual_payoff=1.7,
        actual_motives=(0.8, 0.4),
        candidate_skill_ids=("skill_a", "skill_b"),
        candidate_accept_labels=(1.0, 0.0),
        candidate_delta_r=(0.7, -0.2),
        candidate_delta_n=((0.3, 0.2), (-0.4, -0.5)),
    )


def test_replay_buffer_appends_and_returns_last_entry():
    buffer = AuxiliaryReplayBuffer(capacity=2)
    entry = _entry()
    buffer.append(entry)

    assert len(buffer) == 1
    assert buffer.last() == entry


def test_replay_buffer_enforces_capacity():
    buffer = AuxiliaryReplayBuffer(capacity=1)
    first = _entry()
    second = AuxiliaryReplayEntry(
        context=(0.2,) * 8,
        selected_skill_id="skill_b",
        selected_candidate_index=0,
        behavior_probability=1.0,
        actual_payoff=1.1,
        actual_motives=(0.3, 0.7),
        candidate_skill_ids=("skill_b",),
        candidate_accept_labels=(1.0,),
        candidate_delta_r=(0.1,),
        candidate_delta_n=((0.1, 0.5),),
    )
    buffer.append(first)
    buffer.append(second)

    assert len(buffer) == 1
    assert buffer.last() == second


def test_replay_entry_to_selected_auxiliary_record_preserves_probability_fields():
    record = replay_entry_to_selected_auxiliary_record(_entry(), num_skills=128)

    assert record.behavior_probability == 0.75
    assert record.selected_candidate_index == 0
    assert record.candidate_delta_r == (0.7, -0.2)
    assert record.q_target == (0.8, 0.4)


def test_replay_entry_to_selected_auxiliary_record_rejects_invalid_num_skills():
    with pytest.raises(ValueError, match="num_skills"):
        replay_entry_to_selected_auxiliary_record(_entry(), num_skills=0)

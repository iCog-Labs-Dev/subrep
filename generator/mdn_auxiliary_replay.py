"""Replay structures for richer proposal-conditioned auxiliary MDN training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import zlib
from typing import Iterable, Optional

from generator.mdn_auxiliary_trainer import AuxiliaryTrainingRecord


@dataclass(frozen=True)
class AuxiliaryReplayEntry:
    """One runtime decision step captured for later auxiliary replay training."""

    context: tuple[float, ...]
    selected_skill_id: str
    selected_candidate_index: int
    behavior_probability: float
    actual_payoff: float
    actual_motives: tuple[float, ...]
    candidate_skill_ids: tuple[str, ...]
    candidate_accept_labels: tuple[float, ...]
    candidate_delta_r: tuple[float, ...]
    candidate_delta_n: tuple[tuple[float, ...], ...]


class AuxiliaryReplayBuffer:
    """Bounded FIFO replay buffer for auxiliary candidate-step records."""

    def __init__(self, capacity: int = 1000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._buffer: deque[AuxiliaryReplayEntry] = deque(maxlen=self.capacity)

    def append(self, entry: AuxiliaryReplayEntry) -> None:
        if not isinstance(entry, AuxiliaryReplayEntry):
            raise ValueError("entry must be AuxiliaryReplayEntry")
        self._buffer.append(entry)

    def extend(self, entries: Iterable[AuxiliaryReplayEntry]) -> None:
        for entry in entries:
            self.append(entry)

    def sample_all(self) -> list[AuxiliaryReplayEntry]:
        return list(self._buffer)

    def last(self) -> Optional[AuxiliaryReplayEntry]:
        if not self._buffer:
            return None
        return self._buffer[-1]

    def __len__(self) -> int:
        return len(self._buffer)


def replay_entry_to_selected_auxiliary_record(entry: AuxiliaryReplayEntry, num_skills: int) -> AuxiliaryTrainingRecord:
    """Convert one replay entry into the selected-skill auxiliary training record.
    """
    if num_skills <= 0:
        raise ValueError("num_skills must be positive")
    skill_id_int = zlib.crc32(entry.selected_skill_id.encode()) % int(num_skills)
    return AuxiliaryTrainingRecord(
        context=entry.context,
        skill_id=skill_id_int,
        accept_label=1.0,
        q_target=entry.actual_motives,
        behavior_probability=entry.behavior_probability,
        candidate_delta_r=entry.candidate_delta_r,
        candidate_delta_n=entry.candidate_delta_n,
        selected_candidate_index=entry.selected_candidate_index,
    )

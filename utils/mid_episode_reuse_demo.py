"""Mid-episode motive-shift reuse demonstration helpers.

This module keeps the demo focused on the existing safety path:
when motive priorities change, the certified library is queried again and only
currently admissible skills remain eligible for reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from library.skill_library import SkillLibrary
from library.skill_selector import select_best_skill_entry


@dataclass(frozen=True)
class MidEpisodeReuseTrace:
    """Audit record for a scripted mid-episode motive-priority shift."""

    shift_step: int
    initial_weight: tuple[float, ...]
    shifted_weight: tuple[float, ...]
    pre_shift_admissible: tuple[str, ...]
    post_shift_admissible: tuple[str, ...]
    selected_after_shift: str | None
    rejected_after_shift: tuple[str, ...]
    retraining_performed: bool = False


def demonstrate_mid_episode_motive_shift(
    library: SkillLibrary,
    *,
    initial_weight: Sequence[float],
    shifted_weight: Sequence[float],
    shift_step: int,
    initial_support_directions: Sequence[Sequence[float]] | None = None,
    initial_support_values: Sequence[float] | None = None,
    shifted_support_directions: Sequence[Sequence[float]] | None = None,
    shifted_support_values: Sequence[float] | None = None,
) -> MidEpisodeReuseTrace:
    """Re-query certified skills after a mid-episode motive shift.

    The helper intentionally delegates all safety logic to
    ``SkillLibrary.query_admissible()``. It does not certify new skills, update
    policies, or retrain anything during the episode.
    """
    if int(shift_step) <= 0:
        raise ValueError("shift_step must be positive")

    initial_w = np.asarray(initial_weight, dtype=np.float64)
    shifted_w = np.asarray(shifted_weight, dtype=np.float64)

    initial_support_directions = (
        shifted_support_directions
        if initial_support_directions is None
        else initial_support_directions
    )
    initial_support_values = (
        shifted_support_values if initial_support_values is None else initial_support_values
    )

    pre_shift = library.query_admissible(
        current_weight=initial_w,
        support_directions=_optional_array(initial_support_directions),
        support_values=_optional_array(initial_support_values),
    )
    post_shift = library.query_admissible(
        current_weight=shifted_w,
        support_directions=_optional_array(shifted_support_directions),
        support_values=_optional_array(shifted_support_values),
    )

    pre_ids = tuple(entry.skill_id for entry in pre_shift)
    post_ids = tuple(entry.skill_id for entry in post_shift)
    selected_after_shift = None
    if post_shift:
        selected_after_shift, _ = select_best_skill_entry(post_shift, shifted_w)

    return MidEpisodeReuseTrace(
        shift_step=int(shift_step),
        initial_weight=tuple(float(v) for v in initial_w),
        shifted_weight=tuple(float(v) for v in shifted_w),
        pre_shift_admissible=pre_ids,
        post_shift_admissible=post_ids,
        selected_after_shift=selected_after_shift,
        rejected_after_shift=tuple(skill_id for skill_id in pre_ids if skill_id not in post_ids),
        retraining_performed=False,
    )


def _optional_array(values):
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)

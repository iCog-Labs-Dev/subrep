"""Translate an executed-skill outcome into MetaMo appraisal inputs.

This closes the loop described in the paper: SubRep consumes MetaMo's weights
and budgets, executes an option, and the outcome feeds back into MetaMo's
appraisal comonad Psi.

MetaMo's `Stimulus` carries four fields (core/state.py:39-51):
    novelty, conduciveness, risk, effort

This module produces those as plain floats in [0, 1]. It does NOT import
MetaMo -- `bridge.governor` builds the actual `Stimulus` object -- so the
mapping is unit-testable without a MetaMo checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .protocol import SkillOutcome


@dataclass(frozen=True)
class StimulusValues:
    """MetaMo appraisal inputs, each in [0, 1]."""

    novelty: float
    conduciveness: float
    risk: float
    effort: float

    def __post_init__(self) -> None:
        for name in ("novelty", "conduciveness", "risk", "effort"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
            object.__setattr__(self, name, value)


def _unit_squash(value: float, scale: float) -> float:
    """Map an unbounded signed quantity into [0, 1] with 0 -> 0.5."""
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    return float(0.5 * (1.0 + np.tanh(value / scale)))


def _positive_squash(value: float, scale: float) -> float:
    """Map a non-negative quantity into [0, 1) with 0 -> 0."""
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    return float(np.tanh(max(0.0, value) / scale))


def build_stimulus_values(
    outcome: SkillOutcome,
    *,
    weights: Optional[np.ndarray] = None,
    payoff_scale: float = 1.0,
    motive_scale: float = 1.0,
    rejection_risk: float = 0.25,
) -> StimulusValues:
    """Derive appraisal inputs from what the executed option actually did.

    Args:
        outcome: The observed skill outcome.
        weights: Optional selection weights used to scalarize delta_n. When
            omitted the unweighted mean is used.
        payoff_scale: Characteristic magnitude of `delta_r`, for squashing.
        motive_scale: Characteristic magnitude of `delta_n` entries.
        rejection_risk: Extra risk attributed to a gate rejection, since a
            rejected option is itself evidence the situation is hazardous.

    Returns:
        StimulusValues with every field in [0, 1].

    Semantics:
        conduciveness -- how well the outcome served current objectives, from
            the scalarized improvement `delta_r + w . delta_n`.
        risk -- driven by the WORST motive movement, matching the CDS/PDS
            gates' own `min_i(delta_n_i)` criterion, plus a rejection penalty.
        novelty -- magnitude of motive movement, i.e. how much the world moved,
            unless the caller supplies a measured value.
        effort -- passed through from the outcome.
    """
    delta_n = np.asarray(outcome.delta_n, dtype=np.float64).reshape(-1)

    if weights is None:
        scalarized_motive = float(np.mean(delta_n))
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.size != delta_n.size:
            raise ValueError(
                f"weights length {w.size} does not match delta_n length {delta_n.size}"
            )
        scalarized_motive = float(np.dot(w, delta_n))

    improvement = float(outcome.delta_r) / payoff_scale + scalarized_motive / motive_scale
    conduciveness = _unit_squash(improvement, scale=2.0)

    # The gates admit on delta_r + min_i(delta_n_i), so the worst component is
    # the right risk signal -- a large average gain can still hide a harmful
    # collapse in one objective.
    worst_motive = float(np.min(delta_n))
    risk = _positive_squash(-worst_motive, scale=motive_scale)
    if not outcome.admitted:
        risk = min(1.0, risk + rejection_risk)

    if outcome.novelty is None:
        novelty = _positive_squash(
            float(np.linalg.norm(delta_n)) / max(1, delta_n.size) ** 0.5,
            scale=motive_scale,
        )
    else:
        novelty = float(np.clip(outcome.novelty, 0.0, 1.0))

    effort = float(np.clip(outcome.effort, 0.0, 1.0))

    return StimulusValues(
        novelty=novelty,
        conduciveness=conduciveness,
        risk=risk,
        effort=effort,
    )

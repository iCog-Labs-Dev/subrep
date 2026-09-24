"""The SubRep-side contract for a motivational governor.

This module deliberately contains no MetaMo import and no fixed objective
count. SubRep talks to `MotivationalGovernor`; only `bridge.governor` knows
that MetaMo exists. A future skill source or a different governor can be
substituted at this seam without touching the rest of SubRep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np

# Simplex membership tolerance for weight vectors.
_SIMPLEX_TOL = 1e-6


@dataclass(frozen=True)
class SkillOutcome:
    """What SubRep observed after executing one option.

    This is the feedback half of the loop: it is translated into a MetaMo
    `Stimulus` and fed back into the appraisal comonad.

    Attributes:
        delta_r: Scalar payoff improvement over baseline.
        delta_n: Motive-feature improvement vector (length m).
        admitted: Whether the option passed the admission gates.
        novelty: Optional externally supplied novelty in [0, 1]. When None
            the stimulus builder derives it from the outcome.
        effort: Normalized execution cost in [0, 1].
    """

    delta_r: float
    delta_n: np.ndarray
    admitted: bool
    novelty: Optional[float] = None
    effort: float = 0.0

    def __post_init__(self) -> None:
        delta_n = np.asarray(self.delta_n, dtype=np.float64).reshape(-1)
        if delta_n.size == 0:
            raise ValueError("delta_n must be non-empty")
        if not np.all(np.isfinite(delta_n)):
            raise ValueError(f"delta_n must be finite, got {delta_n}")
        if not np.isfinite(self.delta_r):
            raise ValueError(f"delta_r must be finite, got {self.delta_r}")
        object.__setattr__(self, "delta_n", delta_n)


@dataclass(frozen=True)
class GovernorSignal:
    """Per-step motivational output consumed by SubRep.

    Attributes:
        weights: Selection weights on the objective simplex (length m,
            non-negative, sums to 1).
        pds_epsilon: PDS budget for this step. Passed to
            `RuntimeCertificationPipeline.certify_skill(epsilon=...)`.
        cvar_tail_level: CVaR lower-tail probability mass for this step,
            in (0, 1]. Passed as `CVaRGate(confidence=...)`.

            NOTE: this is a SCALAR and is NOT the MDN's Dirichlet
            concentration vector (`mdn_alpha`), which is a positive array of
            length m produced by the MDN. The two are different quantities
            that unfortunately share the letter alpha. They must never be
            interchanged; `validate()` enforces the scalar half.
    """

    weights: np.ndarray
    pds_epsilon: float
    cvar_tail_level: float

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "pds_epsilon", float(self.pds_epsilon))
        object.__setattr__(self, "cvar_tail_level", float(self.cvar_tail_level))
        self.validate()

    @property
    def num_objectives(self) -> int:
        return int(self.weights.size)

    def validate(self) -> None:
        """Raise ValueError if this signal is not safe to feed to the gates."""
        w = self.weights
        if w.size == 0:
            raise ValueError("weights must be non-empty")
        if not np.all(np.isfinite(w)):
            raise ValueError(f"weights must be finite, got {w}")
        if np.any(w < -_SIMPLEX_TOL):
            raise ValueError(f"weights must be non-negative, got {w}")
        total = float(np.sum(w))
        if abs(total - 1.0) > _SIMPLEX_TOL:
            raise ValueError(f"weights must sum to 1.0, got {total}")

        if not np.isfinite(self.pds_epsilon) or self.pds_epsilon < 0.0:
            raise ValueError(
                f"pds_epsilon must be finite and non-negative, got {self.pds_epsilon}"
            )

        # Mirrors CVaRGate's own contract (certification/cvar_test.py:20-21),
        # so a bad value fails here rather than deep inside the gate.
        if not (0.0 < self.cvar_tail_level <= 1.0):
            raise ValueError(
                "cvar_tail_level must be a scalar in (0, 1] -- did you pass the "
                f"MDN Dirichlet concentration by mistake? got {self.cvar_tail_level}"
            )


@runtime_checkable
class MotivationalGovernor(Protocol):
    """Emits per-step weights and risk budgets, and consumes outcomes."""

    @property
    def num_objectives(self) -> int:
        """Dimension m of the weight vectors this governor emits."""
        ...

    def signal(self) -> GovernorSignal:
        """Return the current signal without advancing internal state."""
        ...

    def step(self, outcome: SkillOutcome) -> GovernorSignal:
        """Advance the motivational state with feedback, return new signal."""
        ...

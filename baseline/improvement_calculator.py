"""Utilities for computing SubRep improvements over a fixed baseline."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class ImprovementCalculator:
    """Compute Δr and Δn by comparing skill statistics against a baseline."""

    def __init__(self, baseline_stats: Dict[str, Any]) -> None:
        if "baseline_payoff" not in baseline_stats or "baseline_motives" not in baseline_stats:
            raise ValueError("baseline_stats must contain 'baseline_payoff' and 'baseline_motives'")

        self.baseline_payoff = float(baseline_stats["baseline_payoff"])
        self.baseline_motives = np.asarray(baseline_stats["baseline_motives"], dtype=np.float32).reshape(-1)

        if self.baseline_motives.ndim != 1:
            raise ValueError(f"baseline_motives must be 1D, got shape {self.baseline_motives.shape}")
        if not np.isfinite(self.baseline_payoff):
            raise ValueError(f"baseline_payoff must be finite, got {self.baseline_payoff}")
        if not np.all(np.isfinite(self.baseline_motives)):
            raise ValueError(f"baseline_motives must be finite, got {self.baseline_motives}")

    def compute_improvements(self, skill_payoff: float, skill_motives) -> Tuple[float, np.ndarray]:
        """Return payoff and motive improvements relative to the baseline."""
        skill_payoff = float(skill_payoff)
        skill_motives = np.asarray(skill_motives, dtype=np.float32).reshape(-1)

        if skill_motives.shape != self.baseline_motives.shape:
            raise ValueError(
                f"skill_motives shape {skill_motives.shape} does not match baseline {self.baseline_motives.shape}"
            )

        delta_r = skill_payoff - self.baseline_payoff
        delta_n = skill_motives - self.baseline_motives
        self.validate_improvements(delta_r, delta_n)
        return float(delta_r), delta_n.astype(np.float32)

    def validate_improvements(self, delta_r, delta_n) -> None:
        """Validate that computed improvements are finite and shape-compatible."""
        if not np.isscalar(delta_r):
            raise ValueError(f"delta_r must be scalar, got {type(delta_r)}")
        if not isinstance(delta_n, np.ndarray):
            raise ValueError(f"delta_n must be np.ndarray, got {type(delta_n)}")
        if delta_n.ndim != 1:
            raise ValueError(f"delta_n must be 1D vector, got shape {delta_n.shape}")
        if delta_n.shape != self.baseline_motives.shape:
            raise ValueError(
                f"delta_n shape {delta_n.shape} does not match baseline motive shape {self.baseline_motives.shape}"
            )
        if not np.isfinite(delta_r):
            raise ValueError(f"delta_r must be finite, got {delta_r}")
        if not np.all(np.isfinite(delta_n)):
            raise ValueError(f"delta_n must be finite, got {delta_n}")
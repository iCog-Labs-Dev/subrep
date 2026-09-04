"""Per-step risk budgets derived from MetaMo's modulators.

Pure functions over plain floats -- no MetaMo import, no numpy dependency on
MetaMo types -- so this module is fully testable without a MetaMo checkout.
`bridge.governor` extracts the modulator scalars and calls in here.

------------------------------------------------------------------------------
THE SIGN CORRECTION
------------------------------------------------------------------------------
The paper (doc/SubRep-Minecraft-AIRIS_v2.txt:494-495) specifies:

    eps   = eps0 - a1*securing + a3*approach
    alpha = a0   + b1*securing + b2*threshold - b3*approach

SubRep's CVaR gate computes a LOWER-TAIL mean at quantile `confidence`
(certification/cvar_test.py:54-58):

    var_threshold = np.quantile(values, self.confidence)
    tail_values   = values[values <= var_threshold]
    return float(np.mean(tail_values))

so the monotonicity is:

    alpha UP -> shallower tail -> CVaR UP -> easier to admit -> LESS conservative

Under the paper's formulas as written, rising `securing` therefore TIGHTENS the
PDS gate (eps down) while LOOSENING the CVaR gate (alpha up). The two gates
move against each other under the same modulator, which cannot be intended.

Resolution, ratified with the project owner:
  * Keep `confidence = alpha_t` NUMERICALLY. The conventions already agree --
    the paper states alpha = 0.1 (doc:544) and eps0 = 0.10 (doc:534), and
    SubRep defaults to cvar_confidence = 0.1 / pds_epsilon = 0.1
    (utils/mdn_runtime_pipeline.py:97-99). Remapping to `1 - alpha_t` would
    send 0.1 -> 0.9, averaging the worst 90% (essentially the plain mean) and
    silently disabling the gate.
  * FLIP the signs of the b-coefficients so alpha tightens with securing.

Both budgets are written as deviations from the modulator neutral point,
because MetaMo squashes modulators through a sigmoid centred on 0.5 every step
(openpsi/appraisal.py:97) and initialises them to 0.5 (core/engine.py:81).
Using raw values instead of deviations would mean the baseline eps0/alpha0 did
not hold at the neutral state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# MetaMo modulators live in (0, 1) with 0.5 as the neutral fixed point.
MODULATOR_NEUTRAL = 0.5

# CVaRGate's own contract (certification/cvar_test.py:20-21).
CVAR_LEVEL_UPPER_BOUND = 1.0


@dataclass(frozen=True)
class BudgetCoefficients:
    """Coupling gains from modulators to risk budgets.

    Defaults are anchored to the paper:

    * `epsilon_0 = 0.1` matches both the paper's stated baseline (doc:534) and
      SubRep's `RuntimePipelineConfig.pds_epsilon` default.
    * `a1_securing = 0.4` is pinned by the paper's own execution trace and
      reproduces it exactly at neutral approach:
          securing 0.55 -> 0.1 - 0.4*(0.05) = 0.08   (doc:534)
          securing 0.45 -> 0.1 - 0.4*(-0.05) = 0.12  (doc:550)
    * `alpha_0 = 0.1` matches the paper's stated alpha (doc:544) and SubRep's
      `cvar_confidence` default.
    * b-coefficients are positive here and applied with CORRECTED signs in
      `compute_cvar_tail_level` -- see the module docstring.

    The sigmoid at openpsi/appraisal.py:97 compresses modulators toward 0.5, so
    realistic securing spans roughly [0.2, 0.85]; the usable deviation range is
    about +/-0.35 rather than +/-0.5. Gains are sized against that.
    """

    # PDS budget.
    epsilon_0: float = 0.1
    a1_securing: float = 0.4
    a3_approach: float = 0.2
    epsilon_max: float = 0.5

    # CVaR tail level.
    alpha_0: float = 0.1
    b1_securing: float = 0.4
    b2_threshold: float = 0.2
    b3_approach: float = 0.2
    alpha_max: float = 0.5

    # Lower bound on alpha. The CVaR tail holds about alpha * n_samples draws;
    # too few makes the estimate noisy and the gate decision flicker. The
    # effective floor is max(alpha_min_floor, min_tail_samples / n_samples).
    alpha_min_floor: float = 0.02
    min_tail_samples: int = 50

    def __post_init__(self) -> None:
        if self.epsilon_0 < 0.0:
            raise ValueError(f"epsilon_0 must be non-negative, got {self.epsilon_0}")
        if self.epsilon_max < 0.0:
            raise ValueError(f"epsilon_max must be non-negative, got {self.epsilon_max}")
        if not (0.0 < self.alpha_min_floor <= self.alpha_max <= CVAR_LEVEL_UPPER_BOUND):
            raise ValueError(
                "require 0 < alpha_min_floor <= alpha_max <= 1, got "
                f"{self.alpha_min_floor} / {self.alpha_max}"
            )
        if self.min_tail_samples < 1:
            raise ValueError(
                f"min_tail_samples must be >= 1, got {self.min_tail_samples}"
            )


def _deviation(value: float) -> float:
    """Modulator deviation from the neutral point, clamped to the valid range."""
    clamped = min(1.0, max(0.0, float(value)))
    return clamped - MODULATOR_NEUTRAL


def alpha_bounds(
    coefficients: BudgetCoefficients,
    n_samples: int,
) -> Tuple[float, float]:
    """Return the (min, max) admissible CVaR tail level for `n_samples` draws.

    The lower bound is a numerical constraint, not a stylistic one: with
    n_samples=1000 and alpha=0.01 the tail holds only ~10 samples.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    sample_floor = coefficients.min_tail_samples / float(n_samples)
    lower = max(coefficients.alpha_min_floor, sample_floor)
    upper = coefficients.alpha_max
    # A very small sample budget can push the floor above the configured
    # ceiling; keeping the interval non-empty matters more than the ceiling.
    if lower > upper:
        lower = upper
    return lower, upper


def compute_pds_epsilon(
    *,
    securing: float,
    approach: float,
    coefficients: BudgetCoefficients = BudgetCoefficients(),
) -> float:
    """PDS budget for this step.

    Rising `securing` lowers the budget (stricter gate); rising `approach`
    raises it (more willing to trade off). Sign convention matches the paper.
    """
    raw = (
        coefficients.epsilon_0
        - coefficients.a1_securing * _deviation(securing)
        + coefficients.a3_approach * _deviation(approach)
    )
    return float(min(coefficients.epsilon_max, max(0.0, raw)))


def compute_cvar_tail_level(
    *,
    securing: float,
    threshold: float,
    approach: float,
    coefficients: BudgetCoefficients = BudgetCoefficients(),
    n_samples: int = 1000,
) -> float:
    """CVaR lower-tail probability mass for this step.

    Signs are CORRECTED relative to the paper so that this moves coherently
    with `compute_pds_epsilon`: rising `securing` or `threshold` DEEPENS the
    tail (lower alpha, stricter gate), while rising `approach` shallows it.

    Returns a scalar in (0, 1] suitable for `CVaRGate(confidence=...)`. This is
    NOT the MDN Dirichlet concentration vector.
    """
    raw = (
        coefficients.alpha_0
        - coefficients.b1_securing * _deviation(securing)
        - coefficients.b2_threshold * _deviation(threshold)
        + coefficients.b3_approach * _deviation(approach)
    )
    lower, upper = alpha_bounds(coefficients, n_samples)
    return float(min(upper, max(lower, raw)))


def compute_budgets(
    *,
    securing: float,
    threshold: float,
    approach: float,
    coefficients: BudgetCoefficients = BudgetCoefficients(),
    n_samples: int = 1000,
) -> Tuple[float, float]:
    """Return `(pds_epsilon, cvar_tail_level)` for one step."""
    epsilon = compute_pds_epsilon(
        securing=securing,
        approach=approach,
        coefficients=coefficients,
    )
    tail_level = compute_cvar_tail_level(
        securing=securing,
        threshold=threshold,
        approach=approach,
        coefficients=coefficients,
        n_samples=n_samples,
    )
    return epsilon, tail_level

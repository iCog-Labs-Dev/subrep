"""w_meta: project MetaMo's motivational state onto the objective simplex.

MetaMo reasons over 8 goals; SubRep's Minecraft objective vector phi(x) is
6-dimensional. This module owns that projection.

No MetaMo import: callers pass plain arrays, so this is testable standalone.

------------------------------------------------------------------------------
WHY NOT MetaMo's TranslationFunctor
------------------------------------------------------------------------------
`category/functors.py:59` looks like a candidate, but it rejects non-square
matrices (`functors.py:76-79`) because it exists for same-space peer
simulation, not dimensionality reduction. It also returns a MotivationalState,
whose __post_init__ pins G to NUM_GOALS=8 (core/state.py:20-21). Using it for
an 8->6 projection would require changing MetaMo, which this integration
forbids. So the projection lives here instead.

------------------------------------------------------------------------------
CALIBRATION STATUS -- READ BEFORE TRUSTING THE NUMBERS
------------------------------------------------------------------------------
The paper reports three weight vectors (doc:530, 539, 552):

    w0 (dusk, patrol risk)  = [0.35, 0.15, 0.20, 0.20, 0.05, 0.05]
    w1 (patrol appears)     = [0.38, 0.17, 0.18, 0.17, 0.05, 0.05]
    w3 (villagers, trading) = [0.25, 0.25, 0.20, 0.20, 0.05, 0.05]

These CANNOT be reproduced exactly, because the paper never states the goal
vector G at those moments -- only qualitative modulator movement ("securing
increases"). Any matrix fitted to hit them numerically would be inventing G.

So the default matrices below are semantically motivated rather than fitted,
and the test suite asserts what the paper actually determines:
  * the ORDERING of w0 (Safety highest; Sustainability/Infrastructure lowest),
  * the DIRECTION of change (securing up -> Safety weight up; social goals up
    -> Reputation up; w0 -> w3 shifts weight from Safety toward Reputation),
  * simplex invariants and determinism.
Treat the coefficients as a defensible starting point to be tuned against a
real environment, not as reproductions of published values.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# MetaMo goal order -- core/config.py:1-3.
GOAL_NAMES: tuple[str, ...] = (
    "Individuation",
    "Transcendence",
    "Help",
    "Curiosity",
    "Novelty",
    "Self",
    "Ethical",
    "Social",
)

# MetaMo modulator order -- core/config.py:5-7.
MODULATOR_NAMES: tuple[str, ...] = (
    "Valence",
    "Arousal",
    "Approach",
    "Resolution",
    "Threshold",
    "Securing",
)

# SubRep Minecraft objective order -- phi(x), per CLAUDE.md.
OBJECTIVE_NAMES: tuple[str, ...] = (
    "Safety",
    "Reputation",
    "DeadlineSlack",
    "InventoryValue",
    "Sustainability",
    "Infrastructure",
)

MODULATOR_NEUTRAL = 0.5

# Goal -> objective affinity, shape (6 objectives, 8 goals), non-negative.
# Rows are objectives in OBJECTIVE_NAMES order; columns are goals in
# GOAL_NAMES order. Each row reads as "which motives make this objective
# matter".
#                     Ind  Trans Help Curio Novel Self Ethic Soc
DEFAULT_GOAL_AFFINITY = np.array([
    [0.70, 0.00, 0.05, 0.00, 0.00, 0.25, 0.35, 0.00],  # Safety
    [0.00, 0.10, 0.45, 0.00, 0.00, 0.00, 0.30, 0.65],  # Reputation
    [0.15, 0.30, 0.00, 0.00, 0.00, 0.40, 0.00, 0.05],  # DeadlineSlack
    [0.30, 0.00, 0.00, 0.05, 0.05, 0.55, 0.00, 0.00],  # InventoryValue
    [0.05, 0.50, 0.10, 0.00, 0.00, 0.00, 0.40, 0.05],  # Sustainability
    [0.00, 0.35, 0.05, 0.45, 0.35, 0.05, 0.00, 0.00],  # Infrastructure
], dtype=np.float64)

# Modulator -> objective gain, shape (6 objectives, 6 modulators). Applied to
# modulator DEVIATIONS from 0.5, so signs matter and may be negative.
#                     Val   Arou  Appr  Reso  Thre  Secu
DEFAULT_MODULATOR_GAIN = np.array([
    [0.00, 0.10, -0.30, 0.00, 0.45, 0.70],  # Safety
    [0.25, 0.00, 0.30, 0.00, -0.10, -0.15],  # Reputation
    [0.00, 0.45, 0.10, 0.20, 0.00, 0.00],  # DeadlineSlack
    [0.10, 0.00, 0.20, 0.10, 0.00, 0.00],  # InventoryValue
    [0.00, -0.15, 0.00, 0.30, 0.05, 0.10],  # Sustainability
    [0.00, -0.05, 0.30, 0.25, -0.10, -0.05],  # Infrastructure
], dtype=np.float64)

# Softmax sharpness. Lower is peakier. Sized so that the default matrices
# produce a spread comparable to the paper's reported vectors rather than a
# near-uniform simplex point.
DEFAULT_TEMPERATURE = 0.35


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax."""
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    shifted = (scores - np.max(scores)) / temperature
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if not np.isfinite(total) or total <= 0.0:
        # Degenerate only under pathological input; fall back to uniform.
        return np.full(scores.shape, 1.0 / scores.size, dtype=np.float64)
    return exp / total


def w_meta(
    goals: Sequence[float] | np.ndarray,
    modulators: Sequence[float] | np.ndarray,
    *,
    goal_affinity: Optional[np.ndarray] = None,
    modulator_gain: Optional[np.ndarray] = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> np.ndarray:
    """Project a motivational state onto the objective simplex.

    Args:
        goals: Goal intensity vector G, length n_goals.
        modulators: Modulator vector M, length n_modulators, values in [0, 1].
        goal_affinity: (m, n_goals) non-negative affinity matrix.
        modulator_gain: (m, n_modulators) signed gain matrix, applied to
            deviations from the neutral point.
        temperature: Softmax sharpness; lower is peakier.

    Returns:
        Length-m array, non-negative, summing to 1.

    The function is dimension-generic: m is taken from the matrices, so
    swapping in a different objective vector needs no change here.
    """
    affinity = DEFAULT_GOAL_AFFINITY if goal_affinity is None else np.asarray(
        goal_affinity, dtype=np.float64
    )
    gain = DEFAULT_MODULATOR_GAIN if modulator_gain is None else np.asarray(
        modulator_gain, dtype=np.float64
    )

    g = np.asarray(goals, dtype=np.float64).reshape(-1)
    m = np.asarray(modulators, dtype=np.float64).reshape(-1)

    if affinity.ndim != 2:
        raise ValueError(f"goal_affinity must be 2D, got shape {affinity.shape}")
    if gain.ndim != 2:
        raise ValueError(f"modulator_gain must be 2D, got shape {gain.shape}")
    if affinity.shape[0] != gain.shape[0]:
        raise ValueError(
            "goal_affinity and modulator_gain must agree on objective count, got "
            f"{affinity.shape[0]} and {gain.shape[0]}"
        )
    if affinity.shape[1] != g.size:
        raise ValueError(
            f"goal vector length {g.size} does not match goal_affinity "
            f"columns {affinity.shape[1]}"
        )
    if gain.shape[1] != m.size:
        raise ValueError(
            f"modulator vector length {m.size} does not match modulator_gain "
            f"columns {gain.shape[1]}"
        )
    if not np.all(np.isfinite(g)):
        raise ValueError(f"goal vector must be finite, got {g}")
    if not np.all(np.isfinite(m)):
        raise ValueError(f"modulator vector must be finite, got {m}")

    # Modulators are squashed into (0, 1) by MetaMo (openpsi/appraisal.py:97);
    # clamp defensively because raise_boundary_caution writes M on a different
    # path (dynamics/stability.py:77-78).
    m_clamped = np.clip(m, 0.0, 1.0)

    scores = affinity @ g + gain @ (m_clamped - MODULATOR_NEUTRAL)
    return _softmax(scores, temperature)


def describe(weights: np.ndarray) -> str:
    """Human-readable weight breakdown, for logs and demo output."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size == len(OBJECTIVE_NAMES):
        names: Sequence[str] = OBJECTIVE_NAMES
    else:
        names = [f"obj{i}" for i in range(w.size)]
    return "  ".join(f"{name}={value:.3f}" for name, value in zip(names, w))

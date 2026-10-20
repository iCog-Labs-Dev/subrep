"""Designed MetaMo objective mapping for LunarLander.

The coefficients are semantically designed and evaluated with a synthetic
grid search, not fitted to actual LunarLander episode rewards or fuel use.
Revisit and retune them when an end-to-end LunarLander demo is available.
This module reuses shared SubRep interfaces without modifying bridge/.
"""

from __future__ import annotations

import numpy as np

from bridge.weights import w_meta

LUNAR_LANDER_OBJECTIVES = ("Safety", "Fuel")

# Goal columns: Individuation, Transcendence, Help, Curiosity,
# Novelty, Self, Ethical, Social.
LUNAR_LANDER_GOAL_AFFINITY = np.array(
    [
        # Safety: Individuation and Self represent caution-related motives.
        [0.40, 0.00, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00],
        # Fuel: Transcendence and Ethical represent purposeful, responsible use.
        [0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.50, 0.00],
    ],
    dtype=np.float64,
)

# Modulator columns: Valence, Arousal, Approach, Resolution, Threshold, Securing.
LUNAR_LANDER_MODULATOR_GAIN = np.array(
    [
        # Safety: Threshold and Securing increase caution.
        [0.00, 0.00, 0.00, 0.00, 0.25, 0.50],
        # Fuel: Approach increases willingness to pursue progress.
        [0.00, 0.00, 0.40, 0.00, 0.00, 0.00],
    ],
    dtype=np.float64,
)

LUNAR_LANDER_TEMPERATURE = 0.35

# Experimental starting values, not empirically optimized LunarLander values.
LUNAR_LANDER_INITIAL_GOALS = np.array(
    [0.70, 0.40, 0.30, 0.50, 0.40, 0.60, 0.50, 0.30],
    dtype=np.float64,
)


def lunar_lander_weights(
    goals,
    modulators,
    *,
    goal_affinity=None,
    modulator_gain=None,
    temperature=None,
) -> np.ndarray:
    """Return normalized weights in (Safety, Fuel) order."""
    return w_meta(
        goals,
        modulators,
        goal_affinity=(
            LUNAR_LANDER_GOAL_AFFINITY
            if goal_affinity is None
            else goal_affinity
        ),
        modulator_gain=(
            LUNAR_LANDER_MODULATOR_GAIN
            if modulator_gain is None
            else modulator_gain
        ),
        temperature=(
            LUNAR_LANDER_TEMPERATURE if temperature is None else temperature
        ),
    )


def make_lunar_lander_governor(**overrides):
    """Lazily construct a MetaMo governor with the approved mapping."""
    from bridge.governor import MetaMoGovernor

    settings = {
        "goal_affinity": LUNAR_LANDER_GOAL_AFFINITY.copy(),
        "modulator_gain": LUNAR_LANDER_MODULATOR_GAIN.copy(),
        "temperature": LUNAR_LANDER_TEMPERATURE,
        "initial_goals": LUNAR_LANDER_INITIAL_GOALS.copy(),
    }
    settings.update(overrides)
    return MetaMoGovernor(**settings)

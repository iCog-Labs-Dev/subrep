"""Symbolic facts about the MO-LunarLander lander.

Turns the 8-number observation into yes/no facts that plans are written in.

Observation layout (Gymnasium LunarLander): x, y, horizontal speed, vertical
speed, angle, angular speed, left leg contact, right leg contact. Positions are
normalised so the landing pad is centred at (0, 0) and spans |x| <= 0.2.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# Thresholds are in observation units.
UPRIGHT_MAX_ANGLE = 0.1  # radians
UPRIGHT_MAX_SPIN = 0.1
PAD_HALF_WIDTH = 0.2
MAX_DRIFT_SPEED = 0.2  # faster sideways motion carries the lander off the pad
NEAR_GROUND_HEIGHT = 0.3
SLOW_DESCENT_SPEED = 0.2
FAST_FALL_SPEED = 0.4

FACT_NAMES = ("upright", "over_pad", "near_ground", "slow", "falling_fast", "landed")


def lander_facts(observation) -> Dict[str, bool]:
    """Return the symbolic facts that hold for one observation.

    - ``upright``: level and not spinning.
    - ``over_pad``: above the pad and not drifting away from it.
    - ``near_ground``: low enough to touch down.
    - ``slow``: descending (or rising) slowly.
    - ``falling_fast``: falling at or beyond the fast-fall speed.
    - ``landed``: both legs touch the ground.
    """
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    if obs.shape != (8,):
        raise ValueError(f"expected an 8-value LunarLander observation, got shape {obs.shape}")
    x, y, vx, vy, angle, spin, left_leg, right_leg = obs
    return {
        "upright": bool(abs(angle) < UPRIGHT_MAX_ANGLE and abs(spin) < UPRIGHT_MAX_SPIN),
        "over_pad": bool(abs(x) <= PAD_HALF_WIDTH and abs(vx) <= MAX_DRIFT_SPEED),
        "near_ground": bool(y < NEAR_GROUND_HEIGHT),
        "slow": bool(abs(vy) <= SLOW_DESCENT_SPEED),
        "falling_fast": bool(vy <= -FAST_FALL_SPEED),
        "landed": bool(left_leg > 0 and right_leg > 0),
    }

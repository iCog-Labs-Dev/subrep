"""Tests for the MO-LunarLander symbolic facts (env/lunar_facts.py)."""

import numpy as np
import pytest

from env.lunar_facts import (
    FACT_NAMES,
    FAST_FALL_SPEED,
    MAX_DRIFT_SPEED,
    NEAR_GROUND_HEIGHT,
    PAD_HALF_WIDTH,
    SLOW_DESCENT_SPEED,
    UPRIGHT_MAX_ANGLE,
    UPRIGHT_MAX_SPIN,
    lander_facts,
)


def observation(x=0.0, y=1.0, vx=0.0, vy=0.0, angle=0.0, spin=0.0, left=0.0, right=0.0):
    return np.array([x, y, vx, vy, angle, spin, left, right], dtype=np.float32)


def test_every_fact_is_reported_as_a_plain_bool():
    facts = lander_facts(observation())
    assert tuple(facts) == FACT_NAMES
    assert all(type(value) is bool for value in facts.values())


@pytest.mark.parametrize(
    "angle, spin, expected",
    [
        (UPRIGHT_MAX_ANGLE - 0.01, 0.0, True),
        (UPRIGHT_MAX_ANGLE + 0.01, 0.0, False),
        (-UPRIGHT_MAX_ANGLE - 0.01, 0.0, False),
        (0.0, UPRIGHT_MAX_SPIN + 0.01, False),
    ],
)
def test_upright(angle, spin, expected):
    assert lander_facts(observation(angle=angle, spin=spin))["upright"] is expected


@pytest.mark.parametrize(
    "x, vx, expected",
    [
        (PAD_HALF_WIDTH - 0.01, 0.0, True),
        (-PAD_HALF_WIDTH + 0.01, 0.0, True),
        (PAD_HALF_WIDTH + 0.01, 0.0, False),
        (0.0, MAX_DRIFT_SPEED + 0.01, False),
    ],
)
def test_over_pad_needs_position_and_no_drift(x, vx, expected):
    assert lander_facts(observation(x=x, vx=vx))["over_pad"] is expected


def test_near_ground():
    assert lander_facts(observation(y=NEAR_GROUND_HEIGHT - 0.01))["near_ground"] is True
    assert lander_facts(observation(y=NEAR_GROUND_HEIGHT + 0.01))["near_ground"] is False


def test_slow_and_falling_fast():
    assert lander_facts(observation(vy=-SLOW_DESCENT_SPEED + 0.01))["slow"] is True
    assert lander_facts(observation(vy=-SLOW_DESCENT_SPEED - 0.01))["slow"] is False
    assert lander_facts(observation(vy=-FAST_FALL_SPEED - 0.01))["falling_fast"] is True
    assert lander_facts(observation(vy=-FAST_FALL_SPEED + 0.01))["falling_fast"] is False


def test_landed_needs_both_legs():
    assert lander_facts(observation(left=1.0, right=1.0))["landed"] is True
    assert lander_facts(observation(left=1.0, right=0.0))["landed"] is False


def test_rejects_observations_of_the_wrong_size():
    with pytest.raises(ValueError, match="8-value"):
        lander_facts(np.zeros(6))

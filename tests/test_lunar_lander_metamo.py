"""Tests for the designed MetaMo-to-LunarLander objective mapping."""

from __future__ import annotations

import numpy as np
import pytest

from bridge._loader import is_available
from bridge.protocol import SkillOutcome
from env.lunar_lander_metamo import (
    LUNAR_LANDER_GOAL_AFFINITY,
    LUNAR_LANDER_INITIAL_GOALS,
    LUNAR_LANDER_MODULATOR_GAIN,
    LUNAR_LANDER_OBJECTIVES,
    LUNAR_LANDER_TEMPERATURE,
    lunar_lander_weights,
    make_lunar_lander_governor,
)


def neutral_modulators() -> np.ndarray:
    """Return the six neutral modulator values."""
    return np.full(6, 0.5, dtype=np.float64)


def assert_valid_weights(weights: np.ndarray) -> None:
    """Check that weights are a valid distribution over Safety and Fuel."""
    weights = np.asarray(weights, dtype=np.float64)

    assert weights.shape == (2,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    assert np.sum(weights) == pytest.approx(1.0)


def test_mapping_constants_have_expected_shapes_and_order() -> None:
    assert LUNAR_LANDER_OBJECTIVES == ("Safety", "Fuel")
    assert LUNAR_LANDER_GOAL_AFFINITY.shape == (2, 8)
    assert LUNAR_LANDER_MODULATOR_GAIN.shape == (2, 6)
    assert LUNAR_LANDER_INITIAL_GOALS.shape == (8,)
    assert LUNAR_LANDER_TEMPERATURE == pytest.approx(0.35)

    assert np.all(np.isfinite(LUNAR_LANDER_GOAL_AFFINITY))
    assert np.all(np.isfinite(LUNAR_LANDER_MODULATOR_GAIN))
    assert np.all(LUNAR_LANDER_GOAL_AFFINITY >= 0.0)
    assert np.all(LUNAR_LANDER_MODULATOR_GAIN >= 0.0)
    assert np.allclose(LUNAR_LANDER_GOAL_AFFINITY.sum(axis=1), [1.0, 1.0])


def test_neutral_mapping_has_expected_safety_weight() -> None:
    weights = lunar_lander_weights(
        LUNAR_LANDER_INITIAL_GOALS,
        neutral_modulators(),
    )

    assert_valid_weights(weights)
    assert 0.55 <= weights[0] <= 0.65
    assert weights[0] == pytest.approx(0.6325, abs=0.002)


def test_maximum_caution_increases_safety_weight() -> None:
    goals = LUNAR_LANDER_INITIAL_GOALS
    baseline = lunar_lander_weights(goals, neutral_modulators())

    cautious_modulators = neutral_modulators()
    cautious_modulators[4] = 1.0  # Threshold
    cautious_modulators[5] = 1.0  # Securing
    cautious = lunar_lander_weights(goals, cautious_modulators)

    assert_valid_weights(cautious)
    assert cautious[0] > baseline[0]


def test_increased_approach_increases_fuel_weight() -> None:
    goals = LUNAR_LANDER_INITIAL_GOALS
    baseline = lunar_lander_weights(goals, neutral_modulators())

    approach_modulators = neutral_modulators()
    approach_modulators[2] = 1.0  # Approach
    bold = lunar_lander_weights(goals, approach_modulators)

    assert_valid_weights(bold)
    assert bold[1] > baseline[1]


def test_safety_weight_is_monotonic_in_securing() -> None:
    goals = LUNAR_LANDER_INITIAL_GOALS
    securing_values = np.linspace(0.0, 1.0, 11)
    safety_weights = []

    for securing in securing_values:
        modulators = neutral_modulators()
        modulators[5] = securing
        weights = lunar_lander_weights(goals, modulators)
        assert_valid_weights(weights)
        safety_weights.append(weights[0])

    assert np.all(np.diff(safety_weights) >= -1e-12)
    assert safety_weights[-1] > safety_weights[0]


@pytest.mark.parametrize("goal_value", [0.0, 1.0])
@pytest.mark.parametrize("modulator_value", [0.0, 1.0])
def test_extreme_inputs_produce_valid_weights(
    goal_value: float,
    modulator_value: float,
) -> None:
    goals = np.full(8, goal_value, dtype=np.float64)
    modulators = np.full(6, modulator_value, dtype=np.float64)

    weights = lunar_lander_weights(goals, modulators)

    assert_valid_weights(weights)


@pytest.mark.skipif(
    not is_available(),
    reason="MetaMo checkout not found (see bridge/_loader.py for search paths)",
)
def test_governor_emits_two_valid_objective_weights() -> None:
    governor = make_lunar_lander_governor()

    signal = governor.signal()

    assert_valid_weights(signal.weights)


@pytest.mark.skipif(
    not is_available(),
    reason="MetaMo checkout not found (see bridge/_loader.py for search paths)",
)
def test_repeated_threatening_outcomes_raise_safety_weight() -> None:
    governor = make_lunar_lander_governor()
    baseline = governor.signal()

    # This governor emits two objective weights: Safety and Fuel.
    delta_n = np.array([-0.9, 0.0], dtype=np.float64)
    threatening_outcome = SkillOutcome(
        delta_r=-0.5,
        delta_n=delta_n,
        admitted=False,
        effort=0.7,
    )

    for _ in range(3):
        signal = governor.step(threatening_outcome)

    assert_valid_weights(signal.weights)
    assert signal.weights[0] > baseline.weights[0]

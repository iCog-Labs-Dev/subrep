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
def test_governor_uses_approved_mapping_and_initial_goals() -> None:
    governor = make_lunar_lander_governor()

    signal = governor.signal()

    assert signal.weights[0] == pytest.approx(0.6325, abs=1e-3)
    assert signal.weights[1] == pytest.approx(0.3675, abs=1e-3)


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


# ---------------------------------------------------------------------------
# Caution follows the task danger, and recovers
# ---------------------------------------------------------------------------
#
# Synthetic outcomes shaped like the median flown episodes on seeds 1-40,
# against the noop baseline with gamma 1: a ppo_deterministic landing
# (delta_r +200, delta_n [+308, -98], 211 steps) and a left_engine crash
# (delta_r 0, delta_n [-441, -68], 68 steps). They carry the LunarLander task
# risk (env/lunar_lander_rollout.py). The scales are the ppo demo's first-step
# appraisal scales. Assertions use margins well inside the measured values.

requires_metamo = pytest.mark.skipif(
    not is_available(),
    reason="MetaMo checkout not found (see bridge/_loader.py for search paths)",
)

PAYOFF_SCALE = 66.667
MOTIVE_SCALE = 266.672


def _landing() -> SkillOutcome:
    from env.lunar_lander_rollout import TASK_RISK_LANDING

    return SkillOutcome(
        delta_r=200.0, delta_n=np.array([308.0, -98.0]), admitted=True,
        effort=0.211, risk=TASK_RISK_LANDING,
    )


def _crash() -> SkillOutcome:
    from env.lunar_lander_rollout import TASK_RISK_CRASH

    return SkillOutcome(
        delta_r=0.0, delta_n=np.array([-441.0, -68.0]), admitted=True,
        effort=0.068, risk=TASK_RISK_CRASH,
    )


def _trace(outcomes):
    """(securing, eps, safety weight) after each outcome, on a fresh governor."""
    governor = make_lunar_lander_governor(
        payoff_scale=PAYOFF_SCALE, motive_scale=MOTIVE_SCALE
    )
    trace = []
    for outcome in outcomes:
        governor.step(outcome)
        signal = governor.signal()
        trace.append(
            (governor.modulator_summary()["securing"], signal.pds_epsilon, float(signal.weights[0]))
        )
    return trace


@requires_metamo
def test_caution_is_higher_after_crashes_than_after_landings() -> None:
    """Measured after 6 of each: Securing 0.998 vs 0.081, eps 0 vs 0.366."""
    after_crashes = _trace([_crash()] * 6)[-1]
    after_landings = _trace([_landing()] * 6)[-1]

    assert after_crashes[0] > after_landings[0] + 0.5
    assert after_crashes[1] < after_landings[1]
    assert after_crashes[2] > after_landings[2]


@requires_metamo
def test_caution_recovers_after_a_crash_streak() -> None:
    """6 crashes then 12 landings: measured Securing 0.998 -> 0.07, eps 0 -> 0.37."""
    trace = _trace([_crash()] * 6 + [_landing()] * 12)
    peak = max(range(len(trace)), key=lambda i: trace[i][0])
    end = trace[-1]

    assert peak <= 6, "caution should peak during or just after the crashes"
    assert end[0] < trace[peak][0] - 0.5
    assert end[1] > trace[peak][1] + 0.2


@requires_metamo
def test_caution_recovers_after_a_long_crash_streak() -> None:
    """Pinning means never coming back, not being near 1 under sustained threat.

    30 crashes take Securing to about 0.999, which is MetaMo's own behaviour
    under a constant high risk. Measured, 2 landings then bring it down by
    0.42 and 4 by 0.86, and eps ends at 0.37.
    """
    trace = _trace([_crash()] * 30 + [_landing()] * 12)
    after_crashes = trace[29]

    assert after_crashes[1] == pytest.approx(0.0)
    assert min(s for s, _, _ in trace[30:34]) < after_crashes[0] - 0.3
    assert trace[-1][1] > 0.1


@requires_metamo
def test_caution_rises_again_after_relaxing() -> None:
    """From the relaxed state, a crash still tightens the budget at once.

    Measured with real flown outcomes: one crash cuts eps from 0.37 to 0.13.
    """
    trace = _trace([_landing()] * 30 + [_crash()] * 2)
    rested, one_crash, two_crashes = trace[29], trace[30], trace[31]

    assert one_crash[0] > rested[0]
    assert one_crash[1] < 0.5 * rested[1]
    assert two_crashes[1] <= one_crash[1]


@requires_metamo
def test_the_same_outcome_sequence_reproduces_the_same_states() -> None:
    sequence = [_crash()] * 6 + [_landing()] * 12

    def run():
        governor = make_lunar_lander_governor(
            payoff_scale=PAYOFF_SCALE, motive_scale=MOTIVE_SCALE
        )
        states = []
        for outcome in sequence:
            governor.step(outcome)
            signal = governor.signal()
            states.append((
                governor.state.M.copy(), governor.state.G.copy(),
                signal.weights.copy(), signal.pds_epsilon, signal.cvar_tail_level,
            ))
        return states

    for a, b in zip(run(), run()):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])
        assert a[3] == b[3] and a[4] == b[4]


@requires_metamo
def test_the_governor_exposes_the_stimulus_it_was_given() -> None:
    from env.lunar_lander_rollout import TASK_RISK_LANDING

    governor = make_lunar_lander_governor(
        payoff_scale=PAYOFF_SCALE, motive_scale=MOTIVE_SCALE
    )
    assert governor.last_stimulus is None
    governor.step(_landing())
    assert governor.last_stimulus.risk == pytest.approx(TASK_RISK_LANDING)
    governor.step(SkillOutcome(delta_r=0.0, delta_n=np.zeros(2), admitted=False))
    assert governor.last_stimulus.risk == pytest.approx(0.25)

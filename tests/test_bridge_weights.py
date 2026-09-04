"""Tests for the w_meta 8-goal -> 6-objective projection.

Note on scope: the paper reports three weight vectors (doc:530, 539, 552) but
never states the goal vector G at those moments, so they cannot be reproduced
numerically without inventing data. These tests therefore assert what the
paper actually determines -- ordering and direction of change -- plus the
structural invariants the gates rely on.
"""

from __future__ import annotations

import numpy as np
import pytest

from bridge.weights import (
    DEFAULT_GOAL_AFFINITY,
    DEFAULT_MODULATOR_GAIN,
    GOAL_NAMES,
    MODULATOR_NAMES,
    OBJECTIVE_NAMES,
    w_meta,
)

# Index shorthands, per core/config.py:1-7.
G_IND, G_TRANS, G_HELP, G_CURIO, G_NOVEL, G_SELF, G_ETHIC, G_SOC = range(8)
M_VALENCE, M_AROUSAL, M_APPROACH, M_RESOLUTION, M_THRESHOLD, M_SECURING = range(6)

SAFETY, REPUTATION, DEADLINE, INVENTORY, SUSTAIN, INFRA = range(6)

NEUTRAL_M = np.full(6, 0.5)


def balanced_goals() -> np.ndarray:
    return np.full(8, 0.5)


# --------------------------------------------------------------------------
# Structure.
# --------------------------------------------------------------------------


def test_matrix_shapes_match_declared_names():
    assert DEFAULT_GOAL_AFFINITY.shape == (len(OBJECTIVE_NAMES), len(GOAL_NAMES))
    assert DEFAULT_MODULATOR_GAIN.shape == (
        len(OBJECTIVE_NAMES),
        len(MODULATOR_NAMES),
    )


def test_goal_affinity_is_non_negative():
    """Goals contribute to objectives; they never subtract from them."""
    assert np.all(DEFAULT_GOAL_AFFINITY >= 0.0)


def test_output_is_on_the_simplex():
    w = w_meta(balanced_goals(), NEUTRAL_M)
    assert w.shape == (len(OBJECTIVE_NAMES),)
    assert np.all(w >= 0.0)
    assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-9)


def test_is_deterministic():
    g, m = balanced_goals(), NEUTRAL_M
    assert np.allclose(w_meta(g, m), w_meta(g, m))


def test_dimension_generic():
    """m is taken from the matrices -- nothing hard-codes 6."""
    affinity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    gain = np.zeros((2, 2))
    w = w_meta(np.array([1.0, 0.5, 0.5]), np.array([0.5, 0.5]),
               goal_affinity=affinity, modulator_gain=gain)
    assert w.shape == (2,)
    assert float(np.sum(w)) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Direction -- what the paper's narrative actually pins down.
# --------------------------------------------------------------------------


def test_securing_raises_the_safety_weight():
    """doc:528 -- 'securing up' accompanies Safety becoming dominant."""
    calm = w_meta(balanced_goals(), NEUTRAL_M)

    tense_m = NEUTRAL_M.copy()
    tense_m[M_SECURING] = 0.9
    tense_m[M_THRESHOLD] = 0.85
    tense = w_meta(balanced_goals(), tense_m)

    assert tense[SAFETY] > calm[SAFETY]


def test_threat_shifts_weight_away_from_reputation():
    """Under threat, trading matters less than surviving (doc:528-534)."""
    calm = w_meta(balanced_goals(), NEUTRAL_M)

    tense_m = NEUTRAL_M.copy()
    tense_m[M_SECURING] = 0.9
    tense = w_meta(balanced_goals(), tense_m)

    assert tense[REPUTATION] < calm[REPUTATION]
    assert tense[SAFETY] / tense[REPUTATION] > calm[SAFETY] / calm[REPUTATION]


def test_social_goals_raise_reputation():
    """The w0 -> w3 move is Safety down, Reputation up as villagers cluster."""
    base = balanced_goals()
    trading = base.copy()
    trading[G_SOC] = 0.95
    trading[G_HELP] = 0.9

    assert w_meta(trading, NEUTRAL_M)[REPUTATION] > w_meta(base, NEUTRAL_M)[REPUTATION]


def test_individuation_raises_safety():
    base = balanced_goals()
    guarded = base.copy()
    guarded[G_IND] = 0.95

    assert w_meta(guarded, NEUTRAL_M)[SAFETY] > w_meta(base, NEUTRAL_M)[SAFETY]


def test_curiosity_raises_infrastructure():
    base = balanced_goals()
    building = base.copy()
    building[G_CURIO] = 0.95
    building[G_NOVEL] = 0.9

    assert w_meta(building, NEUTRAL_M)[INFRA] > w_meta(base, NEUTRAL_M)[INFRA]


def test_safety_dominates_under_sustained_threat():
    """Ordering assertion: under heavy threat Safety must lead the simplex."""
    tense_m = NEUTRAL_M.copy()
    tense_m[M_SECURING] = 0.95
    tense_m[M_THRESHOLD] = 0.95
    tense_m[M_APPROACH] = 0.1

    goals = balanced_goals()
    goals[G_IND] = 0.8

    w = w_meta(goals, tense_m)
    assert int(np.argmax(w)) == SAFETY


# --------------------------------------------------------------------------
# Validation.
# --------------------------------------------------------------------------


def test_rejects_mismatched_goal_length():
    with pytest.raises(ValueError, match="goal vector length"):
        w_meta(np.full(7, 0.5), NEUTRAL_M)


def test_rejects_mismatched_modulator_length():
    with pytest.raises(ValueError, match="modulator vector length"):
        w_meta(balanced_goals(), np.full(5, 0.5))


def test_rejects_non_finite_input():
    bad = balanced_goals()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        w_meta(bad, NEUTRAL_M)


def test_rejects_non_positive_temperature():
    with pytest.raises(ValueError, match="temperature"):
        w_meta(balanced_goals(), NEUTRAL_M, temperature=0.0)


def test_out_of_range_modulators_are_clamped_not_rejected():
    wild = NEUTRAL_M.copy()
    wild[M_SECURING] = 4.0
    at_bound = NEUTRAL_M.copy()
    at_bound[M_SECURING] = 1.0
    assert np.allclose(w_meta(balanced_goals(), wild),
                       w_meta(balanced_goals(), at_bound))

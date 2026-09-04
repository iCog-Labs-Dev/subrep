"""Regression tests for the MetaMo -> SubRep risk-budget coupling.

The direction tests below are the guard against the defect this integration
exists to fix. The paper's alpha formula (doc:495) has signs that, against
SubRep's lower-tail CVaR (certification/cvar_test.py:54-58), make the CVaR
gate LOOSEN as `securing` rises while the PDS gate TIGHTENS. If anyone ever
"restores" the paper's signs, `test_securing_tightens_both_gates` fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from bridge.budget import (
    MODULATOR_NEUTRAL,
    BudgetCoefficients,
    alpha_bounds,
    compute_budgets,
    compute_cvar_tail_level,
    compute_pds_epsilon,
)

NEUTRAL = MODULATOR_NEUTRAL


# --------------------------------------------------------------------------
# The core invariant: both gates must tighten together.
# --------------------------------------------------------------------------


def test_securing_tightens_both_gates():
    """Rising securing must LOWER epsilon AND LOWER the CVaR tail level.

    Lower epsilon = stricter PDS. Lower tail level = deeper tail = stricter
    CVaR. Under the paper's published signs alpha would rise here, loosening
    the CVaR gate while PDS tightened.
    """
    calm_eps, calm_alpha = compute_budgets(
        securing=0.40, threshold=NEUTRAL, approach=NEUTRAL
    )
    tense_eps, tense_alpha = compute_budgets(
        securing=0.75, threshold=NEUTRAL, approach=NEUTRAL
    )

    assert tense_eps < calm_eps, "rising securing must tighten the PDS budget"
    assert tense_alpha < calm_alpha, (
        "rising securing must tighten the CVaR gate (lower tail level); "
        "if this fails, the paper's alpha signs have been reintroduced"
    )


def test_threshold_tightens_cvar():
    base = compute_cvar_tail_level(
        securing=NEUTRAL, threshold=NEUTRAL, approach=NEUTRAL
    )
    raised = compute_cvar_tail_level(
        securing=NEUTRAL, threshold=0.8, approach=NEUTRAL
    )
    assert raised < base


def test_approach_loosens_both_gates():
    """Approach is the risk-taking modulator: it should relax both gates."""
    base_eps, base_alpha = compute_budgets(
        securing=NEUTRAL, threshold=NEUTRAL, approach=NEUTRAL
    )
    bold_eps, bold_alpha = compute_budgets(
        securing=NEUTRAL, threshold=NEUTRAL, approach=0.85
    )
    assert bold_eps > base_eps
    assert bold_alpha > base_alpha


# --------------------------------------------------------------------------
# Anchors taken from the paper's own execution trace.
# --------------------------------------------------------------------------


def test_neutral_state_reproduces_published_baselines():
    """At the neutral modulator point both budgets sit at the paper's values.

    eps0 = 0.10 (doc:534) and alpha = 0.1 (doc:544), which also match SubRep's
    own defaults (utils/mdn_runtime_pipeline.py:97-99).
    """
    epsilon, tail_level = compute_budgets(
        securing=NEUTRAL, threshold=NEUTRAL, approach=NEUTRAL
    )
    assert epsilon == pytest.approx(0.10, abs=1e-9)
    assert tail_level == pytest.approx(0.10, abs=1e-9)


@pytest.mark.parametrize(
    "securing,expected_epsilon",
    [
        (0.55, 0.08),  # doc:534 -- "eps = 0.08 (from baseline 0.10)"
        (0.45, 0.12),  # doc:550 -- "eps = 0.12 as securing decreases"
    ],
)
def test_paper_epsilon_trace(securing, expected_epsilon):
    """a1 = 0.4 reproduces the paper's reported epsilon values exactly."""
    epsilon = compute_pds_epsilon(securing=securing, approach=NEUTRAL)
    assert epsilon == pytest.approx(expected_epsilon, abs=1e-9)


# --------------------------------------------------------------------------
# Clamping and numerical safety.
# --------------------------------------------------------------------------


def test_epsilon_never_negative():
    """PDSGate rejects a negative epsilon (certification/pds_test.py:40-41)."""
    epsilon = compute_pds_epsilon(securing=1.0, approach=0.0)
    assert epsilon >= 0.0


def test_tail_level_respects_cvar_gate_contract():
    """CVaRGate requires confidence in (0, 1] (certification/cvar_test.py:20)."""
    for securing in np.linspace(0.0, 1.0, 21):
        for approach in (0.0, 0.5, 1.0):
            level = compute_cvar_tail_level(
                securing=float(securing), threshold=0.5, approach=approach
            )
            assert 0.0 < level <= 1.0


def test_alpha_floor_guarantees_enough_tail_samples():
    """The tail must retain enough draws for a stable CVaR estimate.

    With n_samples=1000 the floor keeps at least ~50 samples in the tail;
    without it, alpha could fall to 0.01 and leave only 10.
    """
    coefficients = BudgetCoefficients()
    lower, _ = alpha_bounds(coefficients, n_samples=1000)
    assert lower * 1000 >= coefficients.min_tail_samples

    level = compute_cvar_tail_level(
        securing=1.0, threshold=1.0, approach=0.0, n_samples=1000
    )
    assert level >= lower


def test_alpha_floor_scales_with_sample_budget():
    """A smaller sampling budget must raise the floor, not keep it fixed."""
    coefficients = BudgetCoefficients()
    low_budget, _ = alpha_bounds(coefficients, n_samples=200)
    high_budget, _ = alpha_bounds(coefficients, n_samples=5000)
    assert low_budget > high_budget


def test_modulators_outside_unit_range_are_clamped():
    """Defensive: raise_boundary_caution writes M on a different path."""
    clamped = compute_pds_epsilon(securing=5.0, approach=NEUTRAL)
    at_bound = compute_pds_epsilon(securing=1.0, approach=NEUTRAL)
    assert clamped == pytest.approx(at_bound)


def test_rejects_invalid_coefficients():
    with pytest.raises(ValueError):
        BudgetCoefficients(alpha_min_floor=0.0)
    with pytest.raises(ValueError):
        BudgetCoefficients(epsilon_0=-0.1)
    with pytest.raises(ValueError):
        BudgetCoefficients(alpha_min_floor=0.9, alpha_max=0.1)


# --------------------------------------------------------------------------
# The two alphas must never be interchanged.
# --------------------------------------------------------------------------


def test_cvar_tail_level_is_a_scalar_not_a_concentration_vector():
    """`cvar_tail_level` is a scalar; `mdn_alpha` is a length-m vector.

    They share the letter alpha and nothing else. `CVaRGate.__init__` takes
    the scalar; `CVaRGate.admit` takes the vector.
    """
    level = compute_cvar_tail_level(securing=0.6, threshold=0.6, approach=0.4)
    assert isinstance(level, float)
    assert np.ndim(level) == 0


def test_signal_rejects_a_concentration_vector_as_tail_level():
    """Passing the MDN concentration where the scalar belongs must fail."""
    from bridge.protocol import GovernorSignal

    weights = np.full(6, 1.0 / 6.0)
    mdn_concentration = np.array([2.0, 1.5, 1.0, 1.0, 0.8, 1.2])

    with pytest.raises((ValueError, TypeError)):
        GovernorSignal(
            weights=weights,
            pds_epsilon=0.1,
            cvar_tail_level=mdn_concentration,  # type: ignore[arg-type]
        )


def test_signal_rejects_out_of_range_tail_level():
    from bridge.protocol import GovernorSignal

    weights = np.full(6, 1.0 / 6.0)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            GovernorSignal(
                weights=weights, pds_epsilon=0.1, cvar_tail_level=bad
            )


def test_signal_rejects_non_simplex_weights():
    from bridge.protocol import GovernorSignal

    with pytest.raises(ValueError):
        GovernorSignal(
            weights=np.array([0.5, 0.9]), pds_epsilon=0.1, cvar_tail_level=0.1
        )
    with pytest.raises(ValueError):
        GovernorSignal(
            weights=np.array([-0.2, 1.2]), pds_epsilon=0.1, cvar_tail_level=0.1
        )

"""Adapter tests. These need a MetaMo checkout and skip without one.

MetaMo is resolved by bridge/_loader.py from either the pinned submodule at
external/metamo or a sibling MetaMo-Python checkout.
"""

from __future__ import annotations

import numpy as np
import pytest

from bridge._loader import find_metamo_root, is_available
from bridge.protocol import GovernorSignal, SkillOutcome

pytestmark = pytest.mark.skipif(
    not is_available(),
    reason="MetaMo checkout not found (see bridge/_loader.py for search paths)",
)


@pytest.fixture()
def governor():
    from bridge.governor import MetaMoGovernor

    return MetaMoGovernor()


def calm_outcome(m: int = 6) -> SkillOutcome:
    return SkillOutcome(
        delta_r=0.4,
        delta_n=np.full(m, 0.05),
        admitted=True,
        effort=0.1,
    )


def threatening_outcome(m: int = 6) -> SkillOutcome:
    delta_n = np.full(m, 0.0)
    delta_n[0] = -0.9
    return SkillOutcome(
        delta_r=-0.5, delta_n=delta_n, admitted=False, effort=0.7
    )


def test_loader_finds_a_plausible_checkout():
    root = find_metamo_root()
    assert root is not None
    assert (root / "core" / "config.py").is_file()


def test_initial_signal_matches_published_baselines(governor):
    """At the neutral modulator start, eps and alpha are the paper's values."""
    signal = governor.signal()
    assert signal.pds_epsilon == pytest.approx(0.10, abs=1e-9)
    assert signal.cvar_tail_level == pytest.approx(0.10, abs=1e-9)


def test_signal_is_a_valid_simplex_point(governor):
    signal = governor.signal()
    assert isinstance(signal, GovernorSignal)
    assert signal.num_objectives == governor.num_objectives
    assert float(np.sum(signal.weights)) == pytest.approx(1.0, abs=1e-9)
    assert np.all(signal.weights >= 0.0)


def test_signal_does_not_advance_state(governor):
    before = governor.modulator_summary()
    governor.signal()
    governor.signal()
    assert governor.modulator_summary() == before


def test_step_advances_state_and_returns_valid_signal(governor):
    before = governor.modulator_summary()
    signal = governor.step(threatening_outcome())
    after = governor.modulator_summary()

    assert after != before
    signal.validate()


def test_threat_tightens_both_budgets(governor):
    """The end-to-end version of the budget direction test."""
    baseline = governor.signal()
    for _ in range(3):
        tense = governor.step(threatening_outcome())

    assert tense.pds_epsilon <= baseline.pds_epsilon
    assert tense.cvar_tail_level <= baseline.cvar_tail_level
    assert tense.pds_epsilon < baseline.pds_epsilon or tense.pds_epsilon == 0.0


def test_threat_raises_the_safety_weight(governor):
    baseline = governor.signal()
    for _ in range(3):
        tense = governor.step(threatening_outcome())

    assert tense.weights[0] > baseline.weights[0]


def test_metamo_state_is_replaced_not_mutated(governor):
    """MetaMoPseudoBimonad.step is pure (category/bimonad.py:153-171)."""
    original = governor.state
    original_m = np.array(original.M, copy=True)

    governor.step(threatening_outcome())

    assert np.allclose(original.M, original_m), (
        "the previous MotivationalState was mutated in place; step() is "
        "supposed to return a new state"
    )
    assert governor.state is not original


def test_is_deterministic_for_identical_feedback():
    from bridge.governor import MetaMoGovernor

    a, b = MetaMoGovernor(), MetaMoGovernor()
    for _ in range(4):
        sig_a = a.step(calm_outcome())
        sig_b = b.step(calm_outcome())

    assert np.allclose(sig_a.weights, sig_b.weights)
    assert sig_a.pds_epsilon == pytest.approx(sig_b.pds_epsilon)
    assert sig_a.cvar_tail_level == pytest.approx(sig_b.cvar_tail_level)


def test_rejects_wrong_sized_initial_goals():
    from bridge.governor import MetaMoGovernor

    with pytest.raises(ValueError, match="initial_goals"):
        MetaMoGovernor(initial_goals=np.full(5, 0.5))


def test_rejects_non_positive_scales():
    from bridge.governor import MetaMoGovernor

    with pytest.raises(ValueError, match="payoff_scale"):
        MetaMoGovernor(payoff_scale=0.0)
    with pytest.raises(ValueError, match="motive_scale"):
        MetaMoGovernor(motive_scale=-1.0)


def test_scales_change_appraisal_response():
    """Scale matters: an unscaled governor reacts far more strongly.

    Input choice is deliberate. `risk` is squashed through tanh and then
    clipped to [0, 1], and an un-admitted outcome adds a further +0.25
    (bridge/stimulus.py). Feed a large enough delta and BOTH configurations
    clip to risk == 1.0, `delta_securing` becomes identical, and the two
    governors land on exactly the same modulator -- so the test must use a
    moderate, admitted outcome to observe the scaling at all.
    """
    from bridge.governor import MetaMoGovernor

    moderate = SkillOutcome(
        delta_r=-1.5, delta_n=np.full(6, -2.0), admitted=True, effort=0.3
    )

    unscaled = MetaMoGovernor()
    scaled = MetaMoGovernor(payoff_scale=8.0, motive_scale=8.0)

    unscaled.step(moderate)
    scaled.step(moderate)

    assert (
        unscaled.modulator_summary()["securing"]
        > scaled.modulator_summary()["securing"]
    )
    # The stronger reaction must also propagate into a tighter budget.
    assert unscaled.signal().pds_epsilon < scaled.signal().pds_epsilon


def test_extreme_outcomes_saturate_regardless_of_scale():
    """Documents the ceiling: past a point, scaling stops mattering.

    Both configurations clip risk to 1.0, so they converge. This is a real
    limitation of the tanh+clip stimulus mapping, not a bug -- but it means
    scale tuning only buys resolution in the moderate range.
    """
    from bridge.governor import MetaMoGovernor

    extreme = SkillOutcome(
        delta_r=-8.0, delta_n=np.full(6, -6.0), admitted=False, effort=0.5
    )

    unscaled = MetaMoGovernor()
    scaled = MetaMoGovernor(payoff_scale=10.0, motive_scale=6.0)

    unscaled.step(extreme)
    scaled.step(extreme)

    assert unscaled.modulator_summary()["securing"] == pytest.approx(
        scaled.modulator_summary()["securing"], abs=1e-6
    )

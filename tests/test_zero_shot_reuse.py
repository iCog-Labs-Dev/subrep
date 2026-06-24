"""
test_zero_shot_reuse.py — Tests for the Zero-Shot Reuse Validation Framework.

Validates that certified skills can be safely reused under new motive weights
without retraining, covering both full-simplex and MDN/contextual modes.

Run with:
    python -m pytest tests/test_zero_shot_reuse.py -v
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from certification.certificate_schema import Certificate
from utils.reuse_evaluator import ZeroShotEvaluator, FULL_SIMPLEX, MDN_WX


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_cert(
    skill_id: str = "test-skill",
    delta_r: float = 0.8,
    delta_n: tuple[float, float] = (0.5, 0.3),
    gate_type: str = "CDS",
    epsilon: float = 0.0,
    weight_region_type: str = "FULL_SIMPLEX",
    certification_context: tuple[float, ...] | None = None,
    mdn_alpha: tuple[float, ...] | None = None,
    wx_support_directions: tuple[tuple[float, ...], ...] | None = None,
    wx_support_values: tuple[float, ...] | None = None,
) -> Certificate:
    """Build a certificate with standard audit fields for testing."""
    return Certificate(
        skill_id=skill_id,
        gate_type=gate_type,
        delta_r=delta_r,
        delta_n=delta_n,
        admission_margin=max(0.0, delta_r + min(delta_n)),
        epsilon=epsilon,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="baseline-noop",
        environment="MO-LunarLander-v2",
        episode_length=200,
        version="0.1.0",
        weight_region_type=weight_region_type,
        certification_context=certification_context,
        mdn_alpha=mdn_alpha,
        wx_support_directions=wx_support_directions,
        wx_support_values=wx_support_values,
    )


@pytest.fixture
def evaluator():
    return ZeroShotEvaluator()


@pytest.fixture
def cds_full_simplex_cert():
    """CDS, universally beneficial: Δr=0.8, Δn=(0.5, 0.3)."""
    return _make_cert(skill_id="cds-global", delta_r=0.8, delta_n=(0.5, 0.3))


@pytest.fixture
def pds_full_simplex_cert():
    """PDS, trade-off within budget: Δr=0.5, Δn=(0.8, -0.6), ε=0.1."""
    return _make_cert(
        skill_id="pds-global",
        delta_r=0.5,
        delta_n=(0.8, -0.6),
        gate_type="PDS",
        epsilon=0.1,
    )


@pytest.fixture
def mdn_spec_cert():
    """MDN/contextual cert matching the task spec example: Δn=(-0.2, 0.1)."""
    return _make_cert(
        skill_id="mdn-spec-example",
        delta_r=0.15,
        delta_n=(-0.2, 0.1),
    )


# ── Full-Simplex Reuse Tests ────────────────────────────────────────────────

class TestFullSimplexReuse:
    """CDS/PDS skills certified globally should be safe under ANY valid weight."""

    def test_cds_safe_under_multiple_weights(self, evaluator, cds_full_simplex_cert):
        """Global CDS skill must pass for diverse valid simplex weights."""
        for w in [[0.2, 0.8], [0.5, 0.5], [0.9, 0.1], [1.0, 0.0], [0.0, 1.0]]:
            assert evaluator.is_safe_mathematically(cds_full_simplex_cert, w) is True

    def test_pds_safe_under_multiple_weights(self, evaluator, pds_full_simplex_cert):
        """Global PDS skill must pass for diverse valid simplex weights."""
        for w in [[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]]:
            assert evaluator.is_safe_mathematically(pds_full_simplex_cert, w) is True

    def test_rejects_invalid_weight_sum(self, evaluator, cds_full_simplex_cert):
        """Weights that do not sum to 1 must be rejected."""
        with pytest.raises(ValueError, match="sum to 1"):
            evaluator.is_safe_mathematically(cds_full_simplex_cert, [0.3, 0.3])

    def test_rejects_negative_weight(self, evaluator, cds_full_simplex_cert):
        """Weights with negative components must be rejected."""
        with pytest.raises(ValueError, match=">= 0"):
            evaluator.is_safe_mathematically(cds_full_simplex_cert, [1.5, -0.5])


# ── MDN/Contextual Reuse Tests ──────────────────────────────────────────────

class TestMDNContextualReuse:
    """MDN_WX skills must be checked against the current support geometry."""

    # Standard support descriptor from the task spec.
    SPEC_DIRECTIONS = [[1, 0], [0, 1]]
    SPEC_VALUES = [0.8, 0.4]

    def test_spec_example_cds_passes(self, evaluator):
        """Task spec example: delta_r=0.15 >= h_Wx=0.14 → CDS passes."""
        cert = _make_cert(
            skill_id="spec-cds-pass", delta_r=0.15, delta_n=(-0.2, 0.1),
            weight_region_type="MDN_WX",
            wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
            wx_support_values=(0.8, 0.4),
            certification_context=(0.0, 0.0),
            mdn_alpha=(1.0, 1.0),
        )
        result = evaluator.is_safe_mathematically(
            cert, [0.5, 0.5],
            support_directions=self.SPEC_DIRECTIONS,
            support_values=self.SPEC_VALUES,
        )
        assert result is True

    def test_spec_example_cds_fails(self, evaluator):
        """Task spec example: delta_r=0.10 < h_Wx=0.14 → CDS fails."""
        cert = _make_cert(
            skill_id="spec-cds-fail", delta_r=0.10, delta_n=(-0.2, 0.1),
            weight_region_type="MDN_WX",
            wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
            wx_support_values=(0.8, 0.4),
            certification_context=(0.0, 0.0),
            mdn_alpha=(1.0, 1.0),
        )
        result = evaluator.is_safe_mathematically(
            cert, [0.5, 0.5],
            support_directions=self.SPEC_DIRECTIONS,
            support_values=self.SPEC_VALUES,
        )
        assert result is False

    def test_pds_within_epsilon_passes(self, evaluator):
        """PDS: delta_r=0.12 >= h_Wx(0.14) - epsilon(0.05) = 0.09 → passes."""
        cert = _make_cert(
            skill_id="spec-pds-pass",
            delta_r=0.12,
            delta_n=(-0.2, 0.1),
            gate_type="PDS",
            epsilon=0.05,
            weight_region_type="MDN_WX",
            wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
            wx_support_values=(0.8, 0.4),
            certification_context=(0.0, 0.0),
            mdn_alpha=(1.0, 1.0),
        )
        result = evaluator.is_safe_mathematically(
            cert, [0.5, 0.5],
            support_directions=self.SPEC_DIRECTIONS,
            support_values=self.SPEC_VALUES,
        )
        assert result is True

    def test_rejects_when_cost_too_high(self, evaluator):
        """When h_Wx(-delta_n) greatly exceeds delta_r, reuse must fail."""
        cert = _make_cert(
            skill_id="bad-mdn", delta_r=0.5, delta_n=(-2.0, -1.0),
            weight_region_type="MDN_WX",
            wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
            wx_support_values=(0.8, 0.4),
            certification_context=(0.0, 0.0),
            mdn_alpha=(1.0, 1.0),
        )
        result = evaluator.is_safe_mathematically(
            cert, [0.5, 0.5],
            support_directions=self.SPEC_DIRECTIONS,
            support_values=self.SPEC_VALUES,
        )
        assert result is False

    def test_mdn_skill_unusable_without_geometry(self, evaluator):
        """BLOCKER CASE: Contextual skill used without support context must fail reuse."""
        cert = _make_cert(
            skill_id="mdn-blocker",
            delta_n=(-0.2, 0.1),
            weight_region_type="MDN_WX",
            # Add required audit fields for MDN_WX to pass constructor validation
            certification_context=(1.0, 0.0, 0.0),
            mdn_alpha=(1.0, 1.0),
            wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
            wx_support_values=(0.8, 0.4),
        )
        # Calling is_safe_mathematically without support_directions/values.
        # This used to return True (Bug), now must return False (Fixed).
        result = evaluator.is_safe_mathematically(cert, [0.5, 0.5])
        assert result is False


# ── Motive-Shift Coverage ────────────────────────────────────────────────────

class TestMotiveShiftCoverage:
    """Test reuse under 3 representative motive-shift scenarios."""

    DIRECTIONS = [[1, 0], [0, 1]]
    VALUES = [0.8, 0.4]

    def _make_safe_cert(self):
        """A CDS cert known to be safe in the spec context (h_Wx=0.14)."""
        return _make_cert(
            skill_id="shift-test", delta_r=0.20, delta_n=(-0.2, 0.1),
            weight_region_type="MDN_WX",
            wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
            wx_support_values=(0.8, 0.4),
            certification_context=(0.0, 0.0),
            mdn_alpha=(1.0, 1.0),
        )

    def test_small_perturbation(self, evaluator):
        """From [0.5, 0.5] → [0.55, 0.45]: safe skill stays safe."""
        cert = self._make_safe_cert()
        assert evaluator.is_safe_mathematically(
            cert, [0.55, 0.45],
            support_directions=self.DIRECTIONS,
            support_values=self.VALUES,
        ) is True

    def test_extreme_swap(self, evaluator):
        """From [0.5, 0.5] → [0.9, 0.1]: safe skill stays safe."""
        cert = self._make_safe_cert()
        assert evaluator.is_safe_mathematically(
            cert, [0.9, 0.1],
            support_directions=self.DIRECTIONS,
            support_values=self.VALUES,
        ) is True

    def test_reverse_swap(self, evaluator):
        """From [0.5, 0.5] → [0.1, 0.9]: safe skill stays safe."""
        cert = self._make_safe_cert()
        assert evaluator.is_safe_mathematically(
            cert, [0.1, 0.9],
            support_directions=self.DIRECTIONS,
            support_values=self.VALUES,
        ) is True


# ── Empirical Validation ─────────────────────────────────────────────────────

class TestEmpiricalValidation:
    """Secondary performance checks for mathematically safe cases."""

    def test_evaluate_performance_returns_correct_fields(self, evaluator, cds_full_simplex_cert):
        """Return dict must contain exactly the 4 required keys."""
        result = evaluator.evaluate_performance(cds_full_simplex_cert, [0.5, 0.5])
        assert set(result.keys()) == {"delta_r", "delta_n", "weighted_score", "beats_baseline"}

    def test_evaluate_performance_beats_baseline(self, evaluator, cds_full_simplex_cert):
        """A beneficial skill at equal weights should beat baseline."""
        # cert: delta_r=0.8, delta_n=(0.5, 0.3)
        # score = 0.8 + 0.5*0.5 + 0.5*0.3 = 0.8 + 0.4 = 1.2 > 0
        result = evaluator.evaluate_performance(cds_full_simplex_cert, [0.5, 0.5])
        assert result["weighted_score"] > 0.0
        assert result["beats_baseline"] is True

    def test_evaluate_performance_rejects_invalid_weight(self, evaluator, cds_full_simplex_cert):
        """evaluate_performance must also validate the weight vector."""
        with pytest.raises(ValueError):
            evaluator.evaluate_performance(cds_full_simplex_cert, [0.3, 0.3])


# ── Runtime Library Integration ──────────────────────────────────────────────

from library.skill_library import SkillLibrary

class TestLibraryQueryAdmissible:
    """Validate that zero-shot reuse flows correctly through the runtime library."""

    DIRECTIONS = [[1, 0], [0, 1]]
    VALUES = [0.8, 0.4]

    @pytest.fixture
    def populated_library(self, cds_full_simplex_cert) -> SkillLibrary:
        """Build a library with both global and contextual skills."""
        library = SkillLibrary()

        # Add a globally certified FULL_SIMPLEX skill
        library.add_skill(
            skill_id=cds_full_simplex_cert.skill_id,
            certificate=cds_full_simplex_cert,
            policy=lambda obs: 0,
            weight_region_type=FULL_SIMPLEX,
        )

        # Add an MDN_WX certified skill (matching the spec example: h_Wx=0.14)
        mdn_cert = _make_cert(
            skill_id="mdn-spec-example",
            delta_r=0.15,
            delta_n=(-0.2, 0.1),
            weight_region_type=MDN_WX,
            certification_context=(1.0, 0.0, 0.0),
            mdn_alpha=(2.0, 1.0),
            wx_support_directions=tuple(tuple(d) for d in self.DIRECTIONS),
            wx_support_values=tuple(self.VALUES),
        )
        library.add_skill(
            skill_id=mdn_cert.skill_id,
            certificate=mdn_cert,
            policy=lambda obs: 0,
            weight_region_type=mdn_cert.weight_region_type,
            certification_context=mdn_cert.certification_context,
            mdn_alpha=mdn_cert.mdn_alpha,
            wx_support_directions=mdn_cert.wx_support_directions,
            wx_support_values=mdn_cert.wx_support_values,
        )

        return library

    def test_library_returns_full_simplex_always(self, evaluator, populated_library):
        """FULL_SIMPLEX skills should be admissible regardless of MDN context."""
        assert evaluator.is_reusable_via_library(
            populated_library,
            "cds-global",
            current_weight=[0.9, 0.1],
            support_directions=self.DIRECTIONS,
            support_values=self.VALUES,
        ) is True

    def test_library_returns_mdn_when_feasible(self, evaluator, populated_library):
        """MDN_WX skills should be admissible when the geometry supports it."""
        assert evaluator.is_reusable_via_library(
            populated_library,
            "mdn-spec-example",
            current_weight=[0.5, 0.5],
            support_directions=self.DIRECTIONS,
            support_values=self.VALUES,
        ) is True

    def test_library_excludes_mdn_when_cost_too_high(self, evaluator, populated_library):
        """MDN_WX skills must be excluded if the geometry implies the cost exceeds delta_r."""
        # SV=[1.0, 0.8] -> vertices=[1.0, 0.0] and [0.2, 0.8]
        # v1 dot [0.2, -0.1] = 0.2 > 0.15 (delta_r). Fails.
        result = evaluator.is_reusable_via_library(
            populated_library,
            "mdn-spec-example",
            current_weight=[0.5, 0.5],
            support_directions=self.DIRECTIONS,
            support_values=[1.0, 0.8],
        )
        assert result is False

"""
Zero-shot reuse evaluator for SubRep certified skills.

Validates that certified skills can be safely reused under new motive weights
without retraining. Supports two certification modes:

    FULL_SIMPLEX — Globally reusable: safe under any valid simplex weight.
    MDN_WX      — Contextually reusable: safe under the current learned
                  motive geometry described by support directions/values.

Mathematical safety is the primary guarantee; empirical performance checks
are secondary validation.
"""

from __future__ import annotations

import numpy as np
from certification.certificate_schema import Certificate
from library.skill_metadata import FULL_SIMPLEX, MDN_WX

# ── Region-type constants ────────────────────────────────────────────────────
_SUPPORTED_REGION_TYPES = {FULL_SIMPLEX, MDN_WX}


class ZeroShotEvaluator:
    """Evaluates whether a certified skill is safe to reuse under new weights.

    Two modes are supported:

    1. **FULL_SIMPLEX** — A skill certified over the full simplex is safe under
       *any* valid weight vector. Only the validity of ``new_weight`` is checked.

    2. **MDN_WX** — A skill certified within a context-conditioned weight set
       W_x is safe if delta_r covers the worst-case motive cost h_Wx(-delta_n)
       computed from the current context's support descriptor.
    """

    # ── Primary: Mathematical safety check ───────────────────────────────────

    def is_safe_mathematically(
        self,
        certificate: Certificate,
        new_weight: list | np.ndarray,
        support_directions: list | np.ndarray | None = None,
        support_values: list | np.ndarray | None = None,
    ) -> bool:
        """Check whether a certified skill is safe to reuse at *new_weight*.

        Args:
            certificate: The skill's admission certificate.
            new_weight:  Target weight vector (must be a valid simplex point).
            support_directions: (MDN_WX only) Query directions describing W_x.
            support_values:     (MDN_WX only) Support thresholds h_j for W_x.
                                NOTE: These are *not* weight vectors and do NOT
                                need to sum to 1.

        Returns:
            True if reuse is mathematically safe; False otherwise.
        """
        # 1. Mandatory weight validation
        w = np.asarray(new_weight, dtype=np.float64)
        self._validate_simplex(w)

        # 2. Check region type (default to FULL_SIMPLEX for legacy support)
        region_type = getattr(certificate, "weight_region_type", FULL_SIMPLEX)
        if region_type not in _SUPPORTED_REGION_TYPES:
            raise ValueError(
                f"Unsupported weight_region_type {region_type!r}; "
                f"expected one of {_SUPPORTED_REGION_TYPES}"
            )

        # ── Mode 1: Full-simplex global reuse ────────────────────────────
        if region_type == FULL_SIMPLEX:
            # Any valid simplex weight is safe for a full-simplex certificate.
            return True

        # ── Mode 2: MDN/contextual reuse ─────────────────────────────────
        # Blocker case: Contextual skill without context is unsafe.
        if support_directions is None or support_values is None:
            return False

        # Delegate to SkillLibrary for the math (eliminates logic duplication)
        from library.skill_library import SkillLibrary

        temp_lib = SkillLibrary()
        if not temp_lib.add_skill(
            skill_id=certificate.skill_id,
            certificate=certificate,
            policy=lambda *_args, **_kwargs: None,
            weight_region_type=MDN_WX,
            certification_context=certificate.certification_context,
            mdn_alpha=certificate.mdn_alpha,
            wx_support_directions=certificate.wx_support_directions,
            wx_support_values=certificate.wx_support_values,
        ):
            return False

        admissible = temp_lib.query_admissible(
            current_weight=w,
            support_directions=support_directions,
            support_values=support_values,
        )

        return any(entry.skill_id == certificate.skill_id for entry in admissible)

    # ── Secondary: Empirical performance validation ──────────────────────────

    def evaluate_performance(
        self,
        certificate: Certificate,
        new_weight: list | np.ndarray,
    ) -> dict:
        """Compute the weighted performance score under *new_weight*.

        Uses the certificate's stored delta values (computed at certification
        time) as a controlled evaluation — no environment re-simulation needed.

        Args:
            certificate: The skill's admission certificate.
            new_weight:  Target weight vector (valid simplex point).

        Returns:
            Dictionary with keys: delta_r, delta_n, weighted_score,
            beats_baseline.
        """
        w = np.asarray(new_weight, dtype=np.float64)
        self._validate_simplex(w)

        delta_r = float(certificate.delta_r)
        delta_n = np.asarray(certificate.delta_n, dtype=np.float64)

        weighted_score = delta_r + float(np.dot(w, delta_n))

        return {
            "delta_r": delta_r,
            "delta_n": tuple(float(v) for v in delta_n),
            "weighted_score": weighted_score,
            "beats_baseline": weighted_score > 0.0,
        }

    # ── Library Integration ──────────────────────────────────────────────────

    def is_reusable_via_library(
        self,
        library,
        skill_id: str,
        current_weight: list | np.ndarray,
        support_directions: list | np.ndarray | None = None,
        support_values: list | np.ndarray | None = None,
    ) -> bool:
        """Check if a skill is reusable via the runtime SkillLibrary.

        Delegates the admissibility check to the library's unified
        `query_admissible()` method. This is the recommended path for
        runtime selection, replacing standalone mathematical evaluation.

        Args:
            library: The SkillLibrary instance.
            skill_id: The ID of the skill to check.
            current_weight: The active motive weight.
            support_directions: (MDN_WX only) Core W_x directions.
            support_values: (MDN_WX only) Core W_x thresholds.

        Returns:
            True if the skill is found in the returned admissible list.
        """
        admissible = library.query_admissible(
            current_weight=current_weight,
            support_directions=support_directions,
            support_values=support_values,
        )
        return any(entry.skill_id == skill_id for entry in admissible)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_simplex(w: np.ndarray) -> None:
        """Raise ValueError if *w* is not a valid simplex vector."""
        if w.ndim != 1 or len(w) == 0:
            raise ValueError(f"Weight must be a 1-D non-empty array, got shape {w.shape}")
        if np.any(w < 0.0):
            raise ValueError(f"Weight components must be >= 0, got {w}")
        if not np.isclose(w.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"Weight components must sum to 1, got sum={w.sum():.6f}"
            )

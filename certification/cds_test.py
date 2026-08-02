"""
Cone-Dominant Subtask (CDS) Admission Gate.

Implements Definition 1 from SubRep Paper Section 3.2:
    Δr ≥ max_i(-Δn_i)
    
Equivalently: Δr + min_i(Δn_i) ≥ 0

CDS ensures skills are universally beneficial across all weight vectors
in the simplex cone (no coordinate "pays" more than baseline improves).
"""

from __future__ import annotations
import numpy as np
from typing import Union
from .gate import AdmissionGate, Scalar
from utils.cone_utils import compute_worst_case_motive
from utils.support_geometry import worst_case_over_support_region

class CDSGate(AdmissionGate):
    """
    Cone-Dominant Subtask admission gate.
    
    Admits skills that are universally beneficial (no motive coordinate
    worsens more than the baseline payoff improves).
    
    Reference: SubRep Paper Section 3.2, Definition 1
    """
    
    def admit(
        self,
        delta_r: Scalar,
        delta_n: np.ndarray,
        weight_set=None,
        support_values: np.ndarray | None = None,
    ) -> bool:
        """
        Check if skill satisfies CDS condition.

        Formula: Δr + min_i(Δn_i) ≥ 0

        Args:
            delta_r: Scalar payoff improvement.
            delta_n: Motive improvement vector.
            weight_set: Optional explicit vertex set (legacy M=2 path).
            support_values: Optional W_x support geometry. When given, the
                worst case is computed exactly via the greedy support function
                at any M, and no vertices are materialized. Takes precedence
                over `weight_set`.

        Returns:
            True if the worst-case score over the weight set is non-negative.
        """
        self.validate_inputs(delta_r, delta_n)

        if support_values is not None:
            min_score = worst_case_over_support_region(delta_n, support_values)
            return bool(float(delta_r) + min_score >= 0.0)

        if weight_set is None or weight_set.is_empty():
            min_motive = compute_worst_case_motive(delta_n)
            return bool(float(delta_r) + min_motive >= 0.0)

        vertices = weight_set.get_vertices_array()
        if vertices is None:
            min_motive = compute_worst_case_motive(delta_n)
            return bool(float(delta_r) + min_motive >= 0.0)
        delta_n_arr = np.asarray(delta_n, dtype=np.float32)
        scores = vertices @ delta_n_arr
        min_score = float(np.min(scores))
        return bool(float(delta_r) + min_score >= 0.0)
    
    def get_gate_type(self) -> str:
        """Return gate type identifier."""
        return "CDS"
    
    def get_admission_margin(
        self,
        delta_r: Scalar,
        delta_n: np.ndarray,
        weight_set=None,
        support_values: np.ndarray | None = None,
    ) -> float:
        """
        Calculate how much the skill passes/fails by.

        Positive margin = admitted, Negative margin = rejected.

        Returns:
            Margin value under the active weight set.
        """
        self.validate_inputs(delta_r, delta_n)
        if support_values is not None:
            return float(delta_r) + worst_case_over_support_region(delta_n, support_values)
        if weight_set is None or weight_set.is_empty():
            return float(delta_r) + float(np.min(delta_n))
        vertices = weight_set.get_vertices_array()
        if vertices is None:
            return float(delta_r) + float(np.min(delta_n))
        delta_n_arr = np.asarray(delta_n, dtype=np.float32)
        return float(delta_r) + float(np.min(vertices @ delta_n_arr))

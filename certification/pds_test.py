"""
Pareto-Dominant Subtask (PDS) Admission Gate.

Implements Definition 2 from SubRep Paper Section 3.2:
    inf_w∈W [Δr + w^T Δn] ≥ -ε
    
For simplex cone, simplifies to: Δr + min_i(Δn_i) ≥ -ε

PDS allows bounded trade-offs for specialized skills (some motives can
worsen if payoff improvement compensates within ε budget).
"""

from __future__ import annotations
import numpy as np
from typing import Union
from .gate import AdmissionGate, Scalar
from utils.cone_utils import compute_worst_case_motive
from utils.support_geometry import worst_case_over_support_region

# Default epsilon budget (configurable via config.py in production)
DEFAULT_EPSILON = 0.1

class PDSGate(AdmissionGate):
    """
    Pareto-Dominant Subtask admission gate with epsilon budget.
    
    Admits skills with bounded trade-offs (some motives can worsen if
    payoff improvement compensates within ε budget).
    
    Reference: SubRep Paper Section 3.2, Definition 2
    """
    
    def __init__(self, epsilon: float = DEFAULT_EPSILON) -> None:
        """
        Initialize PDS gate with epsilon budget.
        
        Args:
            epsilon: Allowed negative value budget (≥ 0).
        """
        if epsilon < 0:
            raise ValueError(f"epsilon must be non-negative, got {epsilon}")
        self.epsilon = float(epsilon)
    
    def admit(
        self,
        delta_r: Scalar,
        delta_n: np.ndarray,
        weight_set=None,
        support_values: np.ndarray | None = None,
    ) -> bool:
        """
        Check if skill satisfies PDS-ε condition.

        Formula: Δr + min_i(Δn_i) ≥ -ε

        Args:
            delta_r: Scalar payoff improvement.
            delta_n: Motive improvement vector.
            weight_set: Optional explicit vertex set (legacy M=2 path).
            support_values: Optional W_x support geometry. When given, the
                worst case is computed exactly via the greedy support function
                at any M, and no vertices are materialized. Takes precedence
                over `weight_set`.

        Returns:
            True if the worst-case score over the weight set is within ε budget.
        """
        self.validate_inputs(delta_r, delta_n)

        if support_values is not None:
            min_score = worst_case_over_support_region(delta_n, support_values)
            return bool(float(delta_r) + min_score >= -self.epsilon)

        if weight_set is None or weight_set.is_empty():
            min_motive = compute_worst_case_motive(delta_n)
            return bool(float(delta_r) + min_motive >= -self.epsilon)

        vertices = weight_set.get_vertices_array()
        if vertices is None:
            min_motive = compute_worst_case_motive(delta_n)
            return bool(float(delta_r) + min_motive >= -self.epsilon)
        delta_n_arr = np.asarray(delta_n, dtype=np.float32)
        scores = vertices @ delta_n_arr
        min_score = float(np.min(scores))
        return bool(float(delta_r) + min_score >= -self.epsilon)
    
    def get_gate_type(self) -> str:
        """Return gate type identifier."""
        return "PDS"
    
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
            return (
                float(delta_r)
                + worst_case_over_support_region(delta_n, support_values)
                + self.epsilon
            )
        if weight_set is None or weight_set.is_empty():
            return float(delta_r) + float(np.min(delta_n)) + self.epsilon
        vertices = weight_set.get_vertices_array()
        if vertices is None:
            return float(delta_r) + float(np.min(delta_n)) + self.epsilon
        delta_n_arr = np.asarray(delta_n, dtype=np.float32)
        return float(delta_r) + float(np.min(vertices @ delta_n_arr)) + self.epsilon
    
    def get_epsilon(self) -> float:
        """Return the epsilon budget."""
        return self.epsilon
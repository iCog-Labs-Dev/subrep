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
    
    def admit(self, delta_r: Scalar, delta_n: np.ndarray) -> bool:
        """
        Check if skill satisfies PDS-ε condition.
        
        Formula: Δr + min_i(Δn_i) ≥ -ε
        
        Args:
            delta_r: Scalar payoff improvement.
            delta_n: Motive improvement vector.
        
        Returns:
            True if Δr + min(Δn) ≥ -ε, False otherwise.
        """
        self.validate_inputs(delta_r, delta_n)
        
        
        min_motive = compute_worst_case_motive(delta_n)
        return bool(float(delta_r) + min_motive >= -self.epsilon)
    
    def get_gate_type(self) -> str:
        """Return gate type identifier."""
        return "PDS"
    
    def get_admission_margin(self, delta_r: Scalar, delta_n: np.ndarray) -> float:
        """
        Calculate how much the skill passes/fails by.
        
        Positive margin = admitted, Negative margin = rejected.
        
        Returns:
            Margin value (Δr + min(Δn) + ε).
        """
        self.validate_inputs(delta_r, delta_n)
        return float(delta_r) + float(np.min(delta_n)) + self.epsilon
    
    def get_epsilon(self) -> float:
        """Return the epsilon budget."""
        return self.epsilon
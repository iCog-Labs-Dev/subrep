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

class CDSGate(AdmissionGate):
    """
    Cone-Dominant Subtask admission gate.
    
    Admits skills that are universally beneficial (no motive coordinate
    worsens more than the baseline payoff improves).
    
    Reference: SubRep Paper Section 3.2, Definition 1
    """
    
    def admit(self, delta_r: Scalar, delta_n: np.ndarray) -> bool:
        """
        Check if skill satisfies CDS condition.
        
        Formula: Δr + min_i(Δn_i) ≥ 0
        
        Args:
            delta_r: Scalar payoff improvement.
            delta_n: Motive improvement vector.
        
        Returns:
            True if Δr + min(Δn) ≥ 0, False otherwise.
        """
        self.validate_inputs(delta_r, delta_n)
        
     
        min_motive = compute_worst_case_motive(delta_n)
        return bool(float(delta_r) + min_motive >= 0.0)
    
    def get_gate_type(self) -> str:
        """Return gate type identifier."""
        return "CDS"
    
    def get_admission_margin(self, delta_r: Scalar, delta_n: np.ndarray) -> float:
        """
        Calculate how much the skill passes/fails by.
        
        Positive margin = admitted, Negative margin = rejected.
        
        Returns:
            Margin value (Δr + min(Δn)).
        """
        self.validate_inputs(delta_r, delta_n)
        return float(delta_r) + float(np.min(delta_n))
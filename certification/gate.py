"""
Admission Gate Base Class for SubRep.

Defines the interface for skill certification gates (CDS, PDS, etc.).
All gates must implement the admit() method to validate skill improvements.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
import numpy as np

# Type alias for scalar payoff improvement
Scalar = Union[float, np.floating]

class AdmissionGate(ABC):
    """
    Abstract base class for skill admission gates.
    
    Subclasses implement specific gate logic (CDS, PDS, etc.) to determine
    whether a skill's improvements (Δr, Δn) are valid for admission to the
    skill library.
    
    """
    
    @abstractmethod
    def admit(self, delta_r: Scalar, delta_n: np.ndarray) -> bool:
        """
        Determine if a skill passes the admission gate.
        
        Args:
            delta_r: Scalar payoff improvement (Δr = r̂ - r̂_base(Agent takes no action or minimal action)).
            delta_n: Motive improvement vector (Δn = n̂ - n̂_base(Agent takes no action or minimal action)), shape (m,).
        
        Returns:
            True if skill is admitted, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_gate_type(self) -> str:
        """Return the gate type identifier (e.g., 'CDS', 'PDS')."""
        pass
    
    def validate_inputs(self, delta_r: Scalar, delta_n: np.ndarray) -> None:
        """
        Validate input shapes and types before gate evaluation.
        
        Raises:
            ValueError: If inputs are invalid.
        """
        if not np.isscalar(delta_r):
            raise ValueError(f"delta_r must be scalar, got {type(delta_r)}")
        
        if not isinstance(delta_n, np.ndarray):
            raise ValueError(f"delta_n must be np.ndarray, got {type(delta_n)}")
        
        if delta_n.ndim != 1:
            raise ValueError(f"delta_n must be 1D vector, got shape {delta_n.shape}")
        
        if not np.isfinite(delta_r):
            raise ValueError(f"delta_r must be finite, got {delta_r}")
        
        if not np.all(np.isfinite(delta_n)):
            raise ValueError(f"delta_n must be finite, got {delta_n}")
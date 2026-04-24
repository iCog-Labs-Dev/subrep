"""
Cone Geometry Utilities for SubRep.

Helper functions for weight validation and cone operations.
Used by admission gates and future MDN integration.

Reference: SubRep Paper Section 3.2 (Admission Gates)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

def validate_simplex_weights(weights: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Validate that weights form a valid simplex (non-negative, sum to 1).
    
    Args:
        weights: Weight vector to validate.
        tolerance: Numerical tolerance for sum-to-1 check.
    
    Returns:
        True if valid simplex weights, False otherwise.
    """
    if not isinstance(weights, np.ndarray):
        return False
    
    if weights.ndim != 1:
        return False
    
    if np.any(weights < -tolerance):
        return False
    
    if not np.isclose(np.sum(weights), 1.0, atol=tolerance):
        return False
    
    return True

def compute_support_function(u: np.ndarray) -> float:
    """
    Compute support function h_W(u) for simplex cone.
    
    For simplex W: h_W(u) = max_i(u_i)
    
    Args:
        u: Input vector.
    
    Returns:
        Support function value (max component of u).
    
    Reference: SubRep Paper Section 3.2
    """
    return float(np.max(u))

def compute_worst_case_motive(delta_n: np.ndarray) -> float:
    """
    Compute worst-case motive value over simplex.
    
    For simplex: inf_w∈W w^T Δn = min_i(Δn_i)
    
    Args:
        delta_n: Motive improvement vector.
    
    Returns:
        Worst-case motive value (min component).
    """
    return float(np.min(delta_n))

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights to form valid simplex (project onto simplex).
    
    Args:
        weights: Raw weight vector.
    
    Returns:
        Normalized weights (non-negative, sum to 1).
    """
    # Clip negative values to 0
    clipped = np.maximum(weights, 0.0)
    
    # Handle zero-sum case
    total = np.sum(clipped)
    if total < 1e-10:
        # Return uniform weights if all zero
        return np.ones_like(clipped) / len(clipped)
    
    # Normalize to sum to 1
    return clipped / total

def get_simplex_vertices(dim: int) -> np.ndarray:
    """
    Get vertices of the standard simplex in R^dim.
    
    Vertices are unit vectors e_1, e_2, ..., e_dim.
    
    Args:
        dim: Dimension of the simplex.
    
    Returns:
        Array of shape (dim, dim) with vertices as rows.
    """
    return np.eye(dim)
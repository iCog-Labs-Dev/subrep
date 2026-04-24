"""
Utilities package for SubRep.

Exports common mathematical and structural helpers.
"""

from .cone_utils import (
    validate_simplex_weights,
    compute_support_function,
    compute_worst_case_motive,
    normalize_weights,
    get_simplex_vertices,
)

__all__ = [
    "validate_simplex_weights",
    "compute_support_function",
    "compute_worst_case_motive",
    "normalize_weights",
    "get_simplex_vertices",
]
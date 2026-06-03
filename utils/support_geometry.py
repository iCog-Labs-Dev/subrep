"""Support-function geometry utilities for context-conditioned weight sets W_x.
"""

from __future__ import annotations

import numpy as np


def make_basis_query_directions(num_objectives: int) -> np.ndarray:
    """Return standard basis query directions for the objective space.
    """
    if num_objectives <= 0:
        raise ValueError(f"num_objectives must be positive, got {num_objectives}")
    return np.eye(num_objectives, dtype=np.float32)


def compute_support_values_from_vertices(vertices: np.ndarray, query_directions: np.ndarray) -> np.ndarray:
    """Compute support-function values h_W(u_j) = max_{w in W} u_j^T w.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    query_directions = np.asarray(query_directions, dtype=np.float32)

    if vertices.ndim != 2:
        raise ValueError(f"vertices must have shape (N, M), got {vertices.shape}")
    if query_directions.ndim != 2:
        raise ValueError(f"query_directions must have shape (K, M), got {query_directions.shape}")
    if vertices.shape[0] == 0:
        raise ValueError("vertices must contain at least one weight vector")
    if vertices.shape[1] != query_directions.shape[1]:
        raise ValueError(
            f"vertices dimension {vertices.shape[1]} must match query direction dimension {query_directions.shape[1]}"
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must contain only finite values")
    if not np.all(np.isfinite(query_directions)):
        raise ValueError("query_directions must contain only finite values")

    scores = query_directions @ vertices.T
    return np.max(scores, axis=1).astype(np.float32)


def simplex_support_values(query_directions: np.ndarray) -> np.ndarray:
    """Compute support-function values for the full simplex.
    """
    query_directions = np.asarray(query_directions, dtype=np.float32)
    if query_directions.ndim != 2:
        raise ValueError(f"query_directions must have shape (K, M), got {query_directions.shape}")
    if query_directions.shape[0] == 0 or query_directions.shape[1] == 0:
        raise ValueError("query_directions must be non-empty")
    if not np.all(np.isfinite(query_directions)):
        raise ValueError("query_directions must contain only finite values")
    return np.max(query_directions, axis=1).astype(np.float32)

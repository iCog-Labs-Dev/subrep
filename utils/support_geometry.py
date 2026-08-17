"""Support-function geometry utilities for context-conditioned weight sets W_x.
"""

from __future__ import annotations

import numpy as np

# Tolerance for the sum(s) >= 1 feasibility test. Support values arrive from a
# float32 network head, so an exact comparison would spuriously reject regions
# that are feasible up to rounding. Single source of truth: library.skill_library
# imports this rather than defining its own copy.
FEASIBILITY_TOLERANCE = 1e-9


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


def greedy_support_function(coefficients: np.ndarray, support_values: np.ndarray) -> float:
    """Compute h_W(c) = max { w . c : w in simplex, w_i <= s_i } exactly.

    This is a linear program with a closed-form greedy solution: sort the
    coordinates by coefficient descending and allocate weight to each up to its
    cap ``s_i`` until total mass 1 is placed. The greedy is exact because the
    feasible set is the base polytope of a polymatroid (equivalently, the
    problem is a fractional knapsack), for which greedy allocation is optimal.

    Complexity is O(M log M), the computation is permutation-symmetric by
    construction, and no vertices are ever materialized -- which is what makes
    it usable at arbitrary M where explicit vertex enumeration grows
    combinatorially.

    Feasibility is validated, not assumed. `W` is non-empty exactly when every
    ``s_i`` lies in [0, 1] and ``sum(s) >= 1``; only then is full mass placeable
    and the maximum well defined. An infeasible region has no maximum at all, so
    silently allocating whatever mass the caps allow would return a value that is
    too small -- and a too-small worst case makes an admission gate strictly more
    permissive, which is the dangerous direction. Rejecting is the only safe
    behavior.

    The full simplex is the ``s = 1``-vector special case, for which this reduces
    to ``max_i(c_i)``.

    Args:
        coefficients: Objective coefficient vector c, shape (M,).
        support_values: Per-objective caps s, shape (M,).

    Returns:
        The support-function value h_W(c).

    Raises:
        ValueError: on mismatched lengths, non-finite input, or a support vector
            that does not describe a non-empty region.
    """
    c = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    sv = np.asarray(support_values, dtype=np.float64).reshape(-1)

    if c.size != sv.size:
        raise ValueError(
            f"coefficients length {c.size} must match support_values length {sv.size}"
        )
    if c.size == 0:
        raise ValueError("coefficients and support_values must be non-empty")
    if not np.all(np.isfinite(c)):
        raise ValueError(f"coefficients must be finite, got {c.tolist()}")
    if not np.all(np.isfinite(sv)):
        raise ValueError(f"support_values must be finite, got {sv.tolist()}")

    # Feasibility of W = { w in simplex : w_i <= s_i }. Validated here rather
    # than at each call site so every caller inherits it -- the admission gates
    # reach this function directly, without passing through the skill library's
    # validator.
    if not np.all((sv >= 0.0) & (sv <= 1.0)):
        raise ValueError(
            f"support_values must satisfy 0 <= s_i <= 1, got {sv.tolist()}"
        )
    if float(np.sum(sv)) < 1.0 - FEASIBILITY_TOLERANCE:
        raise ValueError(
            f"support_values must satisfy sum(s) >= 1 (otherwise the region is "
            f"empty and has no maximum), got sum={float(np.sum(sv)):.6f} from "
            f"{sv.tolist()}"
        )

    # Stable sort keeps the result deterministic when coefficients tie.
    order = np.argsort(-c, kind="stable")
    remaining = 1.0
    value = 0.0
    for index in order:
        take = min(float(sv[index]), remaining)
        if take <= 0.0:
            continue  # zero cap contributes nothing; s_i < 0 is already rejected
        value += take * float(c[index])
        remaining -= take
        if remaining <= 1e-12:
            break

    return float(value)


def worst_case_over_support_region(delta_n: np.ndarray, support_values: np.ndarray) -> float:
    """Return min_{w in W_x} w · Δn for W_x = { w in simplex : w_i <= s_i }.

    This is the quantity both admission gates need. It is the negated support
    function, so it inherits the greedy solver's exactness and O(M log M) cost:
        min_w (w · Δn) = -max_w (w · (-Δn)) = -h_{W_x}(-Δn)
    """
    delta_n_arr = np.asarray(delta_n, dtype=np.float64).reshape(-1)
    return -greedy_support_function(-delta_n_arr, support_values)


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

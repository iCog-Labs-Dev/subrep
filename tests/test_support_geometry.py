from __future__ import annotations

import numpy as np
import pytest

from utils.support_geometry import (
    compute_support_values_from_vertices,
    make_basis_query_directions,
    simplex_support_values,
)


def test_make_basis_query_directions_returns_identity():
    directions = make_basis_query_directions(3)

    assert directions.shape == (3, 3)
    assert np.allclose(directions, np.eye(3, dtype=np.float32))


def test_make_basis_query_directions_rejects_non_positive_dimension():
    with pytest.raises(ValueError, match="positive"):
        make_basis_query_directions(0)


def test_compute_support_values_from_vertices_matches_manual_values():
    vertices = np.array(
        [
            [0.8, 0.2],
            [0.4, 0.6],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    directions = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )

    support_values = compute_support_values_from_vertices(vertices, directions)

    expected = np.array([0.8, 0.9, 0.5], dtype=np.float32)
    assert support_values.shape == (3,)
    assert np.allclose(support_values, expected)


def test_compute_support_values_from_vertices_rejects_empty_vertices():
    with pytest.raises(ValueError, match="at least one"):
        compute_support_values_from_vertices(np.empty((0, 2), dtype=np.float32), np.eye(2, dtype=np.float32))


def test_compute_support_values_from_vertices_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="must match"):
        compute_support_values_from_vertices(np.ones((2, 2), dtype=np.float32), np.ones((2, 3), dtype=np.float32))


def test_simplex_support_values_matches_coordinate_max():
    directions = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.7],
            [-1.0, 2.0],
        ],
        dtype=np.float32,
    )

    support_values = simplex_support_values(directions)

    expected = np.array([1.0, 1.0, 0.7, 2.0], dtype=np.float32)
    assert support_values.shape == (4,)
    assert np.allclose(support_values, expected)

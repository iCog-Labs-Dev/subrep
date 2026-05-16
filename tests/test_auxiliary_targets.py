from __future__ import annotations

import numpy as np
import pytest

from utils.return_targets import discounted_motive_return, doubly_robust_return, ips_weighted_return


def test_discounted_motive_return_single_trajectory_shape_and_values():
    motives = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target = discounted_motive_return(motives, gamma=0.5)

    expected = np.array([1.0 + 0.5 * 3.0, 2.0 + 0.5 * 4.0], dtype=np.float32)
    assert target.shape == (2,)
    assert np.allclose(target, expected)


def test_discounted_motive_return_batched_shape():
    motives = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 1.0], [4.0, 3.0]],
        ],
        dtype=np.float32,
    )
    target = discounted_motive_return(motives, gamma=0.5)

    assert target.shape == (2, 2)


def test_ips_weighted_return_raises_without_probabilities():
    motives = np.ones((2, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="behavior_probability"):
        ips_weighted_return(motives)


def test_ips_weighted_return_computes_when_probabilities_exist():
    motives = np.ones((1, 2, 2), dtype=np.float32)
    behavior_probability = np.array([[0.5, 0.5]], dtype=np.float32)
    target_probability = np.array([[1.0, 1.0]], dtype=np.float32)

    target = ips_weighted_return(
        motives,
        behavior_probability=behavior_probability,
        target_probability=target_probability,
        gamma=1.0,
    )

    assert target.shape == (1, 2)
    assert np.all(np.isfinite(target))


def test_doubly_robust_return_raises_without_required_fields():
    motives = np.ones((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="q_model_estimate"):
        doubly_robust_return(
            motives,
            behavior_probability=np.array([[0.5, 0.5]], dtype=np.float32),
            target_probability=np.array([[1.0, 1.0]], dtype=np.float32),
        )

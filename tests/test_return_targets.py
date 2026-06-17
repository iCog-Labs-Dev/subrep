from __future__ import annotations

import numpy as np

from utils.return_targets import (
    DEFAULT_IPS_CLIP_VALUE,
    discounted_motive_return,
    doubly_robust_return,
    ips_weighted_return,
)


def test_ips_weighted_return_supports_importance_weight_clipping():
    motives = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    behavior_probability = np.array([[0.1, 0.1]], dtype=np.float32)
    target_probability = np.array([[10.0, 10.0]], dtype=np.float32)

    unclipped = ips_weighted_return(
        motives,
        behavior_probability=behavior_probability,
        target_probability=target_probability,
        clip_value=None,
    )
    clipped = ips_weighted_return(
        motives,
        behavior_probability=behavior_probability,
        target_probability=target_probability,
        clip_value=10.0,
    )

    assert np.all(unclipped > clipped)
    assert clipped.shape == (1, 2)


def test_doubly_robust_return_equals_actual_return_on_policy():
    motives = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    q_model = np.array([[0.5, 0.5]], dtype=np.float32)

    target = doubly_robust_return(
        motives,
        behavior_probability=np.array([0.5], dtype=np.float32),
        target_probability=np.array([0.5], dtype=np.float32),
        q_model_estimate=q_model,
        gamma=1.0,
    )

    assert np.allclose(target, discounted_motive_return(motives, gamma=1.0))


def test_doubly_robust_return_keeps_perfect_baseline_fixed():
    motives = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    q_model = discounted_motive_return(motives, gamma=1.0)

    target = doubly_robust_return(
        motives,
        behavior_probability=np.array([0.1], dtype=np.float32),
        target_probability=np.array([1.0], dtype=np.float32),
        q_model_estimate=q_model,
        gamma=1.0,
    )

    assert np.allclose(target, q_model)


def test_doubly_robust_return_clips_final_importance_ratio():
    motives = np.array([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    q_model = np.array([[0.0, 0.0]], dtype=np.float32)

    target = doubly_robust_return(
        motives,
        behavior_probability=np.array([[0.1, 0.1]], dtype=np.float32),
        target_probability=np.array([[10.0, 10.0]], dtype=np.float32),
        q_model_estimate=q_model,
        gamma=1.0,
    )

    expected = DEFAULT_IPS_CLIP_VALUE * discounted_motive_return(motives, gamma=1.0)
    assert np.allclose(target, expected)


def test_doubly_robust_return_rejects_nonconstant_step_probabilities():
    motives = np.array([[[1.0, 0.0], [1.0, 0.0]]], dtype=np.float32)
    q_model = np.array([[0.0, 0.0]], dtype=np.float32)

    try:
        doubly_robust_return(
            motives,
            behavior_probability=np.array([[0.5, 0.25]], dtype=np.float32),
            target_probability=np.array([[1.0, 1.0]], dtype=np.float32),
            q_model_estimate=q_model,
            gamma=1.0,
        )
    except ValueError as exc:
        assert "option-level" in str(exc)
    else:
        raise AssertionError("Expected nonconstant step probabilities to be rejected")

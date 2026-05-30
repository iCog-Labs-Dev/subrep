"""Return-target helpers for the proposal-conditioned auxiliary MDN trainer."""

from __future__ import annotations

from typing import Optional

import numpy as np


def discounted_motive_return(motives: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Compute discounted motive return from a motive trajectory.

    Supported inputs:
    - shape `(T, M)` -> returns shape `(M,)`
    - shape `(N, T, M)` -> returns shape `(N, M)`
    """
    motives = np.asarray(motives, dtype=np.float32)
    if motives.ndim not in (2, 3):
        raise ValueError(f"motives must have shape (T, M) or (N, T, M), got {motives.shape}")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not np.all(np.isfinite(motives)):
        raise ValueError("motives must contain only finite values")

    if motives.ndim == 2:
        discounts = np.power(gamma, np.arange(motives.shape[0], dtype=np.float32))
        return np.sum(motives * discounts[:, None], axis=0).astype(np.float32)

    discounts = np.power(gamma, np.arange(motives.shape[1], dtype=np.float32))
    return np.sum(motives * discounts[None, :, None], axis=1).astype(np.float32)


def ips_weighted_return(
    motives: np.ndarray,
    behavior_probability: Optional[np.ndarray] = None,
    target_probability: Optional[np.ndarray] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute an IPS-weighted discounted motive return.

    Requires explicit behavior and target probabilities. Raises when they are
    missing or invalid rather than silently faking IPS.
    """
    if behavior_probability is None or target_probability is None:
        raise ValueError("IPS return requires behavior_probability and target_probability")

    motives = np.asarray(motives, dtype=np.float32)
    behavior_probability = np.asarray(behavior_probability, dtype=np.float32)
    target_probability = np.asarray(target_probability, dtype=np.float32)

    if motives.ndim != 3:
        raise ValueError(f"IPS motives must have shape (N, T, M), got {motives.shape}")
    if behavior_probability.shape != motives.shape[:2]:
        raise ValueError(
            f"behavior_probability shape {behavior_probability.shape} must match motives prefix {motives.shape[:2]}"
        )
    if target_probability.shape != motives.shape[:2]:
        raise ValueError(
            f"target_probability shape {target_probability.shape} must match motives prefix {motives.shape[:2]}"
        )
    if np.any(behavior_probability <= 0.0):
        raise ValueError("behavior_probability must be strictly positive for IPS")
    if np.any(target_probability < 0.0):
        raise ValueError("target_probability must be non-negative for IPS")
    if not np.all(np.isfinite(behavior_probability)) or not np.all(np.isfinite(target_probability)):
        raise ValueError("probabilities must contain only finite values")

    discounts = np.power(gamma, np.arange(motives.shape[1], dtype=np.float32))
    importance_weights = target_probability / behavior_probability
    return np.sum(motives * importance_weights[:, :, None] * discounts[None, :, None], axis=1).astype(np.float32)


def doubly_robust_return(
    motives: np.ndarray,
    behavior_probability: Optional[np.ndarray] = None,
    target_probability: Optional[np.ndarray] = None,
    q_model_estimate: Optional[np.ndarray] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute a simple doubly-robust style target when all required inputs exist."""
    if behavior_probability is None or target_probability is None or q_model_estimate is None:
        raise ValueError(
            "Doubly robust return requires behavior_probability, target_probability, and q_model_estimate"
        )

    ips_target = ips_weighted_return(
        motives,
        behavior_probability=behavior_probability,
        target_probability=target_probability,
        gamma=gamma,
    )
    q_model_estimate = np.asarray(q_model_estimate, dtype=np.float32)
    if q_model_estimate.shape != ips_target.shape:
        raise ValueError(
            f"q_model_estimate shape {q_model_estimate.shape} must match IPS target shape {ips_target.shape}"
        )
    if not np.all(np.isfinite(q_model_estimate)):
        raise ValueError("q_model_estimate must contain only finite values")

    direct_target = discounted_motive_return(motives, gamma=gamma)
    return (q_model_estimate + (ips_target - direct_target)).astype(np.float32)

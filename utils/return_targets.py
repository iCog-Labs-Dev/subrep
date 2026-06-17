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


DEFAULT_IPS_CLIP_VALUE = 10.0


def ips_weighted_return(
    motives: np.ndarray,
    behavior_probability: Optional[np.ndarray] = None,
    target_probability: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    clip_value: float | None = None,
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
    if clip_value is not None:
        clip_value = float(clip_value)
        if clip_value <= 0.0 or not np.isfinite(clip_value):
            raise ValueError(f"clip_value must be positive and finite when provided, got {clip_value}")
        importance_weights = np.minimum(importance_weights, clip_value)
    return np.sum(motives * importance_weights[:, :, None] * discounts[None, :, None], axis=1).astype(np.float32)


def doubly_robust_return(
    motives: np.ndarray,
    behavior_probability: Optional[np.ndarray] = None,
    target_probability: Optional[np.ndarray] = None,
    q_model_estimate: Optional[np.ndarray] = None,
    gamma: float = 1.0,
    clip_value: float | None = DEFAULT_IPS_CLIP_VALUE,
) -> np.ndarray:
    """Compute an option-level doubly robust target.

    The correction is `q_model + rho * (actual_return - q_model)`, where
    `rho = pi_target(option | context) / pi_behavior(option | context)`. The
    ratio is clipped after it is formed to bound variance.
    """
    if behavior_probability is None or target_probability is None or q_model_estimate is None:
        raise ValueError(
            "Doubly robust return requires behavior_probability, target_probability, and q_model_estimate"
        )

    direct_target = discounted_motive_return(motives, gamma=gamma)
    q_model_estimate = np.asarray(q_model_estimate, dtype=np.float32)
    if q_model_estimate.shape != direct_target.shape:
        raise ValueError(
            f"q_model_estimate shape {q_model_estimate.shape} must match direct target shape {direct_target.shape}"
        )
    if not np.all(np.isfinite(q_model_estimate)):
        raise ValueError("q_model_estimate must contain only finite values")

    motives_array = np.asarray(motives, dtype=np.float32)
    batch_size = motives_array.shape[0]
    time_steps = motives_array.shape[1]
    behavior = _coerce_option_probability(
        behavior_probability,
        field_name="behavior_probability",
        batch_size=batch_size,
        time_steps=time_steps,
        strictly_positive=True,
    )
    target = _coerce_option_probability(
        target_probability,
        field_name="target_probability",
        batch_size=batch_size,
        time_steps=time_steps,
        strictly_positive=False,
    )

    rho = target / behavior
    if clip_value is not None:
        clip_value = float(clip_value)
        if clip_value <= 0.0 or not np.isfinite(clip_value):
            raise ValueError(f"clip_value must be positive and finite when provided, got {clip_value}")
        rho = np.minimum(rho, clip_value)

    return (q_model_estimate + rho[:, None] * (direct_target - q_model_estimate)).astype(np.float32)


def _coerce_option_probability(
    values: np.ndarray,
    *,
    field_name: str,
    batch_size: int,
    time_steps: int,
    strictly_positive: bool,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float32)
    if probabilities.ndim == 0:
        probabilities = np.full((batch_size,), float(probabilities), dtype=np.float32)
    elif probabilities.shape == (batch_size,):
        probabilities = probabilities.reshape(batch_size)
    elif probabilities.shape == (batch_size, 1):
        probabilities = probabilities[:, 0]
    elif probabilities.shape == (batch_size, time_steps):
        first = probabilities[:, :1]
        if not np.allclose(probabilities, first, rtol=1e-6, atol=1e-6):
            raise ValueError(
                f"{field_name} must be option-level or constant across trajectory steps"
            )
        probabilities = first[:, 0]
    else:
        raise ValueError(
            f"{field_name} must have shape (), ({batch_size},), ({batch_size}, 1), "
            f"or ({batch_size}, {time_steps}), got {probabilities.shape}"
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{field_name} must contain only finite values")
    if strictly_positive and np.any(probabilities <= 0.0):
        raise ValueError(f"{field_name} must be strictly positive")
    if not strictly_positive and np.any(probabilities < 0.0):
        raise ValueError(f"{field_name} must be non-negative")
    return probabilities.astype(np.float32)

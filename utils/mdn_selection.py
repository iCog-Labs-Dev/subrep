"""Stable MDN-side selection helpers that do not encode final policy logic."""

from __future__ import annotations

import numpy as np


def alpha_to_mean_weights(alpha: np.ndarray) -> np.ndarray:
    """Convert strictly positive Dirichlet parameters to simplex mean weights."""
    alpha = np.asarray(alpha, dtype=np.float32)
    if alpha.ndim not in (1, 2):
        raise ValueError(f"alpha must have shape (K,) or (N, K), got {alpha.shape}")
    if alpha.shape[-1] == 0:
        raise ValueError("alpha must have a non-zero objective dimension")
    if not np.all(np.isfinite(alpha)):
        raise ValueError("alpha must contain only finite values")
    if np.any(alpha <= 0.0):
        raise ValueError("alpha must be strictly positive")

    totals = np.sum(alpha, axis=-1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("alpha sums must be strictly positive")
    return alpha / totals

"""Stable MDN-side selection helpers for utility-driven MDN training."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Dirichlet

from utils.mdn_contracts import CandidateSkillRecord


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


def sample_dirichlet_weights(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample simplex weights and their log-probability from Dirichlet(alpha)."""
    if alpha.ndim not in (1, 2):
        raise ValueError(f"alpha must have shape (K,) or (N, K), got {tuple(alpha.shape)}")
    if alpha.shape[-1] == 0:
        raise ValueError("alpha must have a non-zero objective dimension")
    if not torch.isfinite(alpha).all():
        raise ValueError("alpha must contain only finite values")
    if torch.any(alpha <= 0):
        raise ValueError("alpha must be strictly positive")

    distribution = Dirichlet(alpha)
    weights = distribution.rsample()
    log_prob = distribution.log_prob(weights)
    return weights, log_prob


def score_candidate(candidate: CandidateSkillRecord, weights: np.ndarray) -> float:
    """Score a certified candidate using delta_r + w^T delta_n."""
    if not isinstance(candidate, CandidateSkillRecord):
        raise ValueError(f"candidate must be CandidateSkillRecord, got {type(candidate).__name__}")
    if not candidate.is_certified:
        raise ValueError(f"candidate {candidate.skill_id!r} is not certified and cannot be scored")

    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if weights.shape != (2,):
        raise ValueError(f"weights must have shape (2,), got {weights.shape}")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must contain only finite values")

    return float(candidate.delta_r + float(np.dot(weights, np.asarray(candidate.delta_n, dtype=np.float32))))


def select_best_candidate(candidates: tuple[CandidateSkillRecord, ...] | list[CandidateSkillRecord], weights: np.ndarray) -> tuple[str, float]:
    """Select the highest-scoring certified candidate for the provided weights."""
    certified_candidates = [candidate for candidate in candidates if candidate.is_certified]
    if not certified_candidates:
        raise ValueError("No certified candidates available for selection")

    best_candidate = certified_candidates[0]
    best_score = score_candidate(best_candidate, weights)
    for candidate in certified_candidates[1:]:
        candidate_score = score_candidate(candidate, weights)
        if candidate_score > best_score:
            best_candidate = candidate
            best_score = candidate_score

    return best_candidate.skill_id, best_score

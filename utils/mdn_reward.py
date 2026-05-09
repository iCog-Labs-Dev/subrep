"""Utility-driven reward and policy-loss helpers for MDN training."""

from __future__ import annotations

import numpy as np
import torch


def compute_mdn_utility(
    actual_motives: np.ndarray,
    weights_used: np.ndarray,
    actual_payoff: float | None = None,
    payoff_weight: float = 0.0,
) -> float:
    """Compute scalar MDN utility from actual motives and optional payoff."""
    actual_motives = np.asarray(actual_motives, dtype=np.float32).reshape(-1)
    weights_used = np.asarray(weights_used, dtype=np.float32).reshape(-1)

    if actual_motives.shape != weights_used.shape:
        raise ValueError(
            f"actual_motives shape {actual_motives.shape} must match weights_used shape {weights_used.shape}"
        )
    if not np.all(np.isfinite(actual_motives)):
        raise ValueError("actual_motives must contain only finite values")
    if not np.all(np.isfinite(weights_used)):
        raise ValueError("weights_used must contain only finite values")
    if payoff_weight < 0.0:
        raise ValueError(f"payoff_weight must be non-negative, got {payoff_weight}")

    utility = float(np.dot(weights_used, actual_motives))
    if payoff_weight > 0.0:
        if actual_payoff is None:
            raise ValueError("actual_payoff must be provided when payoff_weight > 0")
        actual_payoff = float(actual_payoff)
        if not np.isfinite(actual_payoff):
            raise ValueError(f"actual_payoff must be finite, got {actual_payoff}")
        utility += float(payoff_weight) * actual_payoff
    return utility


def compute_advantage(
    utility: float,
    baseline_utility: float | None = None,
    running_baseline: float | None = None,
) -> float:
    """Compute scalar advantage from utility and optional baselines."""
    utility = float(utility)
    if not np.isfinite(utility):
        raise ValueError(f"utility must be finite, got {utility}")

    if baseline_utility is not None:
        baseline_utility = float(baseline_utility)
        if not np.isfinite(baseline_utility):
            raise ValueError(f"baseline_utility must be finite, got {baseline_utility}")
        return utility - baseline_utility

    if running_baseline is not None:
        running_baseline = float(running_baseline)
        if not np.isfinite(running_baseline):
            raise ValueError(f"running_baseline must be finite, got {running_baseline}")
        return utility - running_baseline

    return utility


def compute_mdn_policy_loss(log_prob: torch.Tensor, advantage: float | torch.Tensor) -> torch.Tensor:
    """Compute REINFORCE-style MDN policy loss from log-probability and advantage."""
    if log_prob.ndim != 0:
        raise ValueError(f"log_prob must be scalar, got shape {tuple(log_prob.shape)}")
    if not torch.isfinite(log_prob):
        raise ValueError("log_prob must be finite")

    if not isinstance(advantage, torch.Tensor):
        advantage = torch.tensor(float(advantage), dtype=log_prob.dtype, device=log_prob.device)
    else:
        advantage = advantage.to(device=log_prob.device, dtype=log_prob.dtype)

    if advantage.ndim != 0:
        raise ValueError(f"advantage must be scalar, got shape {tuple(advantage.shape)}")
    if not torch.isfinite(advantage):
        raise ValueError("advantage must be finite")

    loss = -advantage.detach() * log_prob
    if not torch.isfinite(loss):
        raise ValueError("policy loss must be finite")
    return loss

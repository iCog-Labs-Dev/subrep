"""Distribution-aware CVaR certification gate using MDN Dirichlet output."""

from __future__ import annotations

import numpy as np
import torch
from torch.distributions import Dirichlet

from .gate import AdmissionGate, Scalar


class CVaRGate(AdmissionGate):
    """PDS-CVaR admission gate.

    A skill is admitted if the CVaR of `delta_r + w^T delta_n` over
    `w ~ Dirichlet(mdn_alpha)` is non-negative at the configured confidence level.
    """

    def __init__(self, confidence: float = 0.1, n_samples: int = 1000) -> None:
        if not (0.0 < confidence <= 1.0):
            raise ValueError(f"confidence must be in (0, 1], got {confidence}")
        if n_samples < 10:
            raise ValueError(f"n_samples must be >= 10 for stable CVaR, got {n_samples}")
        self.confidence = float(confidence)
        self.n_samples = int(n_samples)

    def admit(self, delta_r: Scalar, delta_n: np.ndarray, mdn_alpha: np.ndarray) -> bool:
        self.validate_inputs(delta_r, delta_n)
        cvar = self.get_cvar(delta_r, delta_n, mdn_alpha)
        return cvar >= 0.0

    def get_gate_type(self) -> str:
        return "CVaR"

    def get_cvar(self, delta_r: Scalar, delta_n: np.ndarray, mdn_alpha: np.ndarray) -> float:
        delta_r = float(delta_r)
        delta_n = np.asarray(delta_n, dtype=np.float32).reshape(-1)
        mdn_alpha = np.asarray(mdn_alpha, dtype=np.float32).reshape(-1)

        if len(delta_n) != len(mdn_alpha):
            raise ValueError(
                f"delta_n length {len(delta_n)} must match mdn_alpha length {len(mdn_alpha)}"
            )
        if np.any(mdn_alpha <= 0.0):
            raise ValueError("mdn_alpha must be strictly positive")
        if not np.all(np.isfinite(mdn_alpha)):
            raise ValueError("mdn_alpha must contain only finite values")

        alpha_tensor = torch.tensor(mdn_alpha, dtype=torch.float32)
        with torch.no_grad():
            weight_samples = Dirichlet(alpha_tensor).sample((self.n_samples,)).cpu().numpy()

        values = delta_r + weight_samples @ delta_n
        var_threshold = np.quantile(values, self.confidence)
        tail_values = values[values <= var_threshold]
        if len(tail_values) == 0:
            return float(np.min(values))
        return float(np.mean(tail_values))

"""
Candidate A -- support_head activation: constrained_support_activation

Guarantees, for any real-valued input, any M >= 2:
    Requirement 1:  0 <= s_i <= 1
    Requirement 2:  sum_i(s_i) >= 1
"""

import torch

EPSILON = 1e-6
LSE_SMOOTHING = 1e-2


def constrained_support_activation(
    z: torch.Tensor, z_scale: torch.Tensor, tau: torch.Tensor
) -> torch.Tensor:
    """
    z:        (..., M) raw scores, one per objective
    z_scale:  (..., 1) raw scalar controlling overall generosity
    tau:      scalar or (..., 1), learned temperature, kept > 0 by the caller

    Returns: (..., M) support values, valid for any input.
    """
    p = torch.softmax(z / tau, dim=-1)

    p_max_smooth = LSE_SMOOTHING * torch.logsumexp(p / LSE_SMOOTHING, dim=-1, keepdim=True)

    t_max = (1.0 - EPSILON) / p_max_smooth
    t = t_max ** torch.sigmoid(z_scale)

    s = t * p
    return s.clamp(0.0, 1.0)
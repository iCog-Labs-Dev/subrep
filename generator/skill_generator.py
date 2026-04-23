"""
2-head MLP Skill Generator for SubRep.

This module predicts a scalar payoff and a 2D motive vector from an 8D
observation. It defines the model skeleton only; training logic is added later.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union

import torch
from torch import Tensor, nn

# Type alias for file paths accepted by both `str` and `pathlib.Path`.
PathLike = Union[str, Path]

class SkillGenerator(nn.Module):
    """
    Predict payoff and motives from environment observations.

    Attributes:
        trunk: Shared feature extractor.
        payoff_head: Output head for scalar payoff prediction.
        motive_head: Output head for 2D motive prediction [Safety, Fuel].
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        motive_dim: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.motive_dim = motive_dim

        # Shared representation layers used by both prediction heads.
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Head 1: scalar payoff output.
        self.payoff_head = nn.Linear(hidden_dim, 1)
        # Head 2: 2D motive output [Safety, Fuel].
        self.motive_head = nn.Linear(hidden_dim, motive_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply explicit, stable initialization across all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """
        Run the shared trunk and both heads.
        Args:
            obs: Observation tensor of shape (8,) or batch shape (N, 8).

        Returns:
            payoff: Shape (1,) for single input or (N, 1) for batched input.
            motives: Shape (2,) for single input or (N, 2) for batched input.
        """
        # Guard input rank early for clearer debugging and test behavior.
        if obs.ndim not in (1, 2):
            raise ValueError(
                f"Expected obs with shape ({self.input_dim},) or (N, {self.input_dim}), "
                f"got tensor with shape {tuple(obs.shape)}"
            )

        is_single_input = obs.ndim == 1
        if is_single_input:
            # Validate and convert single sample to batch form for shared forward path.
            if obs.shape[0] != self.input_dim:
                raise ValueError(
                    f"Expected single observation shape ({self.input_dim},), "
                    f"got {tuple(obs.shape)}"
                )
            obs = obs.unsqueeze(0)
        elif obs.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected batched observation shape (N, {self.input_dim}), "
                f"got {tuple(obs.shape)}"
            )

        # First compute shared features, then use separate heads for payoff and motives.
        features = self.trunk(obs)
        payoff = self.payoff_head(features)
        motives = self.motive_head(features)

        # Remove the batch dimension for single observation input.
        if is_single_input:
            payoff = payoff.squeeze(0)
            motives = motives.squeeze(0)

        return payoff, motives

    def save(self, path: PathLike) -> None:
        """Save model weights using PyTorch state_dict serialization."""
        torch.save(self.state_dict(), path)

    def load(self, path: PathLike, map_location: str | torch.device = "cpu") -> None:
        """Load model weights with optional device mapping."""
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

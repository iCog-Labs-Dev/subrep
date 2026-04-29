# Motive Decomposition Network (MDN) 

from __future__ import annotations

import torch
from torch import Tensor, nn


class MotiveDecompositionNetwork(nn.Module):
    """ Predict context-conditioned motive distribution and support geometry."""
    def __init__(
        self,
        input_dim: int = 14,
        num_objectives: int = 2,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        alpha_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError(
                f"Expected num_hidden_layers >= 1, got {num_hidden_layers}"
            )

        self.input_dim = input_dim
        self.num_objectives = num_objectives
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.alpha_epsilon = alpha_epsilon

        trunk_layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 1):
            trunk_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )

        self.trunk = nn.Sequential(*trunk_layers)
        self.distribution_head = nn.Linear(hidden_dim, num_objectives)
        self.support_head = nn.Linear(hidden_dim, num_objectives)
        self.softplus = nn.Softplus()
        self.support_activation = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply stable initialization across all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        if context.ndim not in (1, 2):
            raise ValueError(
                f"Expected context with shape ({self.input_dim},) or "
                f"(N, {self.input_dim}), got tensor with shape {tuple(context.shape)}"
            )

        is_single_input = context.ndim == 1
        if is_single_input:
            if context.shape[0] != self.input_dim:
                raise ValueError(
                    f"Expected single context shape ({self.input_dim},), "
                    f"got {tuple(context.shape)}"
                )
            context = context.unsqueeze(0)
        elif context.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected batched context shape (N, {self.input_dim}), "
                f"got {tuple(context.shape)}"
            )

        features = self.trunk(context)
        weight_params = self.softplus(self.distribution_head(features)) + self.alpha_epsilon
        support_values = self.support_activation(self.support_head(features))

        if is_single_input:
            weight_params = weight_params.squeeze(0)
            support_values = support_values.squeeze(0)

        return weight_params, support_values

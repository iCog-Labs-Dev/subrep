"""
Motive Decomposition Network for SubRep.

The MDN maps 14D context features to a Dirichlet concentration vector
used for weight distribution over motive trade-offs.

This Implementation is for Head 1 (Dirichlet α parameters) only. 
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from torch import Tensor, nn

# Type alias for file paths accepted by both `str` and `pathlib.Path`.
PathLike = Union[str, Path]


class MotiveDecompositionNetwork(nn.Module):
	"""
	Predict Dirichlet concentration parameters from context features.

	The network uses a shared MLP trunk and a single weight-distribution head.
	It accepts a single 14D context vector or a batch of 14D context vectors.
	"""

	def __init__(self, input_dim: int = 14, hidden_dim: int = 64, weight_dim: int = 2) -> None:
		super().__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.weight_dim = weight_dim
		self.eps = 1e-6

		self.trunk = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
		)
		self.weight_head = nn.Linear(hidden_dim, weight_dim)
		self.softplus = nn.Softplus()
		
        

		self._initialize_weights()

	def _initialize_weights(self) -> None:
		"""Apply explicit, stable initialization across all Linear layers."""
		for module in self.modules():
			if isinstance(module, nn.Linear):
				nn.init.xavier_uniform_(module.weight)
				nn.init.zeros_(module.bias)

	def forward(self, context: Tensor) -> Tensor:
		"""
		Run the shared trunk and return Dirichlet concentration parameters.

		Args:
			context: Context tensor of shape (14,) or (N, 14).

		Returns:
			Dirichlet concentration parameters with shape (2,) for a single
			context vector or (N, 2) for a batch.
		"""
		if context.ndim not in (1, 2):
			raise ValueError(
				f"Expected context with shape ({self.input_dim},) or (N, {self.input_dim}), "
				f"got tensor with shape {tuple(context.shape)}"
			)

		is_single_input = context.ndim == 1
		if is_single_input:
			if context.shape[0] != self.input_dim:
				raise ValueError(
					f"Expected single context shape ({self.input_dim},), got {tuple(context.shape)}"
				)
			context = context.unsqueeze(0)
		elif context.shape[1] != self.input_dim:
			raise ValueError(
				f"Expected batched context shape (N, {self.input_dim}), got {tuple(context.shape)}"
			)

		features = self.trunk(context)
		weight_params = self.softplus(self.weight_head(features)) + self.eps

		if is_single_input:
			weight_params = weight_params.squeeze(0)

		return weight_params

	def save(self, path: PathLike) -> None:
		"""Save model weights using PyTorch state_dict serialization."""
		torch.save(self.state_dict(), path)

	def load(self, path: PathLike, map_location: str | torch.device = "cpu") -> None:
		"""Load model weights with optional device mapping."""
		state_dict = torch.load(path, map_location=map_location)
		self.load_state_dict(state_dict)

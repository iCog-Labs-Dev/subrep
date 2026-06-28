"""
mdn_stub.py — Deterministic stub for MotiveDecompositionNetwork.

Provides a pluggable interface for testing the MDNRuntimeSelector
without requiring a trained neural network checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor, nn


class StubMDN(nn.Module):
    """A deterministic wrapper mocking MotiveDecompositionNetwork's API.

    Regardless of the observation provided to forward_inference(), this
    stub returns a predefined alpha vector and predefined support values,
    allowing zero-shot reuse math to be tested predictably.
    """

    def __init__(
        self,
        input_dim: int = 8,
        num_objectives: int = 2,
        fixed_alpha: Optional[list[float]] = None,
        fixed_support_values: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_objectives = num_objectives
        self.device = "cpu"

        # Defaults for a standard testing setup (LunarLander 2-objective)
        if fixed_alpha is None:
            fixed_alpha = [1.0, 1.0]  # implies mean weight [0.5, 0.5]
        if fixed_support_values is None:
            fixed_support_values = [1.0, 1.0]

        if len(fixed_alpha) != num_objectives:
            raise ValueError(
                f"fixed_alpha length ({len(fixed_alpha)}) must match num_objectives ({num_objectives})"
            )
        if len(fixed_support_values) != num_objectives:
            raise ValueError(
                f"fixed_support_values length ({len(fixed_support_values)}) must match num_objectives ({num_objectives})"
            )

        # Pre-allocate tensors on CPU to return identically on every forward pass
        self._alpha_tensor = torch.tensor(fixed_alpha, dtype=torch.float32)
        self._support_tensor = torch.tensor(fixed_support_values, dtype=torch.float32)

    def forward_inference(self, context: Tensor) -> tuple[Tensor, Tensor]:
        """Matches MotiveDecompositionNetwork.forward_inference() signature."""
        # Simple validation matching actual MDN validation rules
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
            # Return flat tensors for single input
            return self._alpha_tensor.clone(), self._support_tensor.clone()
        else:
            if context.shape[1] != self.input_dim:
                raise ValueError(
                    f"Expected batched context shape (N, {self.input_dim}), "
                    f"got {tuple(context.shape)}"
                )
            # Return batched tensors
            batch_size = context.shape[0]
            alpha_batch = self._alpha_tensor.clone().unsqueeze(0).expand(batch_size, -1)
            support_batch = self._support_tensor.clone().unsqueeze(0).expand(batch_size, -1)
            return alpha_batch, support_batch

    def to(self, device: str) -> "StubMDN":  # type: ignore
        """Mock the `.to()` method so MDNRuntimeSelector doesn't crash on device move."""
        self.device = device
        self._alpha_tensor = self._alpha_tensor.to(device)
        self._support_tensor = self._support_tensor.to(device)
        return self


def load_mdn_or_stub(
    checkpoint_path: Union[str, Path],
    input_dim: int = 8,
    num_objectives: int = 2,
    device: Optional[str] = None,
) -> nn.Module:
    """Attempt to load a trained MDN checkpoint. Fallback to StubMDN on failure.

    This enables continuous integration and demo pipelines to run cleanly even
    if the developer has not yet trained a full MDN model in their local environment.
    """
    from generator.mdn_runtime_selector import MDNRuntimeSelector

    path = str(checkpoint_path)
    if os.path.exists(path) and os.path.isfile(path):
        try:
            # We use the selector's loader just to get the actual model 
            # (MDNRuntimeSelector.from_checkpoint instantiates the selector, 
            # we just want the model inside it).
            selector = MDNRuntimeSelector.from_checkpoint(
                checkpoint_path=path,
                input_dim=input_dim,
                num_objectives=num_objectives,
                device=device,
            )
            print(f"[MDN Loader] Successfully loaded actual checkpoint from: {path}")
            return selector.model
        except Exception as e:
            print(f"[MDN Loader] Warning: Failed to load existing checkpoint from '{path}'.")
            print(f"             Exception: {e}")
            print(f"             Falling back to StubMDN.")

    else:
        print(f"[MDN Loader] Missing checkpoint at '{path}'. Falling back to StubMDN.")

    # Fallback to stub
    stub = StubMDN(
        input_dim=input_dim,
        num_objectives=num_objectives,
        fixed_alpha=[2.0, 2.0],  # Middle-ground weight prediction
        fixed_support_values=[1.0, 1.0],  # Standard simplex support
    )
    if device is not None:
        stub = stub.to(device)
    return stub

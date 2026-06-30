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
    
    When a checkpoint is found, this function infers model dimensions directly
    from the checkpoint state_dict, making it robust to different model architectures.
    """
    path = str(checkpoint_path)
    if os.path.exists(path) and os.path.isfile(path):
        try:
            # Load checkpoint and infer dimensions from state_dict
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = checkpoint.get("model_state_dict", checkpoint)
            
            # Infer dimensions from checkpoint weights
            inferred_input_dim = int(state["trunk.0.weight"].shape[1])
            hidden_dim = int(state["trunk.0.weight"].shape[0])
            num_hidden_layers = sum(
                1 for key in state if key.startswith("trunk.") and key.endswith(".weight")
            )
            inferred_num_objectives = int(state["distribution_head.weight"].shape[0])
            skill_embedding = state.get("skill_embedding.weight")
            num_skills = int(skill_embedding.shape[0]) if skill_embedding is not None else 128
            skill_embedding_dim = int(skill_embedding.shape[1]) if skill_embedding is not None else 8
            
            # Create model with inferred dimensions
            from generator.mdn import MotiveDecompositionNetwork
            model = MotiveDecompositionNetwork(
                input_dim=inferred_input_dim,
                num_objectives=inferred_num_objectives,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                num_skills=num_skills,
                skill_embedding_dim=skill_embedding_dim,
            )
            model.load_state_dict(state)
            model.to(torch.device(device or "cpu"))
            model.eval()
            
            print(f"[MDN Loader] Successfully loaded checkpoint from: {path}")
            print(f"[MDN Loader] Inferred dimensions: input={inferred_input_dim}, objectives={inferred_num_objectives}, skills={num_skills}")
            return model
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

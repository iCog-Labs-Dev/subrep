"""Shared checkpoint loading helpers for MotiveDecompositionNetwork."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from generator.mdn import MotiveDecompositionNetwork


def load_mdn_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> MotiveDecompositionNetwork:
    """Load an MDN checkpoint while inferring the saved architecture shape.

    Candidate-set training may use a large skill embedding table, so runtime
    loading must reconstruct the model from checkpoint weights instead of
    assuming default constructor values.
    """
    checkpoint = _load_checkpoint_payload(path, map_location=map_location)
    state = extract_model_state_dict(checkpoint)
    model = build_mdn_from_state_dict(state)
    model.load_state_dict(state)
    model.to(torch.device(map_location))
    model.eval()
    return model


def extract_model_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Return the model state dict from raw or wrapped checkpoint payloads."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    if not isinstance(state, dict):
        raise ValueError("MDN checkpoint must be a state_dict or contain model_state_dict")
    return state


def build_mdn_from_state_dict(
    state: dict[str, torch.Tensor],
) -> MotiveDecompositionNetwork:
    """Instantiate MotiveDecompositionNetwork from state-dict tensor shapes."""
    _require_key(state, "trunk.0.weight")
    _require_key(state, "distribution_head.weight")

    first_trunk_weight = state["trunk.0.weight"]
    distribution_weight = state["distribution_head.weight"]

    input_dim = int(first_trunk_weight.shape[1])
    hidden_dim = int(first_trunk_weight.shape[0])
    num_hidden_layers = sum(
        1 for key in state if key.startswith("trunk.") and key.endswith(".weight")
    )
    num_objectives = int(distribution_weight.shape[0])

    skill_embedding = state.get("skill_embedding.weight")
    if skill_embedding is None:
        num_skills = 128
        skill_embedding_dim = 8
    else:
        num_skills = int(skill_embedding.shape[0])
        skill_embedding_dim = int(skill_embedding.shape[1])

    return MotiveDecompositionNetwork(
        input_dim=input_dim,
        num_objectives=num_objectives,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        num_skills=num_skills,
        skill_embedding_dim=skill_embedding_dim,
    )


def _load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device,
) -> Any:
    return torch.load(path, map_location=map_location)


def _require_key(state: dict[str, torch.Tensor], key: str) -> None:
    if key not in state:
        raise ValueError(f"MDN checkpoint is missing required tensor {key!r}")

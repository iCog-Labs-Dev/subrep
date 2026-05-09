"""Offline entrypoint for utility-driven MDN training from decision records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_trainer import MDNTrainer, MDNTrainerConfig, create_trainer_for_model
from utils.mdn_contracts import MDNDecisionRecord


def train_mdn_from_records(
    records: Iterable[MDNDecisionRecord],
    *,
    checkpoint_path: str = "models/mdn_policy_best.pth",
    seed: int = 0,
    device: Optional[str] = None,
) -> dict[str, float | str]:
    """Train MDN from prebuilt offline decision records and save a checkpoint."""
    records = list(records)
    if not records:
        raise ValueError("train_mdn_from_records requires at least one decision record")

    context_dim = len(records[0].context)
    num_objectives = len(records[0].alpha)
    model = MotiveDecompositionNetwork(input_dim=context_dim, num_objectives=num_objectives)
    trainer = create_trainer_for_model(model, seed=seed, device=device)
    trainer.config.checkpoint_path = checkpoint_path

    metrics = trainer.train_records(records)
    saved_path = trainer.save_checkpoint(checkpoint_path)
    return {**metrics, "checkpoint_path": saved_path}


def train() -> None:
    """Placeholder CLI entrypoint until record ingestion is fully wired."""
    raise NotImplementedError(
        "train() requires a decision-record loading path; use train_mdn_from_records(...) from code for now."
    )


if __name__ == "__main__":
    train()

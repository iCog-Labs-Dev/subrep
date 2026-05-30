"""Offline entrypoints for proposal-conditioned auxiliary MDN training."""

from __future__ import annotations

from typing import Iterable, Optional

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_trainer import (
    AuxiliaryTrainingRecord,
    MDNAuxiliaryTrainer,
    MDNAuxiliaryTrainerConfig,
)


def train_auxiliary_from_records(
    records: Iterable[AuxiliaryTrainingRecord],
    *,
    checkpoint_path: str = "models/mdn_auxiliary_best.pth",
    seed: int = 0,
    device: Optional[str] = None,
    use_ips: bool = False,
) -> dict[str, object]:
    records = list(records)
    if not records:
        raise ValueError("train_auxiliary_from_records requires at least one record")

    context_dim = len(records[0].context)
    num_motives = len(records[0].q_target)
    num_skills = max(record.skill_id for record in records) + 1

    model = MotiveDecompositionNetwork(
        input_dim=context_dim,
        num_skills=num_skills,
        num_objectives=num_motives,
    )
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(
            checkpoint_path=checkpoint_path,
            random_seed=seed,
            use_ips=use_ips,
        ),
        device=device,
    )

    if use_ips:
        return trainer.train_probability_aware_records(records)
    return trainer.train_records(records)

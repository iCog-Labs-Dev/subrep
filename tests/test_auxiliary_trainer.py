from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_trainer import (
    MDNAuxiliaryTrainer,
    MDNAuxiliaryTrainerConfig,
    build_auxiliary_record,
)
from generator.train_mdn_auxiliary import train_auxiliary_from_records


def _baseline_stats() -> dict[str, object]:
    return {
        "baseline_payoff": 1.0,
        "baseline_motives": (0.5, 0.2),
    }


def _synthetic_records() -> list:
    records = []
    for index in range(40):
        context = (0.1,) * 14 if index % 2 == 0 else (0.9,) * 14
        skill_id = 1 if index % 2 == 0 else 2
        payoff = 1.7 if index % 2 == 0 else 1.1
        motives = (0.8, 0.4) if index % 2 == 0 else (0.3, 0.7)
        records.append(
            build_auxiliary_record(
                context=context,
                skill_id=skill_id,
                payoff=payoff,
                motives=motives,
                baseline_stats=_baseline_stats(),
            )
        )
    return records


def test_auxiliary_trainer_runs_and_saves_checkpoint(tmp_path: Path):
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(
            checkpoint_path=str(tmp_path / "mdn_auxiliary_best.pth"),
            max_epochs=10,
            patience=3,
            batch_size=8,
            random_seed=0,
        ),
        device="cpu",
    )

    result = trainer.train_records(_synthetic_records())

    assert Path(result["checkpoint_path"]).exists()
    assert result["best_val_loss"] >= 0.0
    assert result["best_metrics"]["epoch"] >= 1


def test_auxiliary_trainer_improves_gate_accuracy_and_q_error():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(
            max_epochs=15,
            patience=5,
            batch_size=8,
            random_seed=0,
        ),
        device="cpu",
    )

    result = trainer.train_records(_synthetic_records())
    train_metrics = result["best_metrics"]["train"]

    assert 0.0 <= train_metrics["gate_accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))
    assert torch.isfinite(torch.tensor(train_metrics["q_loss"]))


def test_build_auxiliary_record_uses_discounted_target_when_trajectory_exists():
    record = build_auxiliary_record(
        context=(0.1,) * 14,
        skill_id=1,
        payoff=1.2,
        motives=(0.5, 0.2),
        baseline_stats=_baseline_stats(),
        motive_trajectory=[(1.0, 0.0), (0.0, 1.0)],
        gamma=0.5,
    )

    assert len(record.q_target) == 2


def test_build_auxiliary_record_raises_for_ips_without_probabilities():
    try:
        build_auxiliary_record(
            context=(0.1,) * 14,
            skill_id=1,
            payoff=1.2,
            motives=(0.5, 0.2),
            baseline_stats=_baseline_stats(),
            motive_trajectory=[[(1.0, 0.0), (0.0, 1.0)]],
            use_ips=True,
        )
    except ValueError as exc:
        assert "behavior_probability" in str(exc)
    else:
        raise AssertionError("Expected ValueError when IPS probabilities are missing")


def test_build_auxiliary_record_preserves_behavior_probability_when_present():
    record = build_auxiliary_record(
        context=(0.1,) * 14,
        skill_id=1,
        payoff=1.2,
        motives=(0.5, 0.2),
        baseline_stats=_baseline_stats(),
        motive_trajectory=[[(1.0, 0.0), (0.0, 1.0)]],
        behavior_probability=np.array([[0.5, 0.5]], dtype=np.float32),
        target_probability=np.array([[1.0, 1.0]], dtype=np.float32),
        record_behavior_probability=0.5,
        use_ips=True,
    )

    assert record.behavior_probability == 0.5


def test_auxiliary_trainer_raises_when_ips_mode_enabled_without_probability_aware_dataset():
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(use_ips=True, batch_size=8, max_epochs=1),
        device="cpu",
    )

    try:
        trainer.train_records(_synthetic_records())
    except ValueError as exc:
        assert "train_probability_aware_records" in str(exc)
    else:
        raise AssertionError("Expected ValueError when IPS mode is enabled without probability-aware dataset support")


def test_probability_aware_auxiliary_training_path_runs(tmp_path: Path):
    records = []
    for index in range(20):
        records.append(
            build_auxiliary_record(
                context=((0.1,) * 14) if index % 2 == 0 else ((0.9,) * 14),
                skill_id=1 if index % 2 == 0 else 2,
                payoff=1.7 if index % 2 == 0 else 1.1,
                motives=(0.8, 0.4) if index % 2 == 0 else (0.3, 0.7),
                baseline_stats=_baseline_stats(),
                motive_trajectory=[[(1.0, 0.0), (0.0, 1.0)]],
                behavior_probability=np.array([[0.5, 0.5]], dtype=np.float32),
                target_probability=np.array([[1.0, 1.0]], dtype=np.float32),
                record_behavior_probability=0.5,
                use_ips=True,
            )
        )

    result = train_auxiliary_from_records(
        records,
        checkpoint_path=str(tmp_path / "mdn_auxiliary_ips_best.pth"),
        seed=0,
        device="cpu",
        use_ips=True,
    )

    assert Path(result["checkpoint_path"]).exists()
    assert result["best_val_loss"] >= 0.0

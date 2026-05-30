from __future__ import annotations

from pathlib import Path

import torch

from generator.train_mdn import train_mdn_from_rollout_records


def _baseline_stats() -> dict[str, object]:
    return {
        "baseline_payoff": 1.0,
        "baseline_motives": (0.5, 0.2),
    }


def _rollout_records() -> tuple[dict[str, object], ...]:
    return (
        {
            "obs": [0.1] * 8,
            "payoff": 1.7,
            "motives": [0.8, 0.4],
            "skill_id": "skill_a",
            "terminated": True,
        },
        {
            "obs": [0.1] * 8,
            "payoff": 1.1,
            "motives": [0.3, 0.7],
            "skill_id": "skill_b",
            "terminated": True,
        },
        {
            "obs": [0.2] * 8,
            "payoff": 1.5,
            "motives": [0.7, 0.3],
            "skill_id": "skill_c",
            "terminated": True,
        },
        {
            "obs": [0.2] * 8,
            "payoff": 1.2,
            "motives": [0.6, 0.5],
            "skill_id": "skill_d",
            "terminated": True,
        },
    )


def test_train_mdn_from_rollout_records_runs_and_saves_checkpoint(tmp_path: Path):
    checkpoint_path = tmp_path / "mdn_policy_best.pth"
    metrics = train_mdn_from_rollout_records(
        rollout_records=_rollout_records(),
        baseline_stats=_baseline_stats(),
        checkpoint_path=str(checkpoint_path),
        seed=0,
        device="cpu",
    )

    assert checkpoint_path.exists()
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))

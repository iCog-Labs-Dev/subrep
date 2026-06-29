from __future__ import annotations

import numpy as np
import pytest

from generator.evaluate_mdn_candidate_sets import evaluate_mdn_candidate_sets
from generator.mdn import MotiveDecompositionNetwork

import torch


def _write_candidate_set(path, context_value: float) -> None:
    np.savez(
        path,
        context=np.array([context_value] * 8, dtype=np.float32),
        candidate_skill_ids=np.array(["ppo_deterministic", "random"]),
        candidate_payoffs=np.array([1.7, 0.4], dtype=np.float32),
        candidate_motives=np.array([[0.8, 0.4], [0.1, 0.2]], dtype=np.float32),
    )


def test_evaluate_mdn_candidate_sets_returns_metrics(tmp_path):
    data_dir = tmp_path / "candidate_sets"
    data_dir.mkdir()
    _write_candidate_set(data_dir / "eval_00001.npz", 0.1)
    _write_candidate_set(data_dir / "eval_00002.npz", 0.2)

    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2, num_skills=16)
    for parameter in model.parameters():
        parameter.data.zero_()
    checkpoint_path = tmp_path / "mdn_policy_best.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "auxiliary_target_normalization": {
                "enabled": True,
                "mean": (0.45, 0.3),
                "std": (1.0, 1.0),
                "count": 4,
            },
            "auxiliary_q_calibration": {
                "enabled": True,
                "type": "affine",
                "slope": (1.0, 1.0),
                "intercept": (0.0, 0.0),
                "count": 4,
            },
        },
        checkpoint_path,
    )

    metrics = evaluate_mdn_candidate_sets(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        pattern="eval_*.npz",
        baseline_stats={"baseline_payoff": 0.0, "baseline_motives": (0.0, 0.0)},
        device="cpu",
        bootstrap_samples=10,
    )

    assert metrics["candidate_outcomes"] == 4.0
    assert metrics["contexts_total"] == 2.0
    assert metrics["contexts_evaluated"] == 2.0
    assert metrics["avg_certified_candidates"] >= 1.0
    assert metrics["mean_score_lift_vs_random"] >= 0.0
    assert "mean_score_lift_vs_ppo" in metrics
    assert "mean_predicted_weight_regret" in metrics
    assert 0.0 <= metrics["balanced_top1_accuracy"] <= 1.0
    assert "score_lift_vs_ppo_ci95_low" in metrics
    assert "score_lift_vs_ppo_ci95_high" in metrics
    assert "std_alpha_weight_0" in metrics
    assert metrics["q_target_normalization_enabled"] == 1.0
    assert metrics["q_calibration_enabled"] == 1.0
    assert 0.0 <= metrics["gate_accuracy"] <= 1.0
    assert 0.0 <= metrics["gate_precision"] <= 1.0
    assert 0.0 <= metrics["gate_recall"] <= 1.0
    assert 0.0 <= metrics["gate_f1"] <= 1.0
    assert metrics["q_motive_mse"] == pytest.approx(0.06625)
    assert metrics["q_motive_mae"] == pytest.approx(0.225)
    assert metrics["q_motive_mse_0"] == pytest.approx(0.1225)
    assert metrics["q_motive_mse_1"] == pytest.approx(0.01)
    assert metrics["q_motive_mae_0"] == pytest.approx(0.35)
    assert metrics["q_motive_mae_1"] == pytest.approx(0.1)

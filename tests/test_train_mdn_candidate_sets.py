from __future__ import annotations

import numpy as np
import torch

from generator.train_mdn_candidate_sets import train_mdn_from_candidate_set_directory


def _write_candidate_set(path, context_value: float) -> None:
    np.savez(
        path,
        context=np.array([context_value] * 8, dtype=np.float32),
        context_seed=42,
        candidate_skill_ids=np.array(["ppo_deterministic", "random"]),
        candidate_payoffs=np.array([1.7, 0.4], dtype=np.float32),
        candidate_motives=np.array([[0.8, 0.4], [0.1, 0.2]], dtype=np.float32),
        terminated_flags=np.array([True, True], dtype=np.bool_),
        behavior_probabilities=np.array([1.0, 0.25], dtype=np.float32),
        step_counts=np.array([10, 10], dtype=np.int32),
        stop_reasons=np.array(["terminated", "terminated"]),
    )


def test_train_mdn_from_candidate_set_directory_runs_and_saves_checkpoints(tmp_path):
    data_dir = tmp_path / "candidate_sets"
    data_dir.mkdir()
    _write_candidate_set(data_dir / "candidate_set_00001.npz", 0.1)
    _write_candidate_set(data_dir / "candidate_set_00002.npz", 0.2)

    result = train_mdn_from_candidate_set_directory(
        data_dir=data_dir,
        baseline_stats={"baseline_payoff": 0.0, "baseline_motives": (0.0, 0.0)},
        seed=0,
        device="cpu",
        policy_checkpoint_path=str(tmp_path / "mdn_policy_best.pth"),
        auxiliary_checkpoint_path=str(tmp_path / "mdn_auxiliary_best.pth"),
        skill_id_bucket_count=128,
        q_loss="huber",
        calibrate_auxiliary_q=True,
    )

    assert result["candidate_outcomes"] == 4
    assert (tmp_path / "mdn_policy_best.pth").exists()
    assert (tmp_path / "mdn_auxiliary_best.pth").exists()
    assert "loss" in result["policy"]
    assert result["auxiliary"]["best_val_loss"] >= 0.0
    assert result["auxiliary_target_normalization"]["enabled"] is True
    assert result["auxiliary_target_normalization"]["count"] == 4
    assert result["auxiliary_q_calibration"]["enabled"] is True
    assert result["auxiliary_q_calibration"]["count"] == 4
    assert len(result["auxiliary_q_calibration"]["slope"]) == 2

    checkpoint = torch.load(tmp_path / "mdn_policy_best.pth", map_location="cpu")
    assert checkpoint["auxiliary_target_normalization"] == result["auxiliary_target_normalization"]
    assert checkpoint["auxiliary_q_calibration"] == result["auxiliary_q_calibration"]

    auxiliary_checkpoint = torch.load(tmp_path / "mdn_auxiliary_best.pth", map_location="cpu")
    assert auxiliary_checkpoint["config"]["q_loss"] == "huber"

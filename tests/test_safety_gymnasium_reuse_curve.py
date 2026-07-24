from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.safety_gymnasium_reuse_curve import build_safety_gymnasium_reuse_curve


def _write_rollout(path: Path, seed: int) -> None:
    candidate_ids = np.asarray(
        [
            "zero_action",
            "safe_task_candidate",
            "safety_candidate",
            "unsafe_candidate",
        ]
    )
    candidate_payoffs = np.asarray([1.0, 2.0, 1.4, 3.0], dtype=np.float32)
    candidate_motives = np.asarray(
        [
            [-0.0, 1.0],
            [-0.0, 2.0],
            [-0.0, 1.4],
            [-4.0, 3.0],
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        env_id=np.asarray("SafetyPointGoal1-v0"),
        context=np.zeros(4, dtype=np.float32),
        context_seed=np.asarray(seed, dtype=np.int32),
        candidate_skill_ids=candidate_ids,
        candidate_payoffs=candidate_payoffs,
        candidate_motives=candidate_motives,
        candidate_safety_costs=-candidate_motives[:, 0],
        candidate_task_returns=candidate_motives[:, 1],
        terminated_flags=np.asarray([False] * len(candidate_ids)),
        step_counts=np.asarray([20] * len(candidate_ids), dtype=np.int32),
        stop_reasons=np.asarray(["max_steps"] * len(candidate_ids)),
    )


def test_reuse_curve_writes_outputs_and_tracks_retraining_cost(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_rollout(rollout_dir / "rollout_00001.npz", seed=42)
    _write_rollout(rollout_dir / "rollout_00002.npz", seed=43)

    plot_path = tmp_path / "reuse_curve.png"
    json_path = tmp_path / "reuse_curve.json"
    summary = build_safety_gymnasium_reuse_curve(
        rollout_dirs=[rollout_dir],
        output_path=plot_path,
        summary_json_path=json_path,
        pds_epsilon=1.0,
        shift_points=(0.0, 0.5, 1.0),
        baseline_retraining_steps=1000,
    )

    assert plot_path.exists()
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["total_contexts"] == 2
    assert len(loaded["curve"]) == 3
    assert summary["mean_success_rate"] == 1.0
    assert all(point["success_rate"] == 1.0 for point in loaded["curve"])
    assert all(point["subrep_retraining_steps"] == 0 for point in loaded["curve"])
    assert all(point["baseline_retraining_steps"] == 1000 for point in loaded["curve"])

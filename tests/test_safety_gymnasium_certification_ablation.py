from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from utils.safety_gymnasium_certification_ablation import (
    build_safety_gymnasium_certification_ablation,
)


def _write_rollout(path: Path, seed: int) -> None:
    candidate_ids = np.asarray(
        [
            "zero_action",
            "safe_candidate",
            "unsafe_high_task_candidate",
        ]
    )
    candidate_payoffs = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    candidate_motives = np.asarray(
        [
            [-0.0, 1.0],
            [-0.0, 2.0],
            [-5.0, 6.0],
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


def test_certification_ablation_shows_blocked_uncertified_selection(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_rollout(rollout_dir / "rollout_00001.npz", seed=42)
    _write_rollout(rollout_dir / "rollout_00002.npz", seed=43)

    json_path = tmp_path / "ablation.json"
    summary = build_safety_gymnasium_certification_ablation(
        rollout_dirs=[rollout_dir],
        summary_json_path=json_path,
        pds_epsilon=1.0,
    )

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    task = loaded["queries"]["task_focused"]
    assert summary["total_contexts"] == 2
    assert task["blocked_without_certification_selection_count"] == 2
    assert task["without_certification_mean_safety_cost"] > task["with_certification_mean_safety_cost"]
    assert task["safety_cost_reduction_from_certification"] > 0.0

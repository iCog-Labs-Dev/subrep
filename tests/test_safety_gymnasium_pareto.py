from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils.safety_gymnasium_pareto import build_safety_gymnasium_pareto_report


def _write_rollout(path: Path, seed: int) -> None:
    candidate_ids = np.asarray(
        [
            "zero_action",
            "safe_candidate",
            "unsafe_high_return",
            "ppo_deterministic",
            "ppo_lagrangian_deterministic",
        ]
    )
    candidate_payoffs = np.asarray([1.0, 2.0, 3.0, 1.5, 1.8], dtype=np.float32)
    candidate_motives = np.asarray(
        [
            [-0.0, 1.0],
            [-0.0, 2.0],
            [-4.0, 3.0],
            [-1.5, 1.5],
            [-0.2, 1.8],
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


def test_pareto_report_summarizes_methods_and_writes_outputs(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_rollout(rollout_dir / "rollout_00001.npz", seed=42)
    _write_rollout(rollout_dir / "rollout_00002.npz", seed=43)

    plot_path = tmp_path / "pareto.png"
    json_path = tmp_path / "pareto.json"
    summary = build_safety_gymnasium_pareto_report(
        rollout_dirs=[rollout_dir],
        output_path=plot_path,
        summary_json_path=json_path,
        pds_epsilon=1.0,
    )

    assert plot_path.exists()
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["methods"]["subrep"]["available"] is True
    assert loaded["methods"]["ppo"]["available"] is True
    assert loaded["methods"]["ppo_lagrangian"]["available"] is True
    assert summary["methods"]["subrep"]["mean_safety_cost"] == 0.0
    assert summary["methods"]["subrep"]["mean_task_return"] == 2.0
    assert summary["methods"]["ppo"]["mean_safety_cost"] == 1.5
    assert summary["methods"]["ppo_lagrangian"]["mean_safety_cost"] == pytest.approx(0.2)

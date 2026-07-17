from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from utils.safety_gymnasium_pipeline import run_safety_gymnasium_certification_pipeline


def _write_rollout_file(path: Path) -> None:
    candidate_skill_ids = np.asarray(
        [
            "zero_action",
            "safe_balanced",
            "task_heavy",
            "bounded_tradeoff",
            "unsafe_costly",
        ]
    )
    candidate_payoffs = np.asarray([1.0, 2.0, 4.0, 1.5, 1.2], dtype=np.float32)
    candidate_motives = np.asarray(
        [
            [-1.0, 1.0],  # baseline: safety cost 1.0, task return 1.0
            [-0.5, 2.0],  # CDS: safer and better task return
            [-3.5, 4.0],  # CDS: task gain offsets safety-cost increase
            [-2.2, 1.5],  # PDS: bounded safety trade-off
            [-4.0, 1.2],  # rejected: too much safety cost for small task gain
        ],
        dtype=np.float32,
    )
    np.savez(
        path,
        env_id=np.asarray("SafetyPointGoal1-v0"),
        context=np.zeros(4, dtype=np.float32),
        context_seed=np.asarray(101, dtype=np.int32),
        candidate_skill_ids=candidate_skill_ids,
        candidate_payoffs=candidate_payoffs,
        candidate_motives=candidate_motives,
        candidate_safety_costs=-candidate_motives[:, 0],
        candidate_task_returns=candidate_motives[:, 1],
        terminated_flags=np.asarray([False] * len(candidate_skill_ids)),
        step_counts=np.asarray([50] * len(candidate_skill_ids), dtype=np.int32),
        stop_reasons=np.asarray(["max_steps"] * len(candidate_skill_ids)),
    )


def test_safety_gymnasium_rollouts_are_certified_and_reported(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_rollout_file(rollout_dir / "safety_rollout_00001.npz")

    result = run_safety_gymnasium_certification_pipeline(
        rollout_dir=rollout_dir,
        pds_epsilon=1.0,
        cert_file=tmp_path / "certificates.metta",
        library_file=tmp_path / "library.json",
        report_json_path=tmp_path / "report.json",
        report_md_path=tmp_path / "report.md",
    )

    stats = result.stats
    assert stats["contexts_processed"] == 1
    assert stats["candidate_outcomes_loaded"] == 5
    assert stats["candidate_outcomes_certified"] == 4
    assert stats["admitted"] == 3
    assert stats["rejected"] == 1
    assert stats["cds_pass_count"] == 2
    assert stats["pds_pass_count"] == 1
    assert stats["cert_store_count"] == stats["library_size"] == 3
    assert result.cert_store.count() == result.library.count() == 3

    stored_ids = {cert.skill_id for cert in result.cert_store.load_all()}
    assert any("safe_balanced" in skill_id for skill_id in stored_ids)
    assert any("task_heavy" in skill_id for skill_id in stored_ids)
    assert any("bounded_tradeoff" in skill_id for skill_id in stored_ids)
    assert not any("unsafe_costly" in skill_id for skill_id in stored_ids)

    report_json = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report_json["safety_cost_behavior"]["safety_cost_increase_count"] == 3
    assert "baseline_comparison" in report_json
    assert report_json["baseline_comparison"]["task_focused"]["ppo_candidate_available"] is False
    assert report_json["baseline_comparison"]["task_focused"]["mean_lift_vs_random_candidate"] > 0.0
    assert report_json["failure_reasons"]
    assert (tmp_path / "report.md").exists()


def test_safety_gymnasium_zero_shot_reuse_changes_with_motive_weights(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    _write_rollout_file(rollout_dir / "safety_rollout_00001.npz")

    result = run_safety_gymnasium_certification_pipeline(
        rollout_dir=rollout_dir,
        pds_epsilon=1.0,
        cert_file=tmp_path / "certificates.metta",
        library_file=tmp_path / "library.json",
        report_json_path=tmp_path / "report.json",
        report_md_path=tmp_path / "report.md",
    )

    reuse = result.stats["zero_shot_reuse"]
    assert reuse["retraining_performed"] is False
    assert reuse["task_focused_admissible_count"] == 3
    assert reuse["safety_focused_admissible_count"] == 3
    assert "task_heavy" in reuse["task_focused_selected_skill"]
    assert "safe_balanced" in reuse["safety_focused_selected_skill"]
    assert reuse["selection_changed"] is True

    comparison = result.stats["baseline_comparison"]
    assert comparison["task_focused"]["mean_subrep_certified_score"] > comparison["task_focused"]["mean_zero_action_score"]
    assert comparison["task_focused"]["mean_subrep_certified_score"] > comparison["task_focused"]["mean_random_candidate_score"]
    assert comparison["safety_focused"]["mean_subrep_certified_score"] > comparison["safety_focused"]["mean_random_candidate_score"]


def test_safety_gymnasium_pipeline_requires_baseline_candidate(tmp_path):
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    path = rollout_dir / "missing_baseline.npz"
    np.savez(
        path,
        env_id=np.asarray("SafetyPointGoal1-v0"),
        context_seed=np.asarray(101, dtype=np.int32),
        candidate_skill_ids=np.asarray(["random"]),
        candidate_payoffs=np.asarray([1.0], dtype=np.float32),
        candidate_motives=np.asarray([[-1.0, 1.0]], dtype=np.float32),
        candidate_safety_costs=np.asarray([1.0], dtype=np.float32),
        candidate_task_returns=np.asarray([1.0], dtype=np.float32),
        step_counts=np.asarray([10], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="baseline candidate"):
        run_safety_gymnasium_certification_pipeline(
            rollout_dir=rollout_dir,
            cert_file=tmp_path / "certificates.metta",
            library_file=tmp_path / "library.json",
            report_json_path=tmp_path / "report.json",
            report_md_path=tmp_path / "report.md",
        )

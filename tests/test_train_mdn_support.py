from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.train_mdn_support import run_support_head_experiment
from utils.probability_aware_logs import save_probability_aware_log
from utils.weight_set_store import WeightSetStore


def test_support_experiment_writes_reproducible_split_and_reports(tmp_path):
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    base_checkpoint = tmp_path / "base.pth"
    torch.save({"model_state_dict": model.state_dict()}, base_checkpoint)

    store = WeightSetStore(num_objectives=2)
    for index in range(6):
        first = 0.25 + index * 0.1
        store.observe_certified_weight(
            np.full(8, index / 10.0, dtype=np.float32),
            np.array([first, 1.0 - first], dtype=np.float32),
        )
    store_path = tmp_path / "weights.json"
    store.save(store_path)
    output_dir = tmp_path / "evaluation"
    output_checkpoint = tmp_path / "support.pth"

    report = run_support_head_experiment(
        base_checkpoint_path=base_checkpoint,
        weight_store_path=store_path,
        output_dir=output_dir,
        output_checkpoint_path=output_checkpoint,
        max_epochs=2,
        early_stopping_patience=1,
        seed=9,
        device="cpu",
    )

    assert report["status"] == "completed"
    assert output_checkpoint.exists()
    assert (output_dir / "support_train.json").exists()
    assert (output_dir / "support_validation.json").exists()
    assert (output_dir / "support_test.json").exists()
    assert (output_dir / "split_manifest.json").exists()
    assert (output_dir / "support_experiment.json").exists()
    assert (output_dir / "support_test_report.md").exists()

    serialized = json.loads((output_dir / "support_experiment.json").read_text(encoding="utf-8"))
    assert serialized["test"]["arms"]["learned"]["cds_agreement_rate"] is None
    assert serialized["test"]["arms"]["learned"]["support_value_mse"] >= 0.0
    assert "train_mean_constant" in serialized["test"]["arms"]


def test_support_experiment_runs_end_to_end_from_runtime_logs(tmp_path):
    model = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    base_checkpoint = tmp_path / "base.pth"
    torch.save({"model_state_dict": model.state_dict()}, base_checkpoint)
    log_dir = tmp_path / "runtime_logs"

    for index in range(6):
        context_index = index // 2
        first = 0.3 if index % 2 == 0 else 0.7
        weights = np.array([first, 1.0 - first], dtype=np.float32)
        save_probability_aware_log(
            log_dir / f"runtime_{index:05d}.npz",
            context=np.full(8, context_index / 10.0, dtype=np.float32),
            alpha=weights,
            support_values=np.ones(2, dtype=np.float32),
            weights_used=weights,
            candidate_skill_ids=np.array(["safe", "unsafe"]),
            candidate_delta_r=np.array([0.2, -0.3], dtype=np.float32),
            candidate_delta_n=np.array([[0.1, 0.0], [-0.2, -0.1]], dtype=np.float32),
            candidate_accept_labels=np.array([True, False]),
            candidate_gate_types=np.array(["CDS", "CDS"]),
            candidate_admission_margins=np.array([0.2, -0.4], dtype=np.float32),
            selected_candidate_index=np.asarray(0, dtype=np.int32),
            selected_skill_id=np.asarray("safe"),
            selected_score=np.asarray(0.25, dtype=np.float32),
            behavior_probability=np.asarray(1.0, dtype=np.float32),
            actual_payoff=np.asarray(0.2, dtype=np.float32),
            actual_motives=np.array([0.1, 0.0], dtype=np.float32),
        )

    output_dir = tmp_path / "runtime_evaluation"
    output_checkpoint = tmp_path / "runtime_support.pth"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "generator.train_mdn_support",
            "--base-checkpoint",
            str(base_checkpoint),
            "--runtime-log-dir",
            str(log_dir),
            "--output-dir",
            str(output_dir),
            "--output-checkpoint",
            str(output_checkpoint),
            "--max-epochs",
            "2",
            "--early-stopping-patience",
            "1",
            "--seed",
            "4",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((output_dir / "support_experiment.json").read_text(encoding="utf-8"))

    assert report["source_type"] == "runtime_logs"
    assert output_checkpoint.exists()
    assert report["test"]["arms"]["learned"]["candidate_context_coverage"] == 1.0
    assert report["test"]["arms"]["learned"]["reuse_motive_shift_context_coverage"] == 1.0
    assert report["test"]["arms"]["learned"]["candidate_admission_decisions"] > 0.0
    assert report["test"]["arms"]["learned"]["reuse_shift_opportunities"] > 0.0

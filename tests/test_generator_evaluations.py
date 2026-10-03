import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from generator import compare_dataset_sizes
from generator.evaluate_generator_mse import evaluate_dataset
from generator.evaluate_generator_report import build_report, load_candidate_set_files


class ZeroPredictionModel(torch.nn.Module):
    def forward(self, obs):
        batch_size = obs.shape[0] if obs.ndim == 2 else 1
        return torch.zeros(batch_size, 1), torch.zeros(batch_size, 2)


class ContextPredictionModel(torch.nn.Module):
    def forward(self, context):
        payoff = context[0].reshape(1)
        motives = context[:2]
        return payoff, motives


def _write_rollout(path, payoff, motives):
    np.savez(
        path,
        obs=np.zeros(8, dtype=np.float32),
        payoff=np.float32(payoff),
        motives=np.asarray(motives, dtype=np.float32),
    )


def test_compare_dataset_split_is_reproducible_disjoint_and_exact():
    files = [f"episode_{index}.npz" for index in range(40)]
    records = [f"record_{index}" for index in range(40)]
    dataset = SimpleNamespace(files=files, data=records)

    first = compare_dataset_sizes.create_dataset_split(dataset, size=40, seed=7)
    repeated = compare_dataset_sizes.create_dataset_split(dataset, size=40, seed=7)

    assert first == repeated
    assert (len(first["train_files"]), len(first["val_files"]), len(first["test_files"])) == (30, 5, 5)
    assert not (set(first["train_files"]) & set(first["val_files"]))
    assert not (set(first["train_files"]) & set(first["test_files"]))
    assert not (set(first["val_files"]) & set(first["test_files"]))
    assert set(first["train_files"] + first["val_files"] + first["test_files"]) == set(files)

    with pytest.raises(ValueError, match="exceeds available files"):
        compare_dataset_sizes.create_dataset_split(dataset, size=41, seed=7)


def test_compare_main_uses_documented_default_sizes(monkeypatch, tmp_path):
    files = [f"episode_{index}.npz" for index in range(7000)]
    dataset = SimpleNamespace(files=files, data=list(range(7000)))
    monkeypatch.setattr(compare_dataset_sizes, "SkillDataset", lambda _: dataset)
    monkeypatch.setattr(compare_dataset_sizes, "save_dataset_files", lambda *args: None)
    monkeypatch.setattr(
        compare_dataset_sizes,
        "run_training_loop",
        lambda model, **kwargs: (model.state_dict(), [], [], 1),
    )
    monkeypatch.setattr(
        compare_dataset_sizes,
        "evaluate_on_test_set",
        lambda model, records: {"payoff_mse": 1.0, "safety_mse": 2.0, "fuel_mse": 3.0},
    )
    monkeypatch.setattr(compare_dataset_sizes, "save_combined_curves_plot", lambda *args: None)
    monkeypatch.setattr(compare_dataset_sizes, "save_mse_vs_size_plot", lambda *args: None)
    monkeypatch.setattr(compare_dataset_sizes.torch.cuda, "is_available", lambda: False)

    output_dir = tmp_path / "plots"
    rollout_dir = tmp_path / "rollouts"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_dataset_sizes",
            "--data-dir",
            "unused",
            "--output-dir",
            str(output_dir),
            "--rollout-output-dir",
            str(rollout_dir),
        ],
    )

    compare_dataset_sizes.main()

    summary = json.loads((output_dir / "dataset_size_comparison_results_epochs150.json").read_text())
    assert [result["dataset_size"] for result in summary["results"]] == [1000, 3000, 7000]
    assert [
        (result["train_size"], result["validation_size"], result["test_size"])
        for result in summary["results"]
    ] == [(750, 125, 125), (2250, 375, 375), (5250, 875, 875)]


def test_evaluate_dataset_uses_test_records_and_train_only_baseline(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    _write_rollout(data_dir / "train_a.npz", 0.0, [0.0, 2.0])
    _write_rollout(data_dir / "train_b.npz", 2.0, [2.0, 4.0])
    _write_rollout(data_dir / "val.npz", 100.0, [100.0, 100.0])
    _write_rollout(data_dir / "test_a.npz", 4.0, [4.0, 6.0])
    _write_rollout(data_dir / "test_b.npz", 6.0, [6.0, 8.0])
    manifest = {
        "assignment": {
            "train_a.npz": "train",
            "train_b.npz": "train",
            "val.npz": "val",
            "test_a.npz": "test",
            "test_b.npz": "test",
        }
    }

    result = evaluate_dataset(ZeroPredictionModel(), str(data_dir), manifest, str(tmp_path / "report"))

    assert result["n_test_records"] == 2
    assert result["model"]["payoff_mse"] == pytest.approx(26.0)
    assert result["mean_baseline"]["payoff_mse"] == pytest.approx(17.0)
    assert result["mean_baseline"]["safety_mse"] == pytest.approx(17.0)
    assert result["mean_baseline"]["fuel_mse"] == pytest.approx(17.0)
    assert result["model_beats_baseline"] == {"payoff": False, "safety": False, "fuel": False}

    saved = json.loads((tmp_path / "report" / "generator_mse_report.json").read_text())
    assert saved == result


def test_report_loader_reads_candidate_set_records(tmp_path):
    np.savez(
        tmp_path / "candidate.npz",
        context=np.arange(8, dtype=np.float32),
        context_seed=np.int64(123),
        candidate_skill_ids=np.asarray(["ppo_deterministic", "noop"]),
        candidate_payoffs=np.asarray([2.0, -2.0]),
        candidate_motives=np.asarray([[1.0, 1.0], [-1.0, -1.0]]),
    )

    records = load_candidate_set_files(str(tmp_path))

    assert len(records) == 1
    assert records[0]["context_seed"] == 123
    assert records[0]["candidate_skill_ids"] == ["ppo_deterministic", "noop"]
    assert records[0]["source_file"] == "candidate.npz"

    with pytest.raises(FileNotFoundError, match="No candidate-set"):
        load_candidate_set_files(str(tmp_path / "missing"))


def test_report_aggregates_certification_and_prediction_metrics():
    records = [
        {
            "context": [1.0, 1.0, 0, 0, 0, 0, 0, 0],
            "context_seed": 10,
            "candidate_skill_ids": ["ppo_deterministic", "noop"],
            "candidate_payoffs": np.asarray([2.0, -2.0]),
            "candidate_motives": np.asarray([[1.0, 1.0], [-1.0, -1.0]]),
        },
        {
            "context": [2.0, 2.0, 0, 0, 0, 0, 0, 0],
            "context_seed": 11,
            "candidate_skill_ids": ["ppo_deterministic", "noop"],
            "candidate_payoffs": np.asarray([3.0, -1.0]),
            "candidate_motives": np.asarray([[2.0, 2.0], [-1.0, -1.0]]),
        },
    ]

    report = build_report(
        ContextPredictionModel(),
        records,
        {"baseline_payoff": 0.0, "baseline_motives": [0.0, 0.0]},
    )

    assert report["n_contexts_evaluated"] == 2
    assert report["n_unique_context_seeds"] == 2
    assert report["context_seed_range"] == "10-11"
    assert report["per_skill"]["ppo_deterministic"]["cds_admission_rate"] == 1.0
    assert report["per_skill"]["ppo_deterministic"]["pds_admission_rate"] == 1.0
    assert report["per_skill"]["noop"]["pds_admission_rate"] == 0.0
    assert report["per_skill"]["noop"]["rejection_reasons"] == {"rejected_by_PDS": 2}
    assert report["generator_predicted_vs_actual_payoff_correlation"] == pytest.approx(1.0)
    assert report["generator_avg_predicted_payoff"] == pytest.approx(1.5)
    assert report["generator_avg_actual_payoff"] == pytest.approx(2.5)
    assert "noop" in report["baseline_comparison"]
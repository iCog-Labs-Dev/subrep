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


class DeviceCheckingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        assert obs.device == self.anchor.device
        return self.anchor.expand(obs.shape[0], 1), self.anchor.expand(obs.shape[0], 2)


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


def _rollout_record(context_id, payoff=0.0, motives=(0.0, 0.0)):
    observation = torch.zeros(8, dtype=torch.float32)
    observation[0] = float(context_id)
    return (
        observation,
        torch.tensor([payoff], dtype=torch.float32),
        torch.tensor(motives, dtype=torch.float32),
    )


def test_split_reuses_holdouts_and_keeps_shared_contexts_together():
    files = [f"episode_{context}_{policy}.npz" for context in range(20) for policy in range(2)]
    records = [_rollout_record(context) for context in range(20) for _ in range(2)]
    dataset = SimpleNamespace(files=files, data=records)

    fixed = compare_dataset_sizes.create_holdout_split(dataset, seed=7)
    repeated_fixed = compare_dataset_sizes.create_holdout_split(dataset, seed=7)
    small = compare_dataset_sizes.create_dataset_split(fixed, size=8)
    large = compare_dataset_sizes.create_dataset_split(fixed, size=20)

    assert fixed["test_files"] == repeated_fixed["test_files"]
    assert fixed["validation_files"] == repeated_fixed["validation_files"]
    assert len(fixed["training_pool_files"]) == 32
    assert len(fixed["validation_files"]) == 4
    assert len(fixed["test_files"]) == 4
    assert set(small["train_files"]) <= set(large["train_files"])
    assert small["validation_files"] == large["validation_files"]
    assert small["test_files"] == large["test_files"]

    context_sets = [
        {round(float(record[0][0]), 6) for record in records_in_split}
        for records_in_split in (
            large["train_records"],
            large["validation_records"],
            large["test_records"],
        )
    ]
    assert not (context_sets[0] & context_sets[1])
    assert not (context_sets[0] & context_sets[2])
    assert not (context_sets[1] & context_sets[2])

    with pytest.raises(ValueError, match="exceeds the available training pool"):
        compare_dataset_sizes.create_dataset_split(fixed, size=33)


def test_save_dataset_files_uses_shared_holdout_and_per_size_train_paths(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    train_file = source / "train.npz"
    validation_file = source / "validation.npz"
    test_file = source / "test.npz"
    for path in (train_file, validation_file, test_file):
        path.write_bytes(b"record")

    split = {
        "train_files": [str(train_file)],
        "validation_files": [str(validation_file)],
        "test_files": [str(test_file)],
    }
    output_dir = tmp_path / "rollouts"
    compare_dataset_sizes.save_dataset_files({1000: split}, output_dir)

    assert (output_dir / "fixed_split" / "test" / "test.npz").exists()
    assert (output_dir / "fixed_split" / "validation" / "validation.npz").exists()
    assert (output_dir / "size_1000" / "train" / "train.npz").exists()
    assert not (output_dir / "split_manifest.json").exists()
    assert not (output_dir / "training_pool").exists()


def test_compare_main_uses_documented_default_sizes(monkeypatch, tmp_path):
    files = [f"episode_{index}.npz" for index in range(7000)]
    dataset = SimpleNamespace(files=files, data=[_rollout_record(index) for index in range(7000)])
    monkeypatch.setattr(compare_dataset_sizes, "SkillDataset", lambda _: dataset)
    monkeypatch.setattr(compare_dataset_sizes, "save_dataset_files", lambda *args: None)
    observed_validation_contexts = []
    observed_test_contexts = []

    def fake_training_loop(model, **kwargs):
        observed_validation_contexts.append(
            tuple(float(record[0][0]) for record in kwargs["val_loader"].dataset.data)
        )
        return model.state_dict(), [], [], 1

    def fake_evaluate(model, train_records, test_records, device=None):
        observed_test_contexts.append(tuple(float(record[0][0]) for record in test_records))
        return {
            "model_mse": {"payoff_mse": 1.0, "safety_mse": 2.0, "fuel_mse": 3.0},
            "training_mean_mse": {"payoff_mse": 2.0, "safety_mse": 3.0, "fuel_mse": 4.0},
            "improvement_mse": {"payoff_mse": 1.0, "safety_mse": 1.0, "fuel_mse": 1.0},
            "improvement_percent": {"payoff_mse": 50.0, "safety_mse": 33.333, "fuel_mse": 25.0},
            "model_beats_training_mean": {
                "payoff_mse": True,
                "safety_mse": True,
                "fuel_mse": True,
            },
        }

    monkeypatch.setattr(compare_dataset_sizes, "run_training_loop", fake_training_loop)
    monkeypatch.setattr(compare_dataset_sizes, "evaluate_on_test_set", fake_evaluate)
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
            "--sizes",
            "1000",
            "3000",
            "5250",
            "--report-json",
            str(tmp_path / "artifacts" / "comparison.json"),
            "--report-md",
            str(tmp_path / "artifacts" / "comparison.md"),
        ],
    )

    compare_dataset_sizes.main()

    summary = json.loads((tmp_path / "artifacts" / "comparison.json").read_text())
    assert [result["training_records"] for result in summary["results"]] == [1000, 3000, 5250]
    assert summary["split"] == {
        "seed": 42,
        "validation_records": 875,
        "validation_contexts": 875,
        "test_records": 875,
        "test_contexts": 875,
    }
    assert all(
        set(result) == {"training_records", "payoff", "safety", "fuel"}
        for result in summary["results"]
    )
    assert all("beats_baseline" not in result["payoff"] for result in summary["results"])
    assert summary["results"][0]["payoff"]["improvement_percent"] == 50.0
    assert len(set(observed_validation_contexts)) == 1
    assert len(set(observed_test_contexts)) == 1
    assert (tmp_path / "artifacts" / "comparison.md").exists()


def test_compare_main_warns_when_requested_sizes_exceed_available_data(
    monkeypatch, tmp_path, capsys
):
    files = [f"episode_{index}.npz" for index in range(20)]
    dataset = SimpleNamespace(files=files, data=[_rollout_record(index) for index in range(20)])
    monkeypatch.setattr(compare_dataset_sizes, "SkillDataset", lambda _: dataset)
    monkeypatch.setattr(
        compare_dataset_sizes,
        "run_training_loop",
        lambda *args, **kwargs: pytest.fail("training must not start without enough source records"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_dataset_sizes",
            "--data-dir",
            "unused",
            "--sizes",
            "100",
            "--report-json",
            str(tmp_path / "artifacts" / "comparison.json"),
            "--report-md",
            str(tmp_path / "artifacts" / "comparison.md"),
        ],
    )

    compare_dataset_sizes.main()

    output = capsys.readouterr().out
    assert "need at least 134 source records" in output
    assert "Collect at least 114 more records" in output
    assert "default seed is 42" in output
    assert "No models were trained" in output
    assert "python -m data_collector.collect --episodes 114" in output
    assert '--save-dir "unused"' in output
    assert "--seed" not in output


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


def test_dataset_size_evaluation_moves_batches_to_model_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeviceCheckingModel().to(device)
    train_records = [_rollout_record(0, 0.0, (0.0, 2.0)), _rollout_record(1, 2.0, (2.0, 4.0))]
    test_records = [_rollout_record(2, 4.0, (4.0, 6.0))]

    result = compare_dataset_sizes.evaluate_on_test_set(
        model,
        train_records,
        test_records,
        device=device,
    )

    assert result["training_mean_mse"]["payoff_mse"] == pytest.approx(9.0)


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
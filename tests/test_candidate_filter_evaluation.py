
"""Tests for offline candidate-filter metrics and split-safe evaluation."""

import numpy as np
import pytest

from generator import evaluate_candidate_filter as evaluator, ranking_config
from generator.policy_registry import POLICY_NAMES



def test_savings_missed_candidates_and_measured_admissions():
    rows = [
        {
            "policy": name,
            "payoff": 10.0 if i < 4 else -10.0,
            "motives": (
                np.array([2.0, -1.0])
                if i < 4
                else np.array([-2.0, -10.0])
            ),
        }
        for i, name in enumerate(POLICY_NAMES)
    ]

    groups = {"seed:1": rows}
    selected = {"seed:1": list(POLICY_NAMES[:2])}
    baseline = {
        "baseline_payoff": 0.0,
        "baseline_motives": [0.0, -5.0],
    }

    result = evaluator.assess(groups, selected, baseline)

    assert result["executed"] == 2
    assert result["avoided"] == 7
    assert result["execution_savings_fraction"] == pytest.approx(7 / 9)
    assert result["genuinely_certifiable"] == 4
    assert result["certifiable_found"] == 2
    assert result["certifiable_missed"] == 2
    assert result["missed_certifiable_fraction"] == pytest.approx(0.5)
    assert result["cds_admission_rate"] == 1.0
    assert result["pds_admission_rate"] == 1.0
    assert result["measured_payoff_mean"] == 10.0


def test_invalid_selections_are_rejected():
    rows = [
        {
            "policy": name,
            "payoff": 1.0,
            "motives": np.array([1.0, -1.0]),
        }
        for name in POLICY_NAMES
    ]

    baseline = {
        "baseline_payoff": 0.0,
        "baseline_motives": [0.0, -2.0],
    }

    with pytest.raises(ValueError, match="Invalid candidate selection"):
        evaluator.assess(
            {"seed:1": rows},
            {"seed:1": [POLICY_NAMES[0], POLICY_NAMES[0]]},
            baseline,
        )


def test_pds_only_admission_counts_as_certifiable_in_filter(monkeypatch):
    class RejectCDS:
        def admit(self, delta_r, delta_n):
            return False

    class AdmitPDS:
        def admit(self, delta_r, delta_n):
            return True

    monkeypatch.setattr(evaluator, "CDSGate", RejectCDS)
    monkeypatch.setattr(evaluator, "PDSGate", AdmitPDS)

    rows = [
        {
            "policy": name,
            "payoff": 1.0,
            "motives": np.array([1.0, -1.0]),
        }
        for name in POLICY_NAMES
    ]

    result = evaluator.assess(
        {"seed:1": rows},
        {"seed:1": [POLICY_NAMES[0]]},
        {
            "baseline_payoff": 0.0,
            "baseline_motives": [0.0, -2.0],
        },
    )

    assert result["genuinely_certifiable"] == len(POLICY_NAMES)
    assert result["certifiable_found"] == 1
    assert result["certifiable_missed"] == len(POLICY_NAMES) - 1
    assert result["cds_admission_rate"] == 0.0
    assert result["pds_admission_rate"] == 1.0


def test_evaluation_does_not_create_missing_manifest(
    tmp_path, monkeypatch
):
    rows = [{"context_id": "seed:1"}]
    manifest_path = tmp_path / "missing_manifest.json"
    model_load_called = False

    monkeypatch.setattr(evaluator, "load_records", lambda _: rows)

    def fail_if_manifest_missing(records, path):
        assert path == str(manifest_path)
        raise FileNotFoundError(
            "Saved training manifest is required; "
            "evaluation will not create a split"
        )

    def track_model_load(*args, **kwargs):
        nonlocal model_load_called
        model_load_called = True
        raise AssertionError("Model must not load without a saved manifest")

    monkeypatch.setattr(evaluator, "load_manifest", fail_if_manifest_missing)
    monkeypatch.setattr(evaluator, "load_model", track_model_load)

    with pytest.raises(FileNotFoundError, match="will not create"):
        evaluator.main([
            "--split-manifest", str(manifest_path),
            "--output-dir", str(tmp_path / "reports"),
        ])

    assert not manifest_path.exists()
    assert model_load_called is False


def test_evaluation_rejects_changed_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(
        evaluator,
        "load_records",
        lambda _: [{"context_id": "changed_context"}],
    )

    def reject_changed_data(records, path):
        raise ValueError("Dataset signature does not match saved manifest")

    monkeypatch.setattr(evaluator, "load_manifest", reject_changed_data)

    with pytest.raises(ValueError, match="Dataset signature"):
        evaluator.main([
            "--split-manifest", str(tmp_path / "manifest.json"),
        ])


def test_evaluation_rejects_checkpoint_with_wrong_split(
    tmp_path, monkeypatch
):
    manifest = {
        "dataset_signature": "dataset-A",
        "split_signature": "split-A",
    }

    monkeypatch.setattr(
        evaluator,
        "load_records",
        lambda _: [{"context_id": "seed:1"}],
    )
    monkeypatch.setattr(
        evaluator,
        "load_manifest",
        lambda records, path: manifest,
    )

    def reject_checkpoint(path, expected_manifest):
        assert expected_manifest == manifest
        raise ValueError("Checkpoint provenance does not match")

    monkeypatch.setattr(evaluator, "load_model", reject_checkpoint)

    with pytest.raises(ValueError) as exc_info:
        evaluator.main([
            "--split-manifest", str(tmp_path / "manifest.json"),
            "--model-path", str(tmp_path / "model.pt"),
        ])

    print("Actual exception:", repr(str(exc_info.value)))


def test_evaluation_uses_only_saved_test_split(tmp_path, monkeypatch):
    manifest = {
        "dataset_signature": "dataset-A",
        "split_signature": "split-A",
    }

    rows = [
        {"context_id": "train-context"},
        {"context_id": "test-context"},
    ]
    test_rows = [{"context_id": "test-context"}]

    monkeypatch.setattr(evaluator, "load_records", lambda _: rows)
    monkeypatch.setattr(
        evaluator,
        "load_manifest",
        lambda records, path: manifest,
    )

    def fake_partition(records, saved_manifest):
        assert saved_manifest == manifest
        return {
            "train": [{"context_id": "train-context"}],
            "val": [],
            "test": test_rows,
        }

    monkeypatch.setattr(evaluator, "partition", fake_partition)

    def fake_group_contexts(records, complete=True):
        assert records == test_rows
        assert complete is True
        return {"test-context": []}

    monkeypatch.setattr(evaluator, "group_contexts", fake_group_contexts)

    def fake_load_model(path, expected_manifest):
        assert expected_manifest == manifest
        return object(), True

    monkeypatch.setattr(evaluator, "load_model", fake_load_model)

    monkeypatch.setattr(
        "generator.evaluate_generator_report.compute_idle_baseline_stats",
        lambda episodes, seed: {"baseline_payoff": 0},
    )

    captured = {}

    
    def fake_evaluate_filter(groups, model, baseline_stats, k, seed, motive_weights=None):
       captured["groups"] = groups

       metrics = {
        "total_candidates": 1,
        "executed": 1,
        "avoided": 0,
        "execution_savings_fraction": 0.0,
        "genuinely_certifiable": 0,
        "certifiable_found": 0,
        "certifiable_missed": 0,
        "missed_certifiable_fraction": None,
        "cds_admission_rate": 0.0,
        "pds_admission_rate": 0.0,
        "cds_rejection_rate": 1.0,
        "pds_rejection_rate": 1.0,
        "measured_payoff_mean": 0.0,
        "measured_motives_mean": [0.0, 0.0],
        "measured_payoff_improvement_mean": 0.0,
        "measured_motive_improvement_mean": [0.0, 0.0],
       }

       return {
        "neural_top_k": metrics.copy(),
        "random_same_budget": metrics.copy(),
        "fixed_order_same_budget": metrics.copy(),
        "all_candidates_full_budget": metrics.copy(),
       }


    monkeypatch.setattr(
        evaluator,
        "evaluate_filter",
        fake_evaluate_filter,
    )
    ranking_config = tmp_path / "motive_weights.json"
    ranking_config.write_text(
      '{"motive_weights": {"safety": 0.5, "fuel": 0.5}, '
      '"split_signature": "split-A"}',
      encoding="utf-8",
   )
    evaluator.main([
        "--split-manifest", str(tmp_path / "manifest.json"),
        "--model-path", str(tmp_path / "model.pt"),
        "--output-dir", str(tmp_path / "reports"),
        "--ranking-config", str(ranking_config),
    ])

    assert list(captured["groups"]) == ["test-context"]
    assert "train-context" not in captured["groups"]
def test_offline_filter_uses_configured_motive_weights(monkeypatch):
    import torch

    class MotiveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, obs, ids):
            ids = ids.float()

            payoff = torch.zeros(
                len(ids), 1, device=ids.device
            ) + self.anchor

            motives = torch.stack(
                [ids, len(POLICY_NAMES) - 1 - ids],
                dim=1,
            )
            return payoff, motives

    rows = [
        {
            "policy": name,
            "obs": np.zeros(8),
            "payoff": 0.0,
            "motives": np.zeros(2),
        }
        for name in POLICY_NAMES
    ]

    groups = {"context-1": rows}

    baseline = {
        "baseline_payoff": 0.0,
        "baseline_motives": [0.0, 0.0],
    }

    captured = []

    def capture_assessment(groups, selections, baseline_stats):
        captured.append(dict(selections))
        return {}

    monkeypatch.setattr(
        evaluator,
        "assess",
        capture_assessment,
    )

    evaluator.evaluate_filter(
        groups,
        MotiveModel(),
        baseline,
        k=1,
        motive_weights=(1.0, 0.0),
    )

    safety_selection = captured[0]["context-1"]
    captured.clear()

    evaluator.evaluate_filter(
        groups,
        MotiveModel(),
        baseline,
        k=1,
        motive_weights=(0.0, 1.0),
    )

    fuel_selection = captured[0]["context-1"]

    assert safety_selection == [POLICY_NAMES[-1]]
    assert fuel_selection == [POLICY_NAMES[0]]
    assert safety_selection != fuel_selection
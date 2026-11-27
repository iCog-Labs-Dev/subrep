"""Multi-policy report coverage without running a real environment."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from generator.evaluate_generator_report import (
    build_multi_policy_report,
    run_multi_policy_report,
)
from generator.policy_registry import POLICY_NAMES
from generator.multi_policy_pipeline import save_model
from generator.multi_policy_data import load_records, load_or_create_manifest
from generator.skill_generator import SkillGenerator, PolicyConditionedSkillGenerator


def _rows(contexts):
    return [
        dict(
            context_id=f"seed:{seed}",
            seed=seed,
            policy=name,
            policy_id=i,
            obs=np.array(
                [seed / 100, 0, 0, 0, 0, 0, 0, 0],
                dtype=np.float32,
            ),
            payoff=float(i),
            motives=np.array([i / 10, -i], dtype=np.float32),
        )
        for seed in contexts
        for i, name in enumerate(POLICY_NAMES)
    ]


def test_report_uses_all_policies_and_measured_certification():
    conditioned = PolicyConditionedSkillGenerator()
    baseline = SkillGenerator()

    for model in (conditioned, baseline):
        for param in model.parameters():
            torch.nn.init.constant_(param, 0.0)

    report = build_multi_policy_report(
        _rows([10]),
        _rows([1, 2]),
        conditioned,
        baseline,
        {
            "baseline_payoff": 0.0,
            "baseline_motives": [0.0, -10.0],
        },
    )

    assert report["n_contexts"] == 1
    assert report["n_candidate_outcomes"] == len(POLICY_NAMES)
    assert set(report["measured_certification_per_policy"]) == set(POLICY_NAMES)
    assert set(report["prediction_metrics"]) == {
        "multi_policy",
        "observation_only",
        "per_policy_training_mean",
    }
    assert "fuel_mae" in report["prediction_metrics"]["multi_policy"]["overall"]
    assert (
        report["measured_certification_per_policy"]["ppo_deterministic"][
            "measured_payoff_mean"
        ]
        == 0.0
    )


def test_report_measured_certification_ignores_predictions():
    class WrongPredictor(torch.nn.Module):
        def forward(self, obs, policy_ids):
            return (
                torch.full((len(obs), 1), 100000.0),
                torch.full((len(obs), 2), 100000.0),
            )

    class WrongBaseline(torch.nn.Module):
        def forward(self, obs):
            return (
                torch.full((len(obs), 1), 100000.0),
                torch.full((len(obs), 2), 100000.0),
            )

    rows = _rows([11])
    for row in rows:
        row["payoff"] = -100.0
        row["motives"] = np.array([-100.0, -100.0])

    report = build_multi_policy_report(
        rows,
        _rows([1]),
        WrongPredictor(),
        WrongBaseline(),
        {
            "baseline_payoff": 0.0,
            "baseline_motives": [0.0, 0.0],
        },
    )

    assert all(
        value["certifiable_count"] == 0
        for value in report["measured_certification_per_policy"].values()
    )


def test_cli_report_reuses_manifest_and_only_test_contexts(tmp_path, monkeypatch):
    raw = tmp_path / "raw_mixed"
    raw.mkdir()

    for row in _rows(range(16)):
        np.savez(
            raw / f"{row['policy']}_{row['seed']}.npz",
            obs=row["obs"],
            payoff=row["payoff"],
            motives=row["motives"],
            context_seed=row["seed"],
            checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            gamma=0.8,
            max_steps=2,
            skill_id=row["policy"],
        )

    rows = load_records(str(raw))
    manifest = load_or_create_manifest(
        rows,
        str(tmp_path / "split.json"),
        seed=42,
    )

    model_path = tmp_path / "model.pt"
    baseline_path = tmp_path / "baseline.pt"

    save_model(
        PolicyConditionedSkillGenerator(),
        model_path,
        conditioned=True,
        manifest=manifest,
    )
    save_model(
        SkillGenerator(),
        baseline_path,
        conditioned=False,
        manifest=manifest,
    )

    stats = tmp_path / "idle.json"
    stats.write_text(
        json.dumps(
            {
                "baseline_payoff": 0.0,
                "baseline_motives": [0.0, -10.0],
                "gamma": 0.8,
                "max_steps": 2,
            }
        )
    )

    manifest_path = tmp_path / "split.json"
    out = tmp_path / "report.json"

    args = SimpleNamespace(
        data_dir=str(raw),
        split_manifest=str(manifest_path),
        seed=42,
        model_path=str(model_path),
        observation_baseline_path=str(baseline_path),
        baseline_stats=str(stats),
        baseline_episodes=1,
        baseline_seed=42,
        output=str(out),
    )
    def fake_baseline(episodes, seed, gamma, max_steps):
        assert gamma == 0.8
        assert max_steps == 2
        return {
        "baseline_payoff": 0.0,
        "baseline_motives": [0.0, -10.0],
        "gamma": gamma,
        "max_steps": max_steps,
        }
    # Avoid starting the real LunarLander environment in this unit test.
    monkeypatch.setattr(
        "generator.evaluate_generator_report.compute_idle_baseline_stats",
       fake_baseline,
    )
    monkeypatch.setattr(
        "generator.evaluate_generator_report.code_revision_metadata",
        lambda: {
            "code_commit": "test-commit",
            "working_tree_dirty": False,
        },
    )

    result = run_multi_policy_report(args)

    assert out.exists()
    saved_report = json.loads(out.read_text())
    assert saved_report["n_contexts"] == result["n_contexts"]
    assert result["run_metadata"]["dataset_signature"] == manifest["dataset_signature"]
    assert result["run_metadata"]["split_signature"] == manifest["split_signature"]
    assert result["run_metadata"]["baseline_settings"] == {
        "source": "measured_idle_execution",
        "episodes": 1,
        "seed": 42,
        "gamma": 0.8,
        "max_steps": 2,
    }
    assert result["run_metadata"]["model_constraints"][
        "noop_fuel_prediction"
    ]["source"] == "forced_by_policy_constraint_not_learned"
    assert saved_report["run_metadata"]["code_commit"] == "test-commit"
    assert result["n_contexts"] < 16
    assert result["n_contexts"] > 0

    assert (
        run_multi_policy_report(args)["n_contexts"]
        == result["n_contexts"]
    )


def test_cli_report_does_not_create_missing_manifest(tmp_path):
    raw = tmp_path / "raw_mixed"
    raw.mkdir()

    for row in _rows(range(16)):
        np.savez(
            raw / f"{row['policy']}_{row['seed']}.npz",
            obs=row["obs"],
            payoff=row["payoff"],
            motives=row["motives"],
            context_seed=row["seed"],
            checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            gamma=0.99,
            max_steps=-1,
            skill_id=row["policy"],
        )

    manifest_path = tmp_path / "missing.json"
    args = SimpleNamespace(
        data_dir=str(raw),
        split_manifest=str(manifest_path),
        seed=42,
        model_path=str(tmp_path / "not-used.pt"),
        observation_baseline_path=str(tmp_path / "not-used-baseline.pt"),
        baseline_stats=str(tmp_path / "idle.json"),
        baseline_episodes=1,
        baseline_seed=42,
        output=str(tmp_path / "report.json"),
    )

    with pytest.raises(
        FileNotFoundError,
        match="evaluation will not create a split",
    ):
        run_multi_policy_report(args)

    assert not manifest_path.exists()
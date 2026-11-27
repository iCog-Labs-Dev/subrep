import json
from types import SimpleNamespace

import numpy as np
import pytest

from data_collector import run_filtered_candidates as live

def test_live_baseline_uses_candidate_rollout_settings(monkeypatch):
    captured = {}

    class FakeIdlePolicy:
        def __init__(self, env, gamma):
            captured["gamma"] = gamma

        def run_baseline_episodes(self, num_episodes, seed, max_steps):
            captured["episodes"] = num_episodes
            captured["seed"] = seed
            captured["max_steps"] = max_steps
            return {
                "baseline_payoff": 1.0,
                "baseline_motives": np.array([0.5, -0.5]),
            }

    monkeypatch.setattr(live, "IdlePolicy", FakeIdlePolicy)
    monkeypatch.setattr(live, "SubRepEnv", lambda seed: object())

    live.measure_idle_baseline(
        episodes=20,
        seed=42,
        gamma=0.8,
        max_steps=100,
    )

    assert captured == {
        "episodes": 20,
        "seed": 42,
        "gamma": 0.8,
        "max_steps": 100,
    }


def test_live_filter_requires_an_explicit_weight_source():
    with pytest.raises(SystemExit) as exc_info:
        live.main([])

    assert exc_info.value.code == 2


def test_live_filter_rejects_ranking_config_from_another_split(
    tmp_path,
    monkeypatch,
):
    manifest = {
        "dataset_signature": "dataset-a",
        "split_signature": "split-a",
        "rollout_provenance": {
            "checkpoint_sha256": "a" * 64,
            "gamma": 0.99,
            "max_steps": -1,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config_path = tmp_path / "weights.json"
    config_path.write_text(
        json.dumps({
            "motive_weights": {"safety": 0.7, "fuel": 0.3},
            "split_signature": "split-b",
        }),
        encoding="utf-8",
    )
    captured = {}

    monkeypatch.setattr(
        live,
        "build_rollout_provenance",
        lambda *args: manifest["rollout_provenance"],
    )

    def fake_load_model(
        model_path,
        *,
        expected_manifest,
        expected_rollout_provenance,
    ):
        captured["expected_manifest"] = expected_manifest
        return object(), True

    monkeypatch.setattr(live, "load_model", fake_load_model)

    with pytest.raises(ValueError, match="different split"):
        live.main([
            "--ranking-config", str(config_path),
            "--split-manifest", str(manifest_path),
        ])

    assert captured["expected_manifest"] == manifest


def test_live_filter_accepts_explicit_equal_weights(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        live,
        "build_rollout_provenance",
        lambda *args: {"checkpoint_sha256": "a" * 64, "gamma": 0.99, "max_steps": -1},
    )

    def fake_load_model(
        model_path,
        *,
        expected_manifest,
        expected_rollout_provenance,
    ):
        captured["expected_manifest"] = expected_manifest
        return object(), True

    monkeypatch.setattr(live, "load_model", fake_load_model)
    monkeypatch.setattr(
        live,
        "measure_idle_baseline",
        lambda *args: SimpleNamespace(
            baseline_payoff=0.0,
            baseline_motives=np.zeros(2),
        ),
    )
    monkeypatch.setattr(live, "SubRepEnv", lambda seed: object())
    monkeypatch.setattr(
        live,
        "build_extended_candidate_policies",
        lambda env, pilot_checkpoint: [
            SimpleNamespace(skill_id=name) for name in live.POLICY_NAMES
        ],
    )

    def fake_evaluate_contexts(model, env, candidates, calculator, args):
        captured["motive_weights"] = args.motive_weights
        return []

    monkeypatch.setattr(live, "evaluate_contexts", fake_evaluate_contexts)
    monkeypatch.setattr(live, "save_report", lambda *args: None)

    live.main(["--equal-weights"])

    assert captured["motive_weights"] == (0.5, 0.5)
    assert captured["expected_manifest"] is None
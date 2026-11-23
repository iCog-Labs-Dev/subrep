
"""Validation-only motive-weight tuning tests."""

import numpy as np
import pytest
from generator import tune_motive_weights as tuner

def test_weight_grid_is_simplex_and_includes_endpoints():
    grid = tuner.weight_grid(0.25)

    assert grid == [
        (0.0, 1.0),
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
        (1.0, 0.0),
    ]

    assert all(
        safety >= 0 and fuel >= 0
        for safety, fuel in grid
    )
    assert all(
        np.isclose(safety + fuel, 1.0)
        for safety, fuel in grid
    )

def test_weight_grid_rejects_invalid_step():
    with pytest.raises(ValueError):
        tuner.weight_grid(0.0)

    with pytest.raises(ValueError):
        tuner.weight_grid(1.1)

def test_tuning_selects_best_validation_recall_and_keeps_top_k_fixed(
    monkeypatch,
):
    monkeypatch.setattr(
        tuner,
        "make_predicted_score_fn",
        lambda baseline, weights: weights,
    )

    seen = []

    def fake_select(model, obs, names, k, *, score_fn):
        seen.append((k, tuple(score_fn)))
        return [names[0]]

    monkeypatch.setattr(
        tuner,
        "select_top_k",
        fake_select,
    )

    def fake_assess(groups, selections, baseline):
        safety_weight = seen[-1][1][0]
        found = 8 if safety_weight == 0.25 else 5

        return {
            "genuinely_certifiable": 10,
            "certifiable_found": found,
            "executed": 4,
        }

    monkeypatch.setattr(
        tuner,
        "assess",
        fake_assess,
    )

    groups = {
        "context-1": [
            {"policy": f"p{i}", "obs": np.zeros(8)}
            for i in range(4)
        ]
    }

    result = tuner.tune_weights(
        groups,
        model=object(),
        baseline_stats={
            "baseline_payoff": 0.0,
            "baseline_motives": [0.0, 0.0],
        },
        fixed_top_k=2,
        grid_step=0.25,
    )

    assert result["motive_weights"] == {
        "safety": 0.25,
        "fuel": 0.75,
    }
    assert result["fixed_top_k"] == 2
    assert result["validation_metrics"]["recall"] == pytest.approx(0.8)
    assert all(k == 2 for k, _ in seen)

def test_tuning_refuses_validation_without_certifiable_candidates(
    monkeypatch,
):
    monkeypatch.setattr(
        tuner,
        "make_predicted_score_fn",
        lambda baseline, weights: weights,
    )

    monkeypatch.setattr(
        tuner,
        "select_top_k",
        lambda model, obs, names, k, *, score_fn: [names[0]],
    )

    monkeypatch.setattr(
        tuner,
        "assess",
        lambda *args: {
            "genuinely_certifiable": 0,
            "certifiable_found": 0,
            "executed": 1,
        },
    )

    groups = {
        "context-1": [
            {"policy": "p0", "obs": np.zeros(8)}
        ]
    }

    with pytest.raises(
        ValueError,
        match="no certifiable candidates",
    ):
        tuner.tune_weights(
            groups,
            model=object(),
            baseline_stats={
                "baseline_payoff": 0.0,
                "baseline_motives": [0.0, 0.0],
            },
            fixed_top_k=1,
        )

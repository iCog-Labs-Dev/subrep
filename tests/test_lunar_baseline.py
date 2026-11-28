"""Tests for the LunarLander baseline measurement (demo/run_lunar_baseline.py)."""

import pytest

from demo.run_lunar_baseline import (
    OUTCOMES,
    EpisodeResult,
    heuristic_policy,
    idle_policy,
    run_baseline,
    run_episode,
    summarise,
    to_markdown,
)
from env.lunar_lander_wrapper import SubRepEnv


def test_same_seed_gives_the_same_episode():
    env = SubRepEnv(seed=3)
    policy = heuristic_policy(env)
    first = run_episode(env, policy, seed=3)
    second = run_episode(env, policy, seed=3)
    env.close()
    assert first == second
    assert first.outcome in OUTCOMES


def test_idle_never_fires_an_engine():
    env = SubRepEnv(seed=0)
    result = run_episode(env, idle_policy(env), seed=0)
    env.close()
    assert result.fuel == 0.0
    assert result.frames > 0


def test_summary_rates_add_up_to_one():
    results = [
        EpisodeResult(seed=0, outcome="landed_on_pad", safety=200.0, fuel=-20.0, frames=300, final_x=0.0),
        EpisodeResult(seed=1, outcome="crashed", safety=-150.0, fuel=-5.0, frames=100, final_x=0.5),
        EpisodeResult(seed=2, outcome="landed_on_pad", safety=180.0, fuel=-25.0, frames=320, final_x=0.1),
        EpisodeResult(seed=3, outcome="timeout", safety=10.0, fuel=-60.0, frames=1000, final_x=0.3),
    ]
    summary = summarise(results)
    assert summary["episodes"] == 4
    assert sum(summary[f"{o}_rate"] for o in OUTCOMES) == pytest.approx(1.0)
    assert summary["landed_on_pad_rate"] == pytest.approx(0.5)
    assert summary["mean_fuel"] == pytest.approx(-27.5)


def test_summary_rejects_no_episodes():
    with pytest.raises(ValueError, match="zero episodes"):
        summarise([])


def test_report_and_markdown_cover_each_controller():
    report = run_baseline(seeds=[0, 1], controllers=["idle", "heuristic"])
    assert set(report) == {"idle", "heuristic"}
    assert len(report["heuristic"]["episodes"]) == 2
    table = to_markdown(report, seeds=[0, 1])
    assert "| idle |" in table and "| heuristic |" in table

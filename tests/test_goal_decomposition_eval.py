"""Tests for the goal decomposition evaluation (demo/run_goal_decomposition_eval.py)."""

import json

import numpy as np
import pytest

from demo.run_goal_decomposition_eval import PROFILES, evaluate, finish_episode, to_markdown
from env.lunar_facts import OUTCOMES, lander_facts
from env.lunar_lander_wrapper import SubRepEnv

SEEDS = range(2000, 2004)


@pytest.fixture(scope="module")
def report():
    profiles = {name: PROFILES[name] for name in ("safety_first", "no_region")}
    return evaluate(SEEDS, profiles=profiles, baselines=("idle",))


def test_every_controller_flies_every_seed(report):
    assert list(report["controllers"]) == [
        "idle",
        "fixed_careful_landing",
        "fixed_fuel_saving_landing",
        "runtime_safety_first",
        "runtime_no_region",
    ]
    for data in report["controllers"].values():
        summary = data["summary"]
        assert summary["episodes"] == len(SEEDS)
        assert sum(summary[f"{outcome}_rate"] for outcome in OUTCOMES) == pytest.approx(1.0)


def test_without_a_certified_plan_the_runtime_does_nothing(report):
    assert report["goal_runs"]["no_region"]["statuses"] == {"no_certified_plan": len(SEEDS)}
    assert report["controllers"]["runtime_no_region"]["summary"] == report["controllers"]["idle"]["summary"]
    runtime = [(e["seed"], e["outcome"], e["frames"]) for e in report["goal_runs"]["no_region"]["episodes"]]
    idle = [(e["seed"], e["outcome"], e["frames"]) for e in report["controllers"]["idle"]["episodes"]]
    assert runtime == idle


def test_the_runtime_flies_the_plan_it_certifies(report):
    run = report["goal_runs"]["safety_first"]
    assert sum(run["statuses"].values()) == len(SEEDS)
    assert run["first_plans"] == {"plan:land_on_pad/fuel_saving_landing": len(SEEDS)}
    assert run["predictions"]["held"] == run["predictions"]["checked"] > 0
    if run["episodes_with_replans"] == 0:
        runtime = report["controllers"]["runtime_safety_first"]["summary"]
        assert runtime == report["controllers"]["fixed_fuel_saving_landing"]["summary"]


def test_feedback_covers_every_certified_start(report):
    feedback = report["goal_runs"]["safety_first"]["feedback"]
    assert sum(entry["samples"] for entry in feedback) == len(SEEDS)
    assert {entry["status"] for entry in feedback} == {"too_few_samples"}


def test_the_table_is_checked_on_the_evaluation_seeds(report):
    rows = report["table_check"]
    assert rows
    assert all(row["samples"] >= 1 for row in rows)
    assert all(set(row["below_committed_lower"]) <= {"delta_r", "delta_safety", "delta_fuel"} for row in rows)


def test_recovery_is_tried_from_every_tilted_start(report):
    env = SubRepEnv(seed=0)
    tilted = sum(not lander_facts(env.reset(seed=seed)[0])["upright"] for seed in SEEDS)
    env.close()
    assert report["recover_stability"]["tilted_starts"] == tilted


def test_the_report_is_json_and_markdown_ready(report):
    assert json.loads(json.dumps(report))["seeds"] == {"start": SEEDS.start, "count": len(SEEDS)}
    markdown = to_markdown(report)
    for heading in ("## Controllers", "## Goal runtime", "## Certificates", "## Effect table", "## Recover stability"):
        assert heading in markdown
    assert all(f"| {name} |" in markdown for name in report["controllers"])


def test_an_unfinished_episode_is_finished_by_doing_nothing():
    env = SubRepEnv(seed=0)
    observation, _ = env.reset(seed=SEEDS.start)
    result = finish_episode(env, SEEDS.start, np.zeros((0, 2)), observation, None, episode_over=False)
    env.close()
    assert result.outcome == "crashed" and result.fuel == 0.0 and result.frames > 0


def test_every_episode_is_kept_and_traceable_to_its_seed(report):
    for data in report["controllers"].values():
        if "episodes" in data:
            assert [episode["seed"] for episode in data["episodes"]] == list(SEEDS)
    runs = report["goal_runs"]["safety_first"]["episodes"]
    assert [run["seed"] for run in runs] == list(SEEDS)
    achieved = [run for run in runs if run["status"] == "achieved"]
    assert all(run["outcome"] == "landed_on_pad" for run in achieved)
    assert all(run["plan"] is None for run in report["goal_runs"]["no_region"]["episodes"])


@pytest.mark.parametrize("seeds, message", [(range(990, 1010), "overlap"), (range(5, 5), "at least one seed")])
def test_the_evaluation_refuses_seeds_it_cannot_use(seeds, message):
    with pytest.raises(ValueError, match=message):
        evaluate(seeds, profiles={}, baselines=())

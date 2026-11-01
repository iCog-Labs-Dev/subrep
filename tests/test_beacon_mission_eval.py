"""Tests for the scan-then-land evaluation (demo/run_beacon_mission_eval.py)."""

import json

import numpy as np
import pytest

from decomposition.effects import LUNAR_BEACON_TABLE, EffectTable
from decomposition.runtime import GoalStatus, pursue
from demo.build_lunar_effect_table import mission_book
from demo.run_beacon_mission_eval import (
    GOALS,
    PROFILES,
    MissionEpisode,
    check_criteria,
    compare_limit,
    evaluate_runtime,
    fly_controller,
    fly_fixed,
    goal_split,
    rates,
    wilson_interval,
)
from env.lunar_beacon import BEACON_MISSION, MISSION_COMPLETE

SEEDS = list(range(3200, 3204))
OUTCOMES = BEACON_MISSION.outcomes


@pytest.fixture(scope="module")
def env():
    environment = BEACON_MISSION.make_env(0)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def book():
    return mission_book()


@pytest.fixture(scope="module")
def table():
    return EffectTable.load(LUNAR_BEACON_TABLE, BEACON_MISSION)


@pytest.fixture(scope="module")
def mission_run(env, book, table):
    return evaluate_runtime(BEACON_MISSION, env, book, table, PROFILES["mission_first"], SEEDS)


def test_the_committed_table_measures_the_mission(table):
    assert tuple(table.data["objectives"]) == ("safety", "fuel", "mission")
    assert any("go_to_beacon" in entry["skills"] for entry in table.data["plans"])
    assert all("delta_mission" in entry for entry in table.data["plans"])


def test_mission_first_priorities_go_for_the_mission(mission_run):
    assert mission_run["first_goals"].get("scan_then_land", 0) == len(SEEDS)
    assert sum(mission_run["statuses"].values()) == len(SEEDS)


def test_fuel_first_priorities_do_not_scan(env, book, table):
    observation, _ = env.reset(seed=SEEDS[0])
    run = pursue(BEACON_MISSION, env, observation, GOALS, book, table, PROFILES["fuel_first"])
    if run.decisions and run.decisions[0].chosen is not None:
        assert "go_to_beacon" not in run.decisions[0].chosen.candidate.skills


def test_the_goal_achieved_is_the_goal_of_the_plan_flown(env, book, table):
    observation, _ = env.reset(seed=SEEDS[0])
    run = pursue(BEACON_MISSION, env, observation, GOALS, book, table, PROFILES["mission_first"])
    assert run.goal == run.flown[-1].chosen.candidate.goal
    if run.status is GoalStatus.ACHIEVED:
        assert BEACON_MISSION.outcome(run.info, run.observation) == MISSION_COMPLETE


def test_every_episode_is_kept_and_traceable(mission_run):
    assert [episode["seed"] for episode in mission_run["episodes"]] == SEEDS
    json.dumps(mission_run)


def test_controllers_fly_a_whole_episode(env):
    episode = fly_controller(BEACON_MISSION, env, lambda obs: BEACON_MISSION.idle_action, SEEDS[0])
    assert episode.outcome == "crashed" and episode.fuel == 0.0 and episode.mission == 0.0 and episode.frames > 0


def test_the_fixed_mission_flies_every_skill(env, book):
    episode = fly_fixed(BEACON_MISSION, env, book, SEEDS[0])
    assert episode.outcome in OUTCOMES
    assert episode.frames > 0


def episode(seed, outcome):
    return MissionEpisode(seed, outcome, 1.0, -1.0, 0.0, 10)


def test_rates_cover_every_outcome():
    summary = rates([episode(0, MISSION_COMPLETE), episode(1, "crashed")], OUTCOMES)
    assert summary["episodes"] == 2
    assert summary[f"{MISSION_COMPLETE}_rate"] == 0.5
    assert sum(summary[f"{outcome}_rate"] for outcome in OUTCOMES) == pytest.approx(1.0)


def test_a_confidence_interval_holds_its_rate():
    low, high = wilson_interval(90, 100)
    assert low < 0.9 < high
    assert wilson_interval(100, 100)[1] == pytest.approx(1.0)


def test_comparing_a_limit_counts_what_changed():
    full = {"episodes": [{"first_plan": "a", "outcome": "x"}, {"first_plan": "a", "outcome": "x"}], "summary": {}}
    limited = {
        "episodes": [{"first_plan": "a", "outcome": "x"}, {"first_plan": "b", "outcome": "y"}],
        "summary": {f"{MISSION_COMPLETE}_rate": 0.5},
    }
    assert compare_limit(full, limited)["same_first_plan_rate"] == 0.5


def test_a_goal_split_counts_episodes_with_different_goals():
    assert goal_split({"episodes": [{"goal": "a"}, {"goal": "b"}]}, {"episodes": [{"goal": "a"}, {"goal": "a"}]}) == 0.5


def fake_report(completion, crashed=0.0):
    run = {
        "summary": {f"{MISSION_COMPLETE}_rate": completion, "crashed_rate": crashed},
        "recertified_rechecks": 9,
        "rechecks": 10,
        "feedback": [],
    }
    return {
        "goal_runs": {"mission_first": run},
        "controllers": {"mission_ppo": {"summary": {f"{MISSION_COMPLETE}_rate": 0.5}}},
        "moved_beacons": [{"runtime": {"summary": {f"{MISSION_COMPLETE}_rate": 0.9}}}],
        "goal_split": 1.0,
        "limit": {"same_first_plan_rate": 1.0},
    }


def test_the_criteria_report_what_was_met():
    assert all(row["met"] for row in check_criteria(fake_report(0.95)))
    missed = {row["criterion"] for row in check_criteria(fake_report(0.5, crashed=0.2)) if not row["met"]}
    assert "Mission completion under mission-first priorities" in missed
    assert "Crashes under mission-first priorities" in missed


def test_profiles_are_on_the_simplex():
    for priorities in PROFILES.values():
        assert sum(priorities.weights) == pytest.approx(1.0)
        assert len(priorities.weights) == 3
    assert np.argmax(PROFILES["mission_first"].weights) == 2
    assert np.argmax(PROFILES["fuel_first"].weights) == 1

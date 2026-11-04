"""Tests for the goal decomposition evaluation (demo/run_goal_decomposition_eval.py)."""

import json

import numpy as np
import pytest

from demo.run_goal_decomposition_eval import (
    INSUFFICIENT_EVIDENCE,
    MET,
    MISSED,
    PROFILES,
    REPORT_ONLY,
    evaluate,
    finish_episode,
    revocation_criterion,
    share_criterion,
    to_markdown,
)
from env.lunar_facts import OUTCOMES, lander_facts
from env.lunar_lander_wrapper import SubRepEnv

SEEDS = range(8000, 8004)


@pytest.fixture(scope="module")
def report():
    profiles = {name: PROFILES[name] for name in ("safety_first", "no_region")}
    return evaluate(SEEDS, profiles=profiles, baselines=("idle",), use_mdn=False)


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
    runs = report["goal_runs"]["no_region"]["episodes"]
    uncertified = [run for run in runs if run["status"] == "no_certified_plan"]
    assert uncertified
    idle = {e["seed"]: (e["outcome"], e["frames"]) for e in report["controllers"]["idle"]["episodes"]}
    assert all((run["outcome"], run["frames"]) == idle[run["seed"]] for run in uncertified)
    assert all(run["plan"] is None for run in uncertified)


def test_the_runtime_flies_the_plan_it_certifies(report):
    run = report["goal_runs"]["safety_first"]
    assert sum(run["statuses"].values()) == len(SEEDS)
    assert sum(run["first_plans"].values()) == len(SEEDS)
    assert all(name.startswith("plan:land_on_pad/") for name in run["first_plans"])
    assert run["predictions"]["held"] == run["predictions"]["checked"] > 0


def test_feedback_covers_certified_plans_flown_to_the_end(report):
    feedback = report["goal_runs"]["safety_first"]["feedback"]
    assert sum(entry["samples"] for entry in feedback) >= len(SEEDS)
    assert {entry["status"] for entry in feedback} == {"too_few_samples"}


def test_the_runtime_rechecks_in_mid_flight(report):
    run = report["goal_runs"]["safety_first"]
    assert run["rechecks"] > 0
    assert run["recertified_rechecks"] > 0


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


def test_the_report_says_which_criteria_were_met(report):
    assert {row["criterion"] for row in report["criteria"]} >= {"On the pad under safety-first priorities"}
    assert all(row["result"] in (MET, MISSED, INSUFFICIENT_EVIDENCE, REPORT_ONLY) for row in report["criteria"])
    assert "## Criteria written before the evaluation" in to_markdown(report)


def test_the_continuation_policy_is_compared_apart_from_the_criteria(report):
    assert all(run["continued"] == 0 for run in report["goal_runs"].values())
    continued = report["continuation"]["safety_first"]
    assert continued["summary"]["episodes"] == len(SEEDS)
    assert "safety_first_continuing" not in report["goal_runs"]
    markdown = to_markdown(report)
    assert "## Continuation policy" in markdown
    assert "| safety_first | current approval |" in markdown and "| safety_first | continuation |" in markdown


def test_revocations_with_too_few_samples_are_insufficient_evidence_not_a_pass(report):
    [row] = [row for row in report["criteria"] if row["criterion"].startswith("Certificates revoked")]
    assert row["result"] == INSUFFICIENT_EVIDENCE and row["value"] is None
    assert row["evidence"].startswith("0 of ")
    markdown = to_markdown(report)
    assert "| insufficient evidence |" in markdown
    assert "the runtime did not apply these revocations during the runs" in markdown


@pytest.mark.parametrize(
    "statuses, result",
    [
        ([], INSUFFICIENT_EVIDENCE),
        (["too_few_samples", "too_few_samples"], INSUFFICIENT_EVIDENCE),
        (["too_few_samples", "held"], MET),
        (["held", "revoked"], MISSED),
    ],
)
def test_revocations_are_judged_only_on_groups_with_enough_flights(statuses, result):
    runs = [{"feedback": [{"status": status} for status in statuses]}]
    assert revocation_criterion(runs)["result"] == result


def test_a_share_of_nothing_is_insufficient_evidence():
    assert share_criterion("re-checks", 0, 0, 0.9, "re-checks")["result"] == INSUFFICIENT_EVIDENCE
    assert share_criterion("re-checks", 9, 10, 0.9, "re-checks")["result"] == MET
    assert share_criterion("re-checks", 8, 10, 0.9, "re-checks")["result"] == MISSED


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


@pytest.mark.parametrize("seeds, message", [(range(990, 1010), "overlap"), (range(5, 5), "at least one seed")])
def test_the_evaluation_refuses_seeds_it_cannot_use(seeds, message):
    with pytest.raises(ValueError, match=message):
        evaluate(seeds, profiles={}, baselines=(), use_mdn=False)

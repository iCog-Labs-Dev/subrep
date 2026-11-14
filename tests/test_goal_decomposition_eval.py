"""Tests for the goal decomposition evaluation (demo/run_goal_decomposition_eval.py)."""

import dataclasses
import json

import numpy as np
import pytest

from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable
from decomposition.executor import Execution, StepRecord, StepStatus
from decomposition.feedback import REVOKED_REASON, FeedbackLog
from decomposition.planner import CandidatePlan
from decomposition.recipe_book import LUNAR_LANDER_BOOK, load_recipe_book
from decomposition.runtime import Continuation, Decision, GoalRun, GoalStatus, pursue
from decomposition.verifier import GOAL_TOO_RARE, PlanVerdict, plan_skill_id
from demo.run_goal_decomposition_eval import (
    GOAL,
    INSUFFICIENT_EVIDENCE,
    MET,
    MISSED,
    NOTHING_PROPOSED,
    PROFILES,
    REPORT_ONLY,
    Safeguards,
    abstention_reasons,
    evaluate,
    evaluate_profile,
    finish_episode,
    plan_flown_to_its_end,
    record_followed_plans,
    revocation_criterion,
    share_criterion,
    shortfalls,
    to_markdown,
)
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import FACT_NAMES, OUTCOMES, lander_facts
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
        "runtime_safety_first_stop",
        "runtime_safety_first_continuation",
    ]
    for data in report["controllers"].values():
        summary = data["summary"]
        assert summary["episodes"] == len(SEEDS)
        assert sum(summary[f"{outcome}_rate"] for outcome in OUTCOMES) == pytest.approx(1.0)


def test_without_a_certified_plan_the_runtime_does_nothing(report):
    env = SubRepEnv(seed=0)
    book = load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, LUNAR_LANDER.skills)
    table = EffectTable.load(DEFAULT_TABLE_PATH, LUNAR_LANDER)
    strict = evaluate_profile(env, book, table, PROFILES["no_region"], list(SEEDS), Continuation.STOP)
    env.close()
    never_flown = [run for run in strict["episodes"] if run["plan"] is None]
    assert never_flown
    assert all(run["status"] == "no_certified_plan" for run in never_flown)
    idle = {e["seed"]: (e["outcome"], e["frames"]) for e in report["controllers"]["idle"]["episodes"]}
    assert all((run["outcome"], run["frames"]) == idle[run["seed"]] for run in never_flown)


def test_the_runtime_flies_the_plan_it_certifies(report):
    run = report["goal_runs"]["safety_first"]
    assert sum(run["statuses"].values()) == len(SEEDS)
    assert sum(run["first_plans"].values()) == len(SEEDS)
    assert all(name.startswith("plan:land_on_pad/") for name in run["first_plans"])
    assert run["predictions"]["held"] == run["predictions"]["checked"] > 0


def test_feedback_covers_certified_plans_flown_to_the_end(report):
    run = report["goal_runs"]["safety_first"]
    assert sum(entry["samples"] for entry in run["feedback"]) >= len(SEEDS) - run["feedback_unchecked_runs"]
    assert {entry["status"] for entry in run["feedback"]} == {"too_few_samples"}
    assert "Runs stopped in the middle of a plan, so compared with no certificate: safety_first " in to_markdown(report)


def one_skill_run(skills_left, status, completed=True, episode_over=False):
    """A run whose only decision flew the first of ``skills_left``."""
    step = StepRecord(skills_left[0], StepStatus.COMPLETED if completed else StepStatus.OUT_OF_FRAMES, 5, {}, {}, None)
    execution = Execution(skills_left[:1], (step,), (0,) * 5, np.zeros((5, 2)), 0, np.zeros(8), episode_over, None)
    decision = Decision({}, (), None, tuple(skills_left), execution, recertified=True)
    return GoalRun("land_on_pad", status, (decision,), np.zeros((5, 2)), np.zeros(8), episode_over, None)


@pytest.mark.parametrize(
    "skills_left, status, completed, episode_over, flown_to_its_end",
    [
        (("touch_down",), GoalStatus.ACHIEVED, True, True, True),
        (("touch_down",), GoalStatus.FAILED, False, True, True),  # a failed flight is charged until the episode ends
        (("brake",), GoalStatus.NO_CERTIFIED_PLAN, True, False, True),  # the whole plan was flown
        (("slow_descent", "touch_down"), GoalStatus.NO_CERTIFIED_PLAN, True, False, False),
        (("slow_descent", "touch_down"), GoalStatus.PRIORITIES_REFUSED, True, False, False),
        (("touch_down",), GoalStatus.FAILED, False, False, False),  # given up on before the episode ended
    ],
)
def test_only_plans_flown_as_they_were_measured_are_compared(skills_left, status, completed, episode_over, flown_to_its_end):
    assert plan_flown_to_its_end(one_skill_run(skills_left, status, completed, episode_over)) is flown_to_its_end


def test_a_run_stopped_in_the_middle_of_its_plan_is_not_recorded():
    feedback = FeedbackLog(LUNAR_LANDER)
    run = one_skill_run(("slow_descent", "touch_down"), GoalStatus.NO_CERTIFIED_PLAN)
    assert record_followed_plans(LUNAR_LANDER, feedback, None, SEEDS.start, run) is False
    assert feedback.report() == []
    assert not plan_flown_to_its_end(dataclasses.replace(run, decisions=()))


def test_fallbacks_are_compared_with_their_certificates():
    env = SubRepEnv(seed=0)
    book = load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, LUNAR_LANDER.skills)
    table = EffectTable.load(DEFAULT_TABLE_PATH, LUNAR_LANDER)
    observation, _ = env.reset(seed=SEEDS.start)
    every_plan_too_rare = 1.0  # no plan passes the goal-rate check, so every skill flies as a fallback
    safety_first = PROFILES["safety_first"]
    run = pursue(LUNAR_LANDER, env, observation, GOAL, book, table, safety_first, min_goal_rate=every_plan_too_rare)
    feedback = FeedbackLog(LUNAR_LANDER)
    recorded = record_followed_plans(LUNAR_LANDER, feedback, env, SEEDS.start, run)
    env.close()
    assert run.flown and all(decision.fallback for decision in run.flown)
    assert recorded and feedback.report()
    assert {entry["plan"] for entry in feedback.report()} <= {plan_skill_id(d.chosen.candidate) for d in run.flown}


def test_the_runtime_rechecks_in_mid_flight(report):
    run = report["goal_runs"]["safety_first"]
    assert run["rechecks"] > 0
    assert run["recertified_rechecks"] > 0


def test_the_table_is_checked_on_the_evaluation_seeds(report):
    rows = report["table_check"]
    assert rows
    assert all(1 <= row["episodes"] <= row["samples"] for row in rows)
    assert all(set(row["below_committed_lower"]) <= {"delta_r", "delta_safety", "delta_fuel"} for row in rows)
    below = sum(bool(row["below_committed_lower"]) for row in rows)
    assert f"{below} of {len(rows)} entries had a mean below the committed lower bound" in to_markdown(report)


def test_recovery_is_tried_from_every_tilted_start(report):
    env = SubRepEnv(seed=0)
    tilted = sum(not lander_facts(env.reset(seed=seed)[0])["upright"] for seed in SEEDS)
    env.close()
    assert report["recover_stability"]["tilted_starts"] == tilted


def test_the_report_says_which_criteria_were_met(report):
    assert {row["criterion"] for row in report["criteria"]} >= {"On the pad under safety-first priorities"}
    assert all(row["result"] in (MET, MISSED, INSUFFICIENT_EVIDENCE, REPORT_ONLY) for row in report["criteria"])
    assert "## Criteria written before the evaluation" in to_markdown(report)


def test_the_other_policies_are_compared_apart_from_the_criteria(report):
    assert all(run["continued"] == 0 for run in report["goal_runs"].values())
    others = report["policies"]["safety_first"]
    assert set(others) == {"stop", "continuation"}
    assert others["stop"]["fallbacks"] == others["stop"]["continued"] == 0
    assert others["continuation"]["fallbacks"] == 0
    assert all(run["summary"]["episodes"] == len(SEEDS) for run in others.values())
    markdown = to_markdown(report)
    assert "## When no plan is admitted" in markdown
    assert all(f"| safety_first | {policy} |" in markdown for policy in ("fallback", "stop", "continuation"))


def test_reliability_and_live_revocation_are_reported(report):
    live = report["live_revocation"]["safety_first"]
    assert live["summary"]["episodes"] == len(SEEDS) and live["revoked_refusals"] >= 0
    markdown = to_markdown(report)
    assert "## Certificate reliability" in markdown and "| safety_first | 0 | 0 |" in markdown
    assert "## Revocations applied during the runs" in markdown


def with_last_decision(run, **changes):
    *earlier, last = run.decisions
    return dataclasses.replace(run, decisions=(*earlier, dataclasses.replace(last, **changes)))


def test_safeguards_count_fallbacks_abstentions_and_refused_revocations():
    candidate = CandidatePlan("land_on_pad", "m", ("m",), ("touch_down",), {})
    stopped = one_skill_run(("slow_descent", "touch_down"), GoalStatus.NO_CERTIFIED_PLAN)
    too_rare = with_last_decision(stopped, verdicts=(PlanVerdict(candidate, reason=GOAL_TOO_RARE),))
    refused = dataclasses.replace(stopped, status=GoalStatus.PRIORITIES_REFUSED)
    revoked = PlanVerdict(candidate, reason=REVOKED_REASON)
    fell_back = with_last_decision(
        dataclasses.replace(stopped, status=GoalStatus.ACHIEVED), fallback=True, verdicts=(revoked,)
    )
    safeguards = Safeguards()
    for seed, (run, outcome) in enumerate(((too_rare, "crashed"), (refused, "crashed"), (fell_back, "landed_on_pad"))):
        safeguards.add(run, outcome, seed)
    record = safeguards.record()
    audited = record.pop("audited")
    assert record == {
        "fallbacks": 1,
        "runs_with_fallback": 1,
        "continued": 0,
        "abstained": 2,
        "revoked_refusals": 1,
        "fallback_outcomes": {"landed_on_pad": 1},
        "abstention_reasons": {GOAL_TOO_RARE: 1, GoalStatus.PRIORITIES_REFUSED.value: 1},
    }
    assert [(entry["seed"], entry["outcome"], entry["fallbacks"]) for entry in audited] == [
        (0, "crashed", 0),
        (1, "crashed", 0),
        (2, "landed_on_pad", 1),
    ]
    assert json.loads(json.dumps(audited))[0]["decisions"][0]["rejected"] == [
        {"plan": plan_skill_id(candidate), "reason": GOAL_TOO_RARE}
    ]


def test_a_run_that_abstained_with_nothing_proposed_says_so():
    stopped = one_skill_run(("slow_descent", "touch_down"), GoalStatus.NO_CERTIFIED_PLAN)
    assert abstention_reasons(stopped) == [NOTHING_PROPOSED]


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


def test_a_revoked_certificate_names_the_gains_that_fell_short(report):
    entry = {
        "status": "revoked",
        "delta_r": {"certified": 55.7, "realised_mean": 64.4, "realised_upper": 70.3},
        "delta_fuel": {"certified": -56.0, "realised_mean": -60.0, "realised_upper": -57.1},
    }
    assert shortfalls(entry) == "delta_fuel"
    assert shortfalls({"delta_r": {"certified": 1.0, "realised_mean": 0.0, "realised_upper": None}}) == "-"
    assert "| Fell short on |" in to_markdown(report)


def test_a_share_of_nothing_is_insufficient_evidence():
    assert share_criterion("re-checks", 0, 0, 0.9, "re-checks")["result"] == INSUFFICIENT_EVIDENCE
    assert share_criterion("re-checks", 9, 10, 0.9, "re-checks")["result"] == MET
    assert share_criterion("re-checks", 8, 10, 0.9, "re-checks")["result"] == MISSED


def test_the_report_is_json_and_markdown_ready(report):
    assert json.loads(json.dumps(report))["seeds"] == {"start": SEEDS.start, "count": len(SEEDS)}
    markdown = to_markdown(report)
    for heading in ("## Controllers", "## Goal runtime", "## Certificates", "## Effect table", "## Recover stability"):
        assert heading in markdown
    settings = report["settings"]
    assert settings["default_policy"] == "fallback" and settings["min_goal_rate"] == 0.9
    assert settings["table"]["seeds"] == {"start": 0, "stop": 2000}
    assert "Settings: `fallback` policy, goal rate at least 0.9, switch margin 0.1" in markdown
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

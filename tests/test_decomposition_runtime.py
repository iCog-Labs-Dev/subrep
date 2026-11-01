"""Tests for pursuing goals with planning, certification before every skill and replanning (decomposition/runtime.py)."""

import dataclasses
import json

import numpy as np
import pytest

from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable, build_table
from decomposition.planner import CandidatePlan, plans_for
from decomposition.recipe_book import LUNAR_LANDER_BOOK, load_recipe_book
from decomposition.runtime import GoalStatus, _choose, _may_go_on, goal_achieved, pursue
from decomposition.verifier import NOT_MEASURED, MdnRegion, PlanVerdict, Priorities
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import FACT_NAMES, landing_outcome, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import NO_ENGINE, SUBSKILLS

TILTED_START_SEED = 0
UPRIGHT_START_SEED = 1  # starts away from the pad; lands with one leg's contact flickering off
SAFETY_FIRST = Priorities(
    weights=(0.7, 0.3),
    region=MdnRegion(
        context=(0.0,),
        alpha=(4.0, 1.0),
        support_directions=((1.0, 0.0), (0.0, 1.0)),
        support_values=(1.0, 0.3),
    ),
)


@pytest.fixture
def env():
    environment = SubRepEnv(seed=0)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def book():
    return load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, SUBSKILLS)


@pytest.fixture(scope="module")
def table():
    return EffectTable.load(DEFAULT_TABLE_PATH, LUNAR_LANDER)


@pytest.fixture(scope="module")
def starts_only(book):
    """A table measured only at the start of episodes, so that nothing in mid-flight is measured."""
    return build_table(LUNAR_LANDER, range(60), plans_for(book), min_samples=5, max_states=1)


def run_goal(env, book, table, seed, goal="land_on_pad", priorities=SAFETY_FIRST, **kwargs):
    observation, _ = env.reset(seed=seed)
    return pursue(LUNAR_LANDER, env, observation, goal, book, table, priorities, **kwargs)


def test_a_certified_landing_plan_is_flown_to_a_landing(env, book, table):
    run = run_goal(env, book, table, TILTED_START_SEED)
    assert run.status is GoalStatus.ACHIEVED
    first = run.decisions[0]
    assert first.chosen.entry.certificate.skill_id.startswith("plan:land_on_pad/")
    assert all(decision.execution.completed for decision in run.decisions)
    assert run.replans == 0 and run.episode_over
    assert run.frames == sum(decision.execution.frames for decision in run.decisions)
    assert run.info["landing_success"]


def test_only_one_skill_is_flown_at_each_decision(env, book, table):
    run = run_goal(env, book, table, TILTED_START_SEED)
    assert len(run.decisions) > 1
    assert all(len(decision.execution.steps) == 1 for decision in run.decisions)
    assert not any(decision.execution.settling_frames for decision in run.decisions[:-1])


def test_the_rest_of_the_plan_is_recertified_in_mid_flight(env, book, table):
    run = run_goal(env, book, table, TILTED_START_SEED)
    assert run.rechecks == len(run.decisions) - 1 > 0
    assert run.recertified_rechecks > 0


def test_the_environment_decides_once_the_episode_is_over(env, book, table):
    runs = (run_goal(env, book, table, seed) for seed in range(100))
    run = next(r for r in runs if r.info["landing_success"] and not lander_facts(r.observation)["landed"])
    assert abs(run.observation[0]) <= 0.2 and run.status is GoalStatus.ACHIEVED


def test_a_goal_that_already_holds_needs_no_plan(env, book, table):
    run = run_goal(env, book, table, UPRIGHT_START_SEED, goal="recover_stability")
    assert run.status is GoalStatus.ALREADY_ACHIEVED
    assert run.decisions == () and run.frames == 0


def test_recovering_stability_leaves_the_lander_upright_in_flight(env, book, table):
    run = run_goal(env, book, table, TILTED_START_SEED, goal="recover_stability")
    assert run.status is GoalStatus.ACHIEVED
    assert lander_facts(run.observation)["upright"]
    assert not run.episode_over and run.frames > 0


def test_without_a_certified_plan_nothing_is_flown(env, book, table):
    observation, _ = env.reset(seed=TILTED_START_SEED)
    nothing_measured = EffectTable({**table.data, "plans": []}, LUNAR_LANDER)
    run = pursue(LUNAR_LANDER, env, observation, "land_on_pad", book, nothing_measured, SAFETY_FIRST)
    assert run.status is GoalStatus.NO_CERTIFIED_PLAN
    [decision] = run.decisions
    assert decision.execution is None and decision.chosen is None
    assert {verdict.reason for verdict in decision.verdicts} == {NOT_MEASURED}
    assert run.frames == 0
    np.testing.assert_array_equal(run.observation, observation)


def test_a_plan_goes_on_under_its_certificate_when_the_rest_was_not_measured(env, book, starts_only):
    runs = (run_goal(env, book, starts_only, seed) for seed in range(2000, 2100))
    run = next(r for r in runs if len(r.flown) > 1 and not r.flown[1].recertified)
    first, second = run.flown[:2]
    assert first.recertified
    assert second.chosen is first.chosen
    assert second.skills == first.skills[1:]


def test_after_a_failed_step_a_replan_needs_measurements_that_apply(env, book, starts_only, monkeypatch):
    monkeypatch.setitem(SUBSKILLS, "slow_descent", dataclasses.replace(SUBSKILLS["slow_descent"], max_frames=20))
    monkeypatch.setitem(SUBSKILLS, "move_over_pad", dataclasses.replace(SUBSKILLS["move_over_pad"], max_frames=20))
    runs = (run_goal(env, book, starts_only, seed) for seed in range(2000, 2100))
    run = next(r for r in runs if r.flown and r.flown[0].execution.stopped_at is not None)
    assert run.status is GoalStatus.NO_CERTIFIED_PLAN
    assert run.decisions[-1].execution is None
    assert {verdict.reason for verdict in run.decisions[-1].verdicts} == {NOT_MEASURED}


def test_a_failed_step_is_resumed_when_measurements_apply(env, book, table, monkeypatch):
    monkeypatch.setitem(SUBSKILLS, "slow_descent", dataclasses.replace(SUBSKILLS["slow_descent"], max_frames=40))
    monkeypatch.setitem(SUBSKILLS, "move_over_pad", dataclasses.replace(SUBSKILLS["move_over_pad"], max_frames=40))
    wide = EffectTable({**table.data, "support_margin": 1000.0}, LUNAR_LANDER)
    runs = (run_goal(env, book, wide, seed, max_attempts=15) for seed in range(2000, 2100))
    run = next(r for r in runs if r.replans >= 1)
    assert any(decision.execution.stopped_at is not None for decision in run.flown)
    assert len(run.flown) > 1


def test_failures_run_out(env, book, table, monkeypatch):
    monkeypatch.setitem(SUBSKILLS, "stabilise", dataclasses.replace(SUBSKILLS["stabilise"], max_frames=1))
    wide = EffectTable({**table.data, "support_margin": 1000.0}, LUNAR_LANDER)
    runs = (run_goal(env, book, wide, seed, goal="recover_stability", max_attempts=2) for seed in range(2000, 2100))
    run = next(r for r in runs if r.status is GoalStatus.FAILED)
    assert len(run.flown) == 2 and run.replans == 1
    assert not run.episode_over


def test_a_landing_beside_the_pad_is_not_achieved(env, book, table):
    runs = (run_goal(env, book, table, seed) for seed in range(2000))
    run = next(r for r in runs if r.info and landing_outcome(r.info, r.observation) == "landed_off_pad")
    assert run.info["landing_success"]
    assert run.status is GoalStatus.FAILED


def test_a_crash_is_a_failure(env, book, table, monkeypatch):
    falling = lambda observation: NO_ENGINE  # noqa: E731
    for name in ("touch_down", "brake", "slow_descent"):
        monkeypatch.setitem(SUBSKILLS, name, dataclasses.replace(SUBSKILLS[name], policy=falling, max_frames=1000))
    run = run_goal(env, book, table, TILTED_START_SEED)
    assert run.status is GoalStatus.FAILED
    assert run.episode_over and run.info["crashed"]


def test_inputs_are_checked(env, book, table):
    observation, _ = env.reset(seed=0)
    with pytest.raises(ValueError, match="unknown goal"):
        pursue(LUNAR_LANDER, env, observation, "hover", book, table, SAFETY_FIRST)
    with pytest.raises(ValueError, match="max_attempts"):
        pursue(LUNAR_LANDER, env, observation, "land_on_pad", book, table, SAFETY_FIRST, max_attempts=0)


def test_the_record_is_json_ready(env, book, table):
    record = json.loads(json.dumps(run_goal(env, book, table, TILTED_START_SEED).record()))
    assert record["status"] == "achieved" and record["replans"] == 0
    first = record["decisions"][0]
    assert first["plan"].startswith("plan:land_on_pad/")
    assert first["gate"] == "CDS" and first["recertified"]
    assert first["goal_graph"]["goal"] == "land_on_pad"
    last = record["decisions"][-1]
    assert last["steps"][-1]["skill"] == "touch_down"
    assert last["settling_frames"] > 0


# Choosing between plans at a decision


def verdict(skills, score=None, reason="rejected by the CDS and PDS gates"):
    candidate = CandidatePlan("g", "m", ("m",), tuple(skills), {})
    entry = object() if score is not None else None
    return PlanVerdict(candidate, reason=reason, entry=entry, score=score)


def test_the_plan_in_progress_is_kept_unless_clearly_beaten():
    current = verdict(["b"], score=100.0)
    slightly_better = verdict(["a"], score=105.0)
    assert _choose([slightly_better, current], ("b",)) is current
    clearly_better = verdict(["a"], score=120.0)
    assert _choose([clearly_better, current], ("b",)) is clearly_better


def test_without_a_plan_in_progress_the_best_plan_is_chosen():
    best = verdict(["a"], score=100.0)
    assert _choose([best, verdict(["b"], score=50.0)], ()) is best
    assert _choose([verdict(["a"])], ()) is None


def test_a_plan_in_progress_the_gates_now_reject_is_replaced():
    other = verdict(["a"], score=1.0)
    assert _choose([other, verdict(["b"])], ("b",)) is other


@pytest.mark.parametrize(
    "verdicts, remaining, allowed",
    [
        ([], (), False),
        ([], ("b",), True),  # the planner no longer proposes this exact rest
        ([verdict(["b"], reason=NOT_MEASURED)], ("b",), True),
        ([verdict(["b"])], ("b",), False),  # the gates rejected it
    ],
)
def test_a_plan_goes_on_only_when_the_rest_was_not_rejected(verdicts, remaining, allowed):
    assert _may_go_on(verdicts, remaining) is allowed


# Confirming a goal


def state(x=0.0, y=0.0, angle=0.0, left=1.0, right=1.0):
    return np.array([x, y, 0.0, 0.0, angle, 0.0, left, right])


@pytest.mark.parametrize(
    "observation, episode_over, info, achieved",
    [
        (state(x=0.1), True, {"landing_success": True}, True),
        (state(x=0.1, right=0.0), True, {"landing_success": True}, True),  # leg contact flickers at rest
        (state(x=0.5), True, {"landing_success": True}, False),  # beside the pad
        (state(x=0.1), True, {"landing_success": False, "crashed": True}, False),  # legs down, then a crash
        (state(x=0.1), True, {"landing_success": False, "crashed": False}, False),  # out of time
        (state(x=0.1), False, None, False),  # legs down mid-episode: not a final result
    ],
)
def test_landing_on_the_pad_is_confirmed_by_the_final_result(book, observation, episode_over, info, achieved):
    assert goal_achieved(LUNAR_LANDER, book.goals["land_on_pad"], observation, episode_over, info) is achieved


def test_a_goal_that_is_not_terminal_is_achieved_while_flying(book):
    recover = book.goals["recover_stability"]
    assert goal_achieved(LUNAR_LANDER, recover, state(y=1.0, left=0.0, right=0.0), False, None)
    assert not goal_achieved(LUNAR_LANDER, recover, state(y=1.0, angle=0.5, left=0.0, right=0.0), False, None)
    assert not goal_achieved(LUNAR_LANDER, recover, state(), True, {"landing_success": False, "crashed": True})

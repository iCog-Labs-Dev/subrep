"""Tests for pursuing goals with planning, certification and replanning (decomposition/runtime.py)."""

import dataclasses
import json

import numpy as np
import pytest

from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable
from decomposition.recipe_book import LUNAR_LANDER_BOOK, load_recipe_book
from decomposition.runtime import GoalStatus, goal_achieved, pursue
from decomposition.verifier import NOT_MEASURED, MdnRegion, Priorities
from env.lunar_facts import FACT_NAMES, landing_outcome, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import NO_ENGINE, SUBSKILLS

TILTED_START_SEED = 0
UPRIGHT_START_SEED = 1  # starts away from the pad; lands with one leg's contact flickering off
OFF_PAD_SEED = 16  # the certified plan lands beside the pad
SAFETY_FIRST = Priorities(
    weights=(0.7, 0.3),
    region=MdnRegion(
        context=(0.0,),
        alpha=(4.0, 1.0),
        support_directions=((1.0, 0.0), (0.0, 1.0)),
        support_values=(1.0, 0.3),
    ),
)
NO_REGION = Priorities(weights=(0.5, 0.5))


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
    return EffectTable.load(DEFAULT_TABLE_PATH)


def run_goal(env, book, table, seed, goal="land_on_pad", priorities=SAFETY_FIRST, **kwargs):
    observation, _ = env.reset(seed=seed)
    return pursue(env, observation, goal, book, table, priorities, **kwargs)


def test_a_certified_landing_plan_is_flown_to_a_landing(env, book, table):
    run = run_goal(env, book, table, TILTED_START_SEED)
    assert run.status is GoalStatus.ACHIEVED
    [attempt] = run.attempts
    assert attempt.chosen.entry.certificate.skill_id == "plan:land_on_pad/fuel_saving_landing"
    assert attempt.execution.completed
    assert run.replans == 0 and run.episode_over
    assert run.frames == attempt.execution.frames
    assert run.info["landing_success"]


def test_the_environment_decides_once_the_episode_is_over(env, book, table):
    run = run_goal(env, book, table, UPRIGHT_START_SEED)
    assert not lander_facts(run.observation)["landed"]
    assert run.info["landing_success"]
    assert run.status is GoalStatus.ACHIEVED


def test_a_goal_that_already_holds_needs_no_plan(env, book, table):
    run = run_goal(env, book, table, UPRIGHT_START_SEED, goal="recover_stability")
    assert run.status is GoalStatus.ALREADY_ACHIEVED
    assert run.attempts == () and run.frames == 0


def test_recovering_stability_leaves_the_lander_upright_in_flight(env, book, table):
    run = run_goal(env, book, table, TILTED_START_SEED, goal="recover_stability")
    assert run.status is GoalStatus.ACHIEVED
    assert lander_facts(run.observation)["upright"]
    assert not run.episode_over and run.frames > 0


def test_without_a_certified_plan_nothing_is_flown(env, book, table):
    observation, _ = env.reset(seed=TILTED_START_SEED)
    run = pursue(env, observation, "land_on_pad", book, table, NO_REGION)
    assert run.status is GoalStatus.NO_CERTIFIED_PLAN
    [attempt] = run.attempts
    assert attempt.execution is None and attempt.chosen is None
    assert all(not verdict.admitted for verdict in attempt.verdicts)
    assert run.frames == 0
    np.testing.assert_array_equal(run.observation, observation)


@pytest.fixture(scope="module")
def wide_table(table):
    """The measured table with its support widened to cover every state, to exercise replanning."""
    return EffectTable({**table.data, "support_margin": 1000.0})


def test_after_a_failed_step_episode_start_measurements_do_not_certify_a_replan(env, book, table, monkeypatch):
    monkeypatch.setitem(SUBSKILLS, "move_over_pad", dataclasses.replace(SUBSKILLS["move_over_pad"], max_frames=20))
    run = run_goal(env, book, table, UPRIGHT_START_SEED, max_attempts=15)
    assert run.status is GoalStatus.NO_CERTIFIED_PLAN
    first, second = run.attempts
    assert first.execution.stopped_at.skill == "move_over_pad"
    assert second.execution is None
    assert {verdict.reason for verdict in second.verdicts} == {NOT_MEASURED}


def test_a_failed_step_is_resumed_when_measurements_apply(env, book, wide_table, monkeypatch):
    monkeypatch.setitem(SUBSKILLS, "move_over_pad", dataclasses.replace(SUBSKILLS["move_over_pad"], max_frames=20))
    run = run_goal(env, book, wide_table, UPRIGHT_START_SEED, max_attempts=15)
    assert run.status is GoalStatus.ACHIEVED
    assert run.replans >= 1
    assert run.attempts[0].execution.stopped_at.skill == "move_over_pad"
    assert run.attempts[-1].execution.completed


def test_attempts_run_out(env, book, wide_table, monkeypatch):
    monkeypatch.setitem(SUBSKILLS, "move_over_pad", dataclasses.replace(SUBSKILLS["move_over_pad"], max_frames=1))
    run = run_goal(env, book, wide_table, UPRIGHT_START_SEED, max_attempts=2)
    assert run.status is GoalStatus.FAILED
    assert len(run.attempts) == 2 and run.replans == 1
    assert not run.episode_over


def test_a_landing_beside_the_pad_is_not_achieved(env, book, table):
    run = run_goal(env, book, table, OFF_PAD_SEED)
    assert run.info["landing_success"]
    assert landing_outcome(run.info, run.observation) == "landed_off_pad"
    assert run.status is GoalStatus.FAILED


def test_a_crash_is_a_failure(env, book, table, monkeypatch):
    falling = dataclasses.replace(SUBSKILLS["touch_down"], policy=lambda observation: NO_ENGINE, max_frames=1000)
    monkeypatch.setitem(SUBSKILLS, "touch_down", falling)
    monkeypatch.setitem(SUBSKILLS, "brake", dataclasses.replace(SUBSKILLS["brake"], policy=lambda observation: NO_ENGINE))
    run = run_goal(env, book, table, TILTED_START_SEED)
    assert run.status is GoalStatus.FAILED
    assert run.episode_over and run.info["crashed"]


def test_inputs_are_checked(env, book, table):
    observation, _ = env.reset(seed=0)
    with pytest.raises(ValueError, match="unknown goal"):
        pursue(env, observation, "hover", book, table, SAFETY_FIRST)
    with pytest.raises(ValueError, match="max_attempts"):
        pursue(env, observation, "land_on_pad", book, table, SAFETY_FIRST, max_attempts=0)


def test_the_record_is_json_ready(env, book, table):
    record = json.loads(json.dumps(run_goal(env, book, table, TILTED_START_SEED).record()))
    assert record["status"] == "achieved" and record["replans"] == 0
    [attempt] = record["attempts"]
    assert attempt["plan"] == "plan:land_on_pad/fuel_saving_landing"
    assert attempt["gate"] == "CDS"
    assert [step["skill"] for step in attempt["steps"]][-1] == "touch_down"
    assert attempt["settling_frames"] > 0


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
    assert goal_achieved(book.goals["land_on_pad"], observation, episode_over, info) is achieved


def test_a_goal_that_is_not_terminal_is_achieved_while_flying(book):
    recover = book.goals["recover_stability"]
    assert goal_achieved(recover, state(y=1.0, left=0.0, right=0.0), False, None)
    assert not goal_achieved(recover, state(y=1.0, angle=0.5, left=0.0, right=0.0), False, None)
    assert not goal_achieved(recover, state(), True, {"landing_success": False, "crashed": True})

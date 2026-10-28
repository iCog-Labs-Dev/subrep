"""Tests for flying plans with runtime checks (decomposition/executor.py)."""

import dataclasses
import json

import numpy as np
import pytest

from decomposition.executor import MAX_SETTLING_FRAMES, StepStatus, execute
from decomposition.recipe_book import LUNAR_LANDER_BOOK, load_recipe_book, parse_recipe_book
from demo.run_lunar_subskill_reliability import SEQUENCES, fly
from env.lunar_facts import FACT_NAMES
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import NO_ENGINE, SUBSKILLS

CAREFUL = SEQUENCES["careful_landing"]
TILTED_START_SEED = 0
OVER_PAD_START_SEED = 2


@pytest.fixture
def env():
    environment = SubRepEnv(seed=0)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def book():
    return load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, SUBSKILLS)


def start(env, seed):
    observation, _ = env.reset(seed=seed)
    return observation


def test_a_landing_plan_completes_and_flies_until_the_episode_ends(env, book):
    execution = execute(env, book, CAREFUL, start(env, TILTED_START_SEED))
    assert execution.completed and execution.stopped_at is None
    assert [step.skill for step in execution.steps] == CAREFUL
    assert all(step.prediction_held for step in execution.steps)
    assert execution.episode_over and execution.settling_frames > 0
    assert execution.info["landing_success"]
    assert execution.frames == len(execution.rewards) == fly(env, CAREFUL, TILTED_START_SEED).frames


def test_a_step_whose_requirements_do_not_hold_is_not_run(env, book):
    execution = execute(env, book, ["move_over_pad", "slow_descent"], start(env, TILTED_START_SEED))
    assert not execution.completed
    [step] = execution.steps
    assert step.status is StepStatus.REQUIREMENTS_NOT_MET
    assert step.frames == 0 and execution.frames == 0
    assert not step.facts_before["upright"]
    assert step.prediction_held is None
    assert not execution.episode_over


def test_a_sub_skill_out_of_frames_stops_the_plan(env, book, monkeypatch):
    monkeypatch.setitem(SUBSKILLS, "stabilise", dataclasses.replace(SUBSKILLS["stabilise"], max_frames=1))
    execution = execute(env, book, CAREFUL, start(env, TILTED_START_SEED))
    assert execution.stopped_at.skill == "stabilise"
    assert execution.stopped_at.status is StepStatus.OUT_OF_FRAMES
    assert execution.frames == 1 and len(execution.steps) == 1


def test_the_end_of_the_episode_stops_the_plan(env, book, monkeypatch):
    never_done = dataclasses.replace(SUBSKILLS["free_fall"], is_done=lambda facts: False, max_frames=1000)
    monkeypatch.setitem(SUBSKILLS, "free_fall", never_done)
    execution = execute(env, book, ["free_fall", "brake"], start(env, OVER_PAD_START_SEED))
    assert execution.stopped_at.status is StepStatus.EPISODE_ENDED
    assert execution.episode_over and execution.info["crashed"]
    assert set(execution.actions) == {NO_ENGINE}


def test_a_wrong_prediction_is_recorded_and_the_plan_goes_on(env):
    data = json.loads(LUNAR_LANDER_BOOK.read_text(encoding="utf-8"))
    data["primitives"]["stabilise"]["effects"]["near_ground"] = True
    wrong_book = parse_recipe_book(data, FACT_NAMES, SUBSKILLS)
    execution = execute(env, wrong_book, CAREFUL[:2], start(env, TILTED_START_SEED))
    assert execution.steps[0].prediction_held is False
    assert execution.completed


def test_a_plan_that_does_not_land_does_not_settle(env, book):
    execution = execute(env, book, ["stabilise"], start(env, TILTED_START_SEED))
    assert execution.completed and execution.steps[0].facts_after["upright"]
    assert execution.settling_frames == 0 and not execution.episode_over


def test_execution_is_reproducible(env, book):
    first = execute(env, book, SEQUENCES["fuel_saving_landing"], start(env, 5))
    second = execute(env, book, SEQUENCES["fuel_saving_landing"], start(env, 5))
    assert first.actions == second.actions
    np.testing.assert_array_equal(first.rewards, second.rewards)


def test_a_plan_needs_sub_skills(env, book):
    with pytest.raises(ValueError, match="at least one"):
        execute(env, book, [], start(env, 0))


def lander(angle=0.0, left=0.0, right=0.0):
    """An observation: level and slow at height 1, unless tilted or with legs down."""
    return np.array([0.0, 1.0, 0.0, 0.0, angle, 0.0, left, right], dtype=np.float32)


class ScriptedEnv:
    """A stand-in environment that returns the same observation and episode end on every step."""

    def __init__(self, observation, ends):
        self.observation, self.ends, self.steps = observation, ends, 0

    def step(self, action):
        self.steps += 1
        return self.observation, np.zeros(2), self.ends, False, {"landing_success": False, "crashed": self.ends}


def test_an_episode_that_ends_as_a_step_completes_stops_the_plan_at_the_next_step(book):
    env = ScriptedEnv(lander(), ends=True)  # the first frame levels the lander and ends the episode
    execution = execute(env, book, CAREFUL[:2], lander(angle=0.5))
    first, second = execution.steps
    assert first.skill == "stabilise" and first.status is StepStatus.COMPLETED
    assert second.skill == "move_over_pad" and second.status is StepStatus.EPISODE_ENDED and second.frames == 0
    assert execution.stopped_at is second and not execution.completed
    assert execution.episode_over and execution.frames == 1


def test_an_episode_that_ends_as_the_last_step_completes_completes_the_plan(book):
    execution = execute(ScriptedEnv(lander(), ends=True), book, ["stabilise"], lander(angle=0.5))
    assert execution.completed and execution.stopped_at is None
    assert execution.episode_over and execution.settling_frames == 0


def test_settling_stops_if_the_environment_never_ends_the_episode(book):
    env = ScriptedEnv(lander(left=1.0, right=1.0), ends=False)
    execution = execute(env, book, ["touch_down"], lander(left=1.0, right=1.0))
    assert execution.completed
    assert execution.settling_frames == MAX_SETTLING_FRAMES == env.steps
    assert not execution.episode_over

"""Tests for the scripted MO-LunarLander sub-skills (env/lunar_subskills.py)."""

import numpy as np
import pytest

from demo.run_lunar_subskill_reliability import SEQUENCES, fly, summarise
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import (
    LEFT_ENGINE,
    MAIN_ENGINE,
    NO_ENGINE,
    RIGHT_ENGINE,
    SUBSKILLS,
    SubSkill,
    angle_towards_pad,
    run_subskill,
    steer,
)

SEEDS = range(10)


def observation(x=0.0, y=1.0, vx=0.0, vy=0.0, angle=0.0, spin=0.0, left=0.0, right=0.0):
    return np.array([x, y, vx, vy, angle, spin, left, right], dtype=np.float32)


@pytest.fixture
def env():
    environment = SubRepEnv(seed=0)
    yield environment
    environment.close()


def test_falling_level_lander_fires_the_main_engine():
    assert steer(observation(vy=-0.5), angle_target=0.0, height_target=1.0) == MAIN_ENGINE


def test_tilt_is_corrected_with_the_side_engines():
    assert steer(observation(angle=0.3), angle_target=0.0, height_target=1.0) == RIGHT_ENGINE
    assert steer(observation(angle=-0.3), angle_target=0.0, height_target=1.0) == LEFT_ENGINE


def test_steady_lander_on_target_does_nothing():
    assert steer(observation(), angle_target=0.0, height_target=1.0) == NO_ENGINE


def test_touching_legs_only_soften_the_fall():
    touching = observation(angle=0.3, vy=-0.5, left=1.0)
    assert steer(touching, angle_target=0.0, height_target=0.0) == MAIN_ENGINE


def test_pad_angle_points_back_to_the_pad_and_is_limited():
    assert angle_towards_pad(observation(x=0.4)) > 0.0
    assert angle_towards_pad(observation(x=-0.4)) < 0.0
    assert angle_towards_pad(observation(x=5.0, vx=5.0)) == pytest.approx(0.4)


def test_a_finished_sub_skill_takes_no_frames(env):
    obs, _ = env.reset(seed=0)
    always_done = SubSkill(name="done", policy=lambda s: NO_ENGINE, is_done=lambda f: True, max_frames=10)
    run = run_subskill(env, always_done, obs)
    assert run.completed and run.frames == 0
    assert run.rewards.shape == (0, 2)


def test_frame_limit_stops_a_sub_skill_that_never_finishes(env):
    obs, _ = env.reset(seed=0)
    never_done = SubSkill(name="never", policy=lambda s: MAIN_ENGINE, is_done=lambda f: False, max_frames=5)
    run = run_subskill(env, never_done, obs)
    assert not run.completed
    assert run.frames == 5
    assert run.actions == (MAIN_ENGINE,) * 5
    assert run.rewards.shape == (5, 2)


def test_same_seed_gives_the_same_frames(env):
    first_obs, _ = env.reset(seed=4)
    first = run_subskill(env, SUBSKILLS["stabilise"], first_obs)
    second_obs, _ = env.reset(seed=4)
    second = run_subskill(env, SUBSKILLS["stabilise"], second_obs)
    assert first.actions == second.actions
    np.testing.assert_array_equal(first.observation, second.observation)


@pytest.mark.parametrize("name", list(SUBSKILLS))
def test_every_sub_skill_finishes_its_job_in_the_flights(name, env):
    sequences = [steps for steps in SEQUENCES.values() if name in steps]
    attempted = completed = 0
    for steps in sequences:
        for seed in SEEDS:
            flight = fly(env, steps, seed)
            attempted += name in flight.completed_steps or flight.failed_step == name
            completed += name in flight.completed_steps
    assert attempted > 0
    assert completed / attempted >= 0.9


@pytest.mark.parametrize("sequence", list(SEQUENCES))
def test_sequences_land_on_the_pad(sequence, env):
    flights = [fly(env, SEQUENCES[sequence], seed) for seed in SEEDS]
    summary = summarise(flights, SEQUENCES[sequence])
    assert summary["landed_on_pad_rate"] >= 0.8
    assert summary["flights"] == len(SEEDS)


def test_fuel_saving_flight_uses_less_fuel_than_careful_flight(env):
    careful = np.mean([fly(env, SEQUENCES["careful_landing"], seed).fuel for seed in SEEDS])
    fuel_saving = np.mean([fly(env, SEQUENCES["fuel_saving_landing"], seed).fuel for seed in SEEDS])
    assert fuel_saving > careful  # fuel totals are negative; closer to zero is less fuel

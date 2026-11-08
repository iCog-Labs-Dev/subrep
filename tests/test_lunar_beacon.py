"""Tests for the scan-then-land mission (env/lunar_beacon.py) and its recipe book."""

import itertools

import numpy as np
import pytest

from decomposition.planner import propose
from decomposition.recipe_book import LUNAR_BEACON_BOOK, holds, load_recipe_book
from env.lunar_beacon import (
    BEACON_FACT_NAMES,
    BEACON_MISSION,
    CRUISE_TO_PAD,
    DEFAULT_BEACON,
    GO_TO_BEACON,
    MISSION_COMPLETE,
    MISSION_REWARD,
    OBJECTIVES,
    SCAN_FRAMES,
    SCAN_REWARD,
    BeaconMissionEnv,
    beacon_domain,
    beacon_facts,
    mission_outcome,
)
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import FACT_NAMES
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_pilots import landing_pilots, mission_pilots
from env.lunar_subskills import NO_ENGINE, SUBSKILLS

SEED = 3
MISSION = ("go_to_beacon", "cruise_to_pad", "slow_descent", "touch_down")


@pytest.fixture
def env():
    environment = BeaconMissionEnv(seed=0)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def book():
    return load_recipe_book(LUNAR_BEACON_BOOK, BEACON_FACT_NAMES, BEACON_MISSION.skills)


def observation(x=0.0, y=1.0, vx=0.0, vy=0.0, progress=0.0):
    return np.array([x, y, vx, vy, 0.0, 0.0, 0.0, 0.0, x + 0.5, y - 0.7, progress], dtype=np.float32)


def fly_mission(env, seed):
    obs, _ = env.reset(seed=seed)
    for name in MISSION:
        run = BEACON_MISSION.run_skill(env, BEACON_MISSION.skills[name], obs)
        obs, info = run.observation, run.info
        if run.terminated or run.truncated:
            return obs, info, run
    while True:
        obs, _, terminated, truncated, info = env.step(int(SUBSKILLS["touch_down"].policy(obs)))
        if terminated or truncated:
            return obs, info, run


# The environment


def test_the_physics_and_the_first_two_rewards_do_not_change(env):
    plain = SubRepEnv(seed=0)
    try:
        a, _ = env.reset(seed=SEED)
        b, _ = plain.reset(seed=SEED)
        np.testing.assert_array_equal(a[:8], b)
        for action in [2, 1, 3, 0] * 10:
            a, reward_a, _, _, _ = env.step(action)
            b, reward_b, _, _, _ = plain.step(action)
            np.testing.assert_array_equal(a[:8], b)
            np.testing.assert_array_equal(reward_a[:2], reward_b)
            assert reward_a[2] == 0.0
    finally:
        plain.close()


def test_observations_carry_the_offset_to_the_beacon_and_the_scan_progress(env):
    obs, _ = env.reset(seed=SEED)
    assert obs.shape == (11,)
    assert obs[8] == pytest.approx(obs[0] - DEFAULT_BEACON[0])
    assert obs[9] == pytest.approx(obs[1] - DEFAULT_BEACON[1])
    assert obs[10] == 0.0


def test_a_scan_needs_the_lander_near_and_steady_for_enough_frames_in_a_row(env):
    near = np.array([DEFAULT_BEACON[0], DEFAULT_BEACON[1], 0.0, 0.0], dtype=np.float32)
    far = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    fast = np.array([DEFAULT_BEACON[0], DEFAULT_BEACON[1], 1.0, 0.0], dtype=np.float32)
    env.reset(seed=SEED)
    assert [env._update_scan(near) for _ in range(SCAN_FRAMES - 1)] == [0.0] * (SCAN_FRAMES - 1)
    assert env._update_scan(far) == 0.0
    assert [env._update_scan(near) for _ in range(SCAN_FRAMES - 1)] == [0.0] * (SCAN_FRAMES - 1)
    assert env._update_scan(fast) == 0.0
    assert [env._update_scan(near) for _ in range(SCAN_FRAMES - 1)] == [0.0] * (SCAN_FRAMES - 1)
    assert env._update_scan(near) == SCAN_REWARD
    assert env._update_scan(near) == 0.0  # the reward is paid once


def test_a_reset_forgets_the_scan(env):
    env.reset(seed=SEED)
    env._scanned = True
    obs, _ = env.reset(seed=SEED)
    assert obs[10] == 0.0 and not beacon_facts(obs)["beacon_scanned"]


def test_the_scripted_mission_earns_both_mission_rewards(env):
    obs, _ = env.reset(seed=SEED)
    total = 0.0
    done = False
    for name in MISSION:
        run = BEACON_MISSION.run_skill(env, BEACON_MISSION.skills[name], obs)
        obs = run.observation
        total += float(run.rewards[:, 2].sum())
        done = run.terminated or run.truncated
        if done:
            break
    while not done:
        obs, reward, terminated, truncated, info = env.step(int(SUBSKILLS["touch_down"].policy(obs)))
        total += float(reward[2])
        done = terminated or truncated
    assert beacon_facts(obs)["beacon_scanned"]
    assert total == SCAN_REWARD + MISSION_REWARD


def test_landing_without_a_scan_does_not_complete_the_mission():
    assert mission_outcome({"landing_success": True, "mission_complete": False}, observation(x=0.1)) == "landed_on_pad"
    assert mission_outcome({"landing_success": True, "mission_complete": False}, observation(x=0.5)) == "landed_off_pad"
    assert mission_outcome({"landing_success": True, "mission_complete": True}, observation(x=0.1)) == MISSION_COMPLETE
    assert mission_outcome({"crashed": True}, observation()) == "crashed"
    assert mission_outcome({}, observation()) == "timeout"


# Facts and domain


def test_facts_extend_the_landers_facts():
    facts = beacon_facts(observation(progress=1.0))
    assert tuple(facts) == BEACON_FACT_NAMES and set(FACT_NAMES) <= set(facts)
    assert facts["beacon_scanned"]
    assert not beacon_facts(observation(progress=0.9))["beacon_scanned"]


@pytest.mark.parametrize("x, vx, centred", [(0.0, 0.0, True), (0.15, 0.0, False), (0.0, 0.15, False), (-0.09, 0.09, True)])
def test_being_centred_needs_the_middle_of_the_pad_and_little_drift(x, vx, centred):
    assert beacon_facts(observation(x=x, vx=vx))["centred_over_pad"] is centred


def test_the_discount_looks_as_far_ahead_as_a_mission_takes():
    assert 1 / (1 - LUNAR_LANDER.gamma) < 500 <= 1 / (1 - BEACON_MISSION.gamma)


def test_a_moved_beacon_gets_its_own_domain():
    moved = beacon_domain((0.5, 0.7))
    assert moved.environment != BEACON_MISSION.environment
    assert moved.objectives == OBJECTIVES
    env = moved.make_env(0)
    obs, _ = env.reset(seed=SEED)
    env.close()
    assert obs[8] == pytest.approx(obs[0] - 0.5)


def test_the_final_result_decides_the_mission(book):
    final = BEACON_MISSION.final_facts({"landing_success": True}, observation(x=0.1, y=0.0, progress=1.0))
    assert holds(book.goals["scan_then_land"].achieved_when, final)
    unscanned = BEACON_MISSION.final_facts({"landing_success": True}, observation(x=0.1, y=0.0))
    assert not holds(book.goals["scan_then_land"].achieved_when, unscanned)
    assert holds(book.goals["land_on_pad"].achieved_when, unscanned)
    assert BEACON_MISSION.final_facts({"crashed": True}, observation()) is None


# The book


FLYING = {
    **dict.fromkeys(BEACON_FACT_NAMES, False),
    "upright": True,
}


def test_the_book_decomposes_the_mission_into_subgoals_and_skills(book):
    candidates = propose(book, "scan_then_land", FLYING)
    assert MISSION in [c.skills for c in candidates]
    careful = next(c for c in candidates if c.skills == MISSION)
    assert careful.option == "scan_then_careful_landing"
    assert [step.skills for step in careful.subgoals] == [("go_to_beacon",), ("cruise_to_pad",), ("slow_descent",), ("touch_down",)]


def test_a_finished_scan_is_skipped(book):
    candidates = propose(book, "scan_then_land", {**FLYING, "beacon_scanned": True})
    assert all("go_to_beacon" not in c.skills for c in candidates)
    assert ("cruise_to_pad", "slow_descent", "touch_down") in [c.skills for c in candidates]


def test_landing_alone_does_not_need_the_beacon(book):
    candidates = propose(book, "land_on_pad", FLYING)
    assert ("slow_descent", "touch_down") in [c.skills for c in candidates]
    assert all("go_to_beacon" not in c.skills for c in candidates)


def test_only_the_mission_pilot_is_offered_for_the_mission(book):
    scanning = {c.skills for c in propose(book, "scan_then_land", FLYING)}
    landing = {c.skills for c in propose(book, "land_on_pad", FLYING)}
    assert ("mission_pilot",) in scanning
    assert ("heuristic_pilot",) not in scanning and ("ppo_pilot",) not in scanning
    assert ("heuristic_pilot",) in landing and ("ppo_pilot",) in landing


def test_each_goal_names_the_endings_that_achieve_it(book):
    assert book.goals["scan_then_land"].success_outcomes == (MISSION_COMPLETE,)
    assert set(book.goals["land_on_pad"].success_outcomes) == {MISSION_COMPLETE, "landed_on_pad"}


def possible_facts():
    for values in itertools.product((False, True), repeat=len(BEACON_FACT_NAMES)):
        facts = dict(zip(BEACON_FACT_NAMES, values))
        consistent = facts["descent_underway"] == (facts["falling_fast"] or facts["near_ground"])
        if consistent and not (facts["slow"] and facts["falling_fast"]) and (facts["over_pad"] or not facts["centred_over_pad"]):
            yield facts


@pytest.mark.parametrize("name", ["go_to_beacon", "cruise_to_pad", "slow_descent", "free_fall", "brake", "touch_down"])
def test_the_book_effects_follow_from_how_each_skill_stops(book, name):
    stops = [facts for facts in possible_facts() if BEACON_MISSION.skills[name].is_done(facts)]
    assert stops
    for facts in stops:
        assert holds(book.skills[name].effects, facts), (name, facts)


def test_the_book_predicts_the_facts_after_every_completed_step(book):
    env = BEACON_MISSION.make_env(0)
    try:
        for seed in range(3200, 3240):
            obs, _ = env.reset(seed=seed)
            for name in MISSION:
                before = beacon_facts(obs)
                assert holds(book.skills[name].requires, before), (seed, name)
                run = BEACON_MISSION.run_skill(env, BEACON_MISSION.skills[name], obs)
                obs = run.observation
                if run.completed:
                    assert holds(book.skills[name].apply(before), beacon_facts(obs)), (seed, name)
                if run.terminated or run.truncated or not run.completed:
                    break
    finally:
        env.close()


# Skills and pilots


def test_the_mission_skills_mostly_complete_the_mission(env):
    completed = 0
    for seed in range(3200, 3220):
        obs, info, _ = fly_mission(env, seed)
        completed += mission_outcome(info, obs) == MISSION_COMPLETE
    assert completed >= 16


def test_the_skills_stop_on_their_conditions():
    assert GO_TO_BEACON.is_done({"beacon_scanned": True})
    assert not GO_TO_BEACON.is_done({"beacon_scanned": False})
    assert CRUISE_TO_PAD.is_done({"centred_over_pad": True})


@pytest.mark.parametrize("name", ["heuristic_pilot", "ppo_pilot"])
def test_landing_pilots_read_only_the_landers_numbers(env, name):
    obs, _ = env.reset(seed=SEED)
    assert landing_pilots()[name].policy(obs) in (0, 1, 2, 3)


def test_the_mission_pilot_reads_the_whole_observation(env):
    obs, _ = env.reset(seed=SEED)
    assert mission_pilots()["mission_pilot"].policy(obs) in (0, 1, 2, 3)


def test_a_pilot_stops_when_the_lander_is_down():
    pilot = landing_pilots()["heuristic_pilot"]
    assert pilot.is_done({"landed": True}) and not pilot.is_done({"landed": False})


def test_doing_nothing_crashes_without_a_mission_reward(env):
    env.reset(seed=SEED)
    total, done = 0.0, False
    while not done:
        _, reward, terminated, truncated, info = env.step(NO_ENGINE)
        total += float(reward[2])
        done = terminated or truncated
    assert total == 0.0 and info["crashed"] and not info["mission_complete"]

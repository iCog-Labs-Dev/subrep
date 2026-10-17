"""Tests for paired-replay effect measurement (decomposition/effects.py)."""

import numpy as np
import pytest

from decomposition.effects import (
    DEFAULT_TABLE_PATH,
    GAMMA,
    EffectTable,
    Sample,
    build_table,
    discounted_totals,
    fly_plan,
    group_key,
    improvement,
    lower_bound,
    replay,
    summarise,
)
from demo.build_lunar_effect_table import PLANS
from demo.run_lunar_subskill_reliability import SEQUENCES, fly
from env.lunar_facts import lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import LEFT_ENGINE, MAIN_ENGINE, NO_ENGINE, RIGHT_ENGINE

CAREFUL = SEQUENCES["careful_landing"]
FUEL_SAVING = SEQUENCES["fuel_saving_landing"]
TILTED_START_SEED = 0  # starts tilted, so stabilise has work to do
UPRIGHT_START_SEED = 1  # starts upright and not falling fast, so free_fall runs for some frames


def facts_of(group):
    return {name: value == "1" for name, value in (part.split("=") for part in group.split(","))}


@pytest.fixture
def env():
    environment = SubRepEnv(seed=0)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def small_table():
    return build_table(range(4), [CAREFUL, FUEL_SAVING], min_samples=1)


@pytest.fixture(scope="module")
def committed_table():
    return EffectTable.load(DEFAULT_TABLE_PATH)


# Paired measurement


def test_replay_reaches_the_same_state(env):
    actions = [MAIN_ENGINE, LEFT_ENGINE, NO_ENGINE, RIGHT_ENGINE] * 5
    np.testing.assert_array_equal(replay(env, 3, actions), replay(env, 3, actions))


def test_replay_refuses_actions_past_the_end_of_the_episode(env):
    with pytest.raises(ValueError, match="ended"):
        replay(env, 0, [NO_ENGINE] * 1000)


def test_discounting_follows_the_skill_executor():
    payoff, motives = discounted_totals(np.array([[1.0, -1.0], [2.0, 0.0], [0.0, -1.0]]))
    expected = np.array([1.0 + GAMMA * 2.0, -1.0 - GAMMA**2])
    np.testing.assert_allclose(motives, expected)
    assert payoff == pytest.approx(expected.sum())


def test_no_frames_gain_nothing():
    empty = np.zeros((0, 2))
    assert improvement(empty, empty) == (0.0, (0.0, 0.0))


def test_doing_nothing_gains_nothing_over_doing_nothing(env):
    flight = fly_plan(env, UPRIGHT_START_SEED, ["free_fall"])
    assert flight.plan.frames > 0
    assert flight.plan.delta_r == 0.0
    assert flight.plan.delta_n == (0.0, 0.0)


def test_firing_engines_costs_fuel(env):
    flight = fly_plan(env, TILTED_START_SEED, ["stabilise"])
    assert flight.plan.frames > 0
    assert flight.plan.delta_n[1] < 0.0


def test_a_finished_step_takes_no_frames_and_gains_nothing(env):
    flight = fly_plan(env, UPRIGHT_START_SEED, ["stabilise"])
    assert flight.plan.completed
    assert flight.plan.frames == 0
    assert flight.plan.delta_r == 0.0


def test_a_step_inside_a_plan_matches_the_step_flown_alone(env):
    in_plan = fly_plan(env, TILTED_START_SEED, CAREFUL).steps[0]
    alone = fly_plan(env, TILTED_START_SEED, CAREFUL[:1])
    assert in_plan == alone.steps[0] == alone.plan
    np.testing.assert_array_equal(alone.interaction, np.zeros(3))


def test_a_landing_plan_runs_until_the_episode_ends(env):
    flight = fly_plan(env, TILTED_START_SEED, CAREFUL)
    assert flight.plan.completed
    assert [step.name for step in flight.steps] == CAREFUL
    assert flight.plan.frames == fly(env, CAREFUL, TILTED_START_SEED).frames
    assert flight.plan.frames > sum(step.frames for step in flight.steps)


def test_samples_are_grouped_by_their_start_facts(env):
    flight = fly_plan(env, TILTED_START_SEED, CAREFUL)
    start_facts = lander_facts(replay(env, TILTED_START_SEED, ()))
    assert flight.plan.group == flight.steps[0].group == group_key(start_facts)


def test_a_plan_needs_sub_skills(env):
    with pytest.raises(ValueError, match="at least one"):
        fly_plan(env, 0, [])


# Summaries


def test_lower_bound_is_below_the_mean_and_reproducible():
    values = np.random.default_rng(0).normal(5.0, 2.0, size=200)
    bound = lower_bound(values, seed=1)
    assert bound < values.mean()
    assert bound == lower_bound(values, seed=1)


def test_lower_bound_of_a_constant_is_the_constant():
    assert lower_bound([3.0] * 10, seed=0) == pytest.approx(3.0)


def test_lower_bound_needs_values():
    with pytest.raises(ValueError):
        lower_bound([], seed=0)


def test_summarise_rejects_samples_from_different_groups():
    first = Sample("brake", "upright=1,over_pad=1,near_ground=0", 1.0, (1.0, 0.0), 5, True)
    second = Sample("brake", "upright=0,over_pad=1,near_ground=0", 1.0, (1.0, 0.0), 5, True)
    with pytest.raises(ValueError, match="one group"):
        summarise([first, second])


# The table


def test_the_table_is_reproducible_from_its_seeds(small_table):
    assert build_table(range(4), [CAREFUL, FUEL_SAVING], min_samples=1).data == small_table.data


def test_a_shared_first_step_is_counted_once_per_seed(small_table):
    stabilise = [entry for entry in small_table.data["steps"] if entry["name"] == "stabilise"]
    assert sum(entry["samples"] for entry in stabilise) == 4


def test_the_table_round_trips_through_json(small_table, tmp_path):
    path = tmp_path / "effects.json"
    small_table.save(path)
    assert EffectTable.load(path).data == small_table.data


def test_lookup_returns_the_entry_for_the_facts(small_table):
    entry = small_table.data["plans"][0]
    estimate = small_table.plan(entry["skills"], facts_of(entry["group"]))
    assert estimate.samples == entry["samples"]
    assert estimate.mean.delta_r == entry["delta_r"]["mean"]
    assert estimate.lower.delta_n == (entry["delta_safety"]["lower"], entry["delta_fuel"]["lower"])
    assert estimate.interaction.delta_r == entry["interaction"]["delta_r"]["mean"]


def test_steps_have_no_interaction(small_table):
    entry = small_table.data["steps"][0]
    assert small_table.step(entry["name"], facts_of(entry["group"])).interaction is None


def test_unmeasured_or_thin_groups_are_unknown(small_table):
    facts = facts_of(small_table.data["plans"][0]["group"])
    assert small_table.step("hover", facts) is None
    strict = EffectTable({**small_table.data, "min_samples": 1000})
    assert strict.plan(CAREFUL, facts) is None


def test_the_table_checks_its_format(small_table):
    with pytest.raises(ValueError, match="schema_version"):
        EffectTable({**small_table.data, "schema_version": "other"})
    with pytest.raises(ValueError, match="groups by"):
        EffectTable({**small_table.data, "group_facts": ["upright"]})


# The committed table


def test_committed_table_covers_every_plan_from_unseen_episode_starts(committed_table, env):
    unseen = range(committed_table.data["seeds"]["stop"], committed_table.data["seeds"]["stop"] + 30)
    for seed in unseen:
        facts = lander_facts(replay(env, seed, ()))
        for skills in PLANS:
            assert committed_table.plan(skills, facts) is not None, (seed, skills)


def test_committed_landing_plans_are_safer_than_doing_nothing_and_differ_in_fuel(committed_table):
    groups = {entry["group"] for entry in committed_table.data["plans"]}
    assert groups
    for group in groups:
        careful = committed_table.plan(CAREFUL, facts_of(group))
        fuel_saving = committed_table.plan(FUEL_SAVING, facts_of(group))
        assert careful.lower.delta_n[0] > 0.0, group
        assert fuel_saving.lower.delta_n[0] > 0.0, group
        assert fuel_saving.mean.delta_n[1] > careful.mean.delta_n[1], group  # fuel gains are negative


def test_committed_doing_nothing_steps_gain_nothing(committed_table):
    free_fall = [entry for entry in committed_table.data["steps"] if entry["name"] == "free_fall"]
    assert free_fall
    assert all(entry["delta_r"]["mean"] == 0.0 and entry["delta_fuel"]["mean"] == 0.0 for entry in free_fall)

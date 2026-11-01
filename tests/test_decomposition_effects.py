"""Tests for paired-replay effect measurement (decomposition/effects.py)."""

import dataclasses

import numpy as np
import pytest

from decomposition.effects import (
    DEFAULT_TABLE_PATH,
    GAMMA,
    IN_FLIGHT,
    EffectTable,
    Sample,
    build_table,
    discounted_totals,
    fly_plan,
    group_key,
    improvement,
    lower_bound,
    measure_seed,
    replay,
    summarise,
)
from decomposition.planner import plans_for
from decomposition.recipe_book import LUNAR_LANDER_BOOK, load_recipe_book
from demo.run_lunar_subskill_reliability import SEQUENCES, fly
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import OUTCOMES, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_facts import FACT_NAMES
from env.lunar_subskills import LEFT_ENGINE, MAIN_ENGINE, NO_ENGINE, RIGHT_ENGINE, SUBSKILLS

CAREFUL = SEQUENCES["careful_landing"]
FUEL_SAVING = SEQUENCES["fuel_saving_landing"]
TILTED_START_SEED = 0  # starts tilted, so stabilise has work to do
UPRIGHT_START_SEED = 1  # starts upright and not falling fast, so free_fall runs for some frames
PLAN_OUTCOMES = OUTCOMES + (IN_FLIGHT,)


def episode_start(group, seeds=range(1000)):
    """The start observation of the first seed whose start falls in ``group``."""
    env = SubRepEnv(seed=0)
    try:
        for seed in seeds:
            observation = replay(env, seed, ())
            if group_key(LUNAR_LANDER, lander_facts(observation)) == group:
                return observation
    finally:
        env.close()
    raise AssertionError(f"no episode start in group {group}")


def both_plans(facts):
    return [CAREFUL, FUEL_SAVING]


@pytest.fixture
def env():
    environment = SubRepEnv(seed=0)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def small_table():
    return build_table(LUNAR_LANDER, range(4), both_plans, min_samples=1, max_states=1)


@pytest.fixture(scope="module")
def committed_table():
    return EffectTable.load(DEFAULT_TABLE_PATH, LUNAR_LANDER)


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
    flight = fly_plan(LUNAR_LANDER, env, UPRIGHT_START_SEED, ["free_fall"])
    assert flight.plan.frames > 0
    assert flight.plan.delta_r == 0.0
    assert flight.plan.delta_n == (0.0, 0.0)


def test_firing_engines_costs_fuel(env):
    flight = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, ["stabilise"])
    assert flight.plan.frames > 0
    assert flight.plan.delta_n[1] < 0.0


def test_a_finished_step_takes_no_frames_and_gains_nothing(env):
    flight = fly_plan(LUNAR_LANDER, env, UPRIGHT_START_SEED, ["stabilise"])
    assert flight.plan.completed
    assert flight.plan.frames == 0
    assert flight.plan.delta_r == 0.0


def test_a_step_inside_a_plan_matches_the_step_flown_alone(env):
    in_plan = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL).steps[0]
    alone = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL[:1])
    assert alone.outcome == IN_FLIGHT
    assert in_plan == alone.steps[0] == alone.plan
    np.testing.assert_array_equal(alone.interaction, np.zeros(3))


def test_a_plan_can_be_measured_from_a_state_in_mid_flight(env):
    whole = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL)
    prefix = whole.next_states[0]
    assert len(prefix) > 0
    rest = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL[1:], prefix)
    assert rest.plan.start == tuple(float(v) for v in replay(env, TILTED_START_SEED, prefix))
    assert rest.outcome == whole.outcome
    assert rest.prefix == prefix


def test_a_flight_lists_the_states_the_runtime_would_decide_at(env):
    flight = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL[:2])
    assert len(flight.next_states) == 2
    assert flight.next_states[0] == flight.next_states[1][: len(flight.next_states[0])]


def test_measuring_a_seed_visits_the_states_along_each_plan(env):
    flights = measure_seed(LUNAR_LANDER, env, TILTED_START_SEED, lambda facts: [CAREFUL[-2:]])
    assert flights[0].prefix == () and flights[0].skills == tuple(CAREFUL[-2:])
    assert {flight.prefix for flight in flights} == {(), *flights[0].next_states[:2]} or len(flights) > 1
    assert all(flight.steps == () for flight in flights if flight.prefix)


def test_a_flight_from_mid_flight_measures_only_the_whole_plan(env):
    whole = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL)
    rest = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL[1:], whole.next_states[0])
    assert rest.steps == ()
    assert len(whole.steps) == len(CAREFUL)


def test_measuring_a_seed_stops_at_its_state_limit(env):
    flights = measure_seed(LUNAR_LANDER, env, TILTED_START_SEED, lambda facts: [CAREFUL[-2:]], max_states=1)
    assert {flight.prefix for flight in flights} == {()}


def test_a_failed_flight_is_charged_for_what_happens_after_it(env, monkeypatch):
    monkeypatch.setitem(SUBSKILLS, "stabilise", dataclasses.replace(SUBSKILLS["stabilise"], max_frames=1))
    flight = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL)
    assert not flight.plan.completed
    assert flight.plan.frames > 1
    assert flight.outcome == "crashed"


def test_a_landing_plan_runs_until_the_episode_ends(env):
    flight = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL)
    assert flight.plan.completed
    assert flight.outcome == "landed_on_pad"
    assert [step.name for step in flight.steps] == CAREFUL
    assert flight.plan.frames == fly(env, CAREFUL, TILTED_START_SEED).frames
    assert flight.plan.frames > sum(step.frames for step in flight.steps)


def test_samples_are_grouped_by_their_start_facts(env):
    flight = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL)
    start_facts = lander_facts(replay(env, TILTED_START_SEED, ()))
    assert flight.plan.group == flight.steps[0].group == group_key(LUNAR_LANDER, start_facts)


def test_a_plan_needs_sub_skills(env):
    with pytest.raises(ValueError, match="at least one"):
        fly_plan(LUNAR_LANDER, env, 0, [])


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
    first = Sample("brake", "upright=1,over_pad=1,near_ground=0", 1.0, (1.0, 0.0), 5, True, (0.0,) * 8)
    second = Sample("brake", "upright=0,over_pad=1,near_ground=0", 1.0, (1.0, 0.0), 5, True, (0.0,) * 8)
    with pytest.raises(ValueError, match="one group"):
        summarise([first, second], LUNAR_LANDER.objectives)


# The table


def test_the_table_is_reproducible_from_its_seeds(small_table):
    assert build_table(LUNAR_LANDER, range(4), both_plans, min_samples=1, max_states=1).data == small_table.data


def test_a_shared_first_step_is_counted_once_per_seed(small_table):
    stabilise = [entry for entry in small_table.data["steps"] if entry["name"] == "stabilise"]
    assert sum(entry["samples"] for entry in stabilise) == 4


def test_the_table_round_trips_through_json(small_table, tmp_path):
    path = tmp_path / "effects.json"
    small_table.save(path)
    assert EffectTable.load(path, LUNAR_LANDER).data == small_table.data


def test_lookup_returns_the_entry_for_the_state(small_table):
    entry = small_table.data["plans"][0]
    estimate = small_table.plan(entry["skills"], episode_start(entry["group"], range(4)))
    assert estimate.samples == entry["samples"]
    assert estimate.mean.delta_r == entry["delta_r"]["mean"]
    assert estimate.lower.delta_n == (entry["delta_safety"]["lower"], entry["delta_fuel"]["lower"])
    assert estimate.interaction.delta_r == entry["interaction"]["delta_r"]["mean"]


def first_steps(table):
    """Step entries for stabilise, which starts every plan at the start of an episode."""
    return [entry for entry in table.data["steps"] if entry["name"] == "stabilise"]


def test_steps_have_no_interaction(small_table):
    entry = first_steps(small_table)[0]
    assert small_table.step("stabilise", episode_start(entry["group"], range(4))).interaction is None


def test_unmeasured_or_thin_groups_are_unknown(small_table):
    start = episode_start(small_table.data["plans"][0]["group"], range(4))
    assert small_table.step("hover", start) is None
    strict = EffectTable({**small_table.data, "min_samples": 1000}, LUNAR_LANDER)
    assert strict.plan(CAREFUL, start) is None


def test_the_support_holds_every_measured_start(small_table, env):
    for entry in small_table.data["plans"]:
        for seed in range(4):
            start = replay(env, seed, ())
            if group_key(LUNAR_LANDER, lander_facts(start)) == entry["group"]:
                assert small_table.supports(entry, start)


def test_a_state_outside_the_support_is_not_measured(small_table):
    entry = small_table.data["plans"][0]
    start = episode_start(entry["group"], range(4))
    lower = np.array(start, dtype=np.float64)
    lower[1] -= 0.5  # same facts, but half a unit lower than any measured start
    assert group_key(LUNAR_LANDER, lander_facts(lower)) == entry["group"]
    assert small_table.plan(entry["skills"], start) is not None
    assert small_table.plan(entry["skills"], lower) is None


def test_the_table_checks_its_format(small_table):
    with pytest.raises(ValueError, match="schema_version"):
        EffectTable({**small_table.data, "schema_version": "other"}, LUNAR_LANDER)
    with pytest.raises(ValueError, match="measures 'other'"):
        EffectTable({**small_table.data, "environment": "other"}, LUNAR_LANDER)
    with pytest.raises(ValueError, match="groups by"):
        EffectTable({**small_table.data, "group_facts": ["upright"]}, LUNAR_LANDER)
    with pytest.raises(ValueError, match="objectives"):
        EffectTable({**small_table.data, "objectives": ["safety"]}, LUNAR_LANDER)
    with pytest.raises(KeyError):
        EffectTable({key: value for key, value in small_table.data.items() if key != "support_margin"}, LUNAR_LANDER)


# The committed table


def test_committed_table_covers_every_plan_from_unseen_episode_starts(committed_table, env):
    plans = plans_for(load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, SUBSKILLS))
    unseen = range(committed_table.data["seeds"]["stop"], committed_table.data["seeds"]["stop"] + 30)
    for seed in unseen:
        start = replay(env, seed, ())
        for skills in plans(lander_facts(start)):
            assert committed_table.plan(skills, start) is not None, (seed, skills)


def test_committed_table_answers_from_states_reached_in_mid_flight(committed_table, env):
    unseen = committed_table.data["seeds"]["stop"]
    plans = plans_for(load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, SUBSKILLS))
    answered = 0
    for seed in range(unseen, unseen + 30):
        flight = fly_plan(LUNAR_LANDER, env, seed, CAREFUL)
        for prefix in flight.next_states[:-1]:
            state = replay(env, seed, prefix)
            answered += any(committed_table.plan(skills, state) is not None for skills in plans(lander_facts(state)))
    assert answered > 30


def test_committed_table_refuses_a_state_no_flight_reached(committed_table):
    entry = committed_table.data["plans"][0]
    outside = np.asarray(entry["support"]["low"], dtype=np.float64) - 1.0
    assert committed_table.plan(entry["skills"], outside) is None


def test_committed_landing_plans_are_safer_than_doing_nothing_and_differ_in_fuel(committed_table):
    groups = {e["group"]: e for e in committed_table.data["plans"] if tuple(e["skills"]) == tuple(CAREFUL[1:])}
    compared = 0
    for group, careful in groups.items():
        fuel_saving = [
            e for e in committed_table.data["plans"] if e["group"] == group and e["skills"] == ["move_over_pad", "free_fall", "brake", "touch_down"]
        ]
        if fuel_saving:
            compared += 1
            assert careful["delta_safety"]["lower"] > 0.0, group
            assert fuel_saving[0]["delta_safety"]["lower"] > 0.0, group
            assert fuel_saving[0]["delta_fuel"]["mean"] > careful["delta_fuel"]["mean"], group  # fuel gains are negative
    assert compared > 0


def test_committed_doing_nothing_steps_gain_nothing(committed_table):
    free_fall = [entry for entry in committed_table.data["steps"] if entry["name"] == "free_fall"]
    assert free_fall
    assert all(entry["delta_r"]["mean"] == 0.0 and entry["delta_fuel"]["mean"] == 0.0 for entry in free_fall)


def test_plan_outcomes_come_from_the_final_result(small_table, committed_table):
    for table in (small_table, committed_table):
        for entry in table.data["plans"]:
            assert set(entry["outcomes"]) == set(PLAN_OUTCOMES)
            assert sum(entry["outcomes"].values()) == pytest.approx(1.0, abs=1e-5)
    landings = [e for e in committed_table.data["plans"] if "touch_down" in e["skills"]]
    # Completed steps overstate landings on the pad: some end off the pad or crash after touching down.
    assert max(e["completion_rate"] - e["outcomes"]["landed_on_pad"] for e in landings) > 0.05


def test_estimates_report_how_plans_really_ended(small_table):
    entry = small_table.data["plans"][0]
    estimate = small_table.plan(entry["skills"], episode_start(entry["group"], range(4)))
    assert estimate.outcomes == entry["outcomes"]
    step = first_steps(small_table)[0]
    assert small_table.step("stabilise", episode_start(step["group"], range(4))).outcomes is None

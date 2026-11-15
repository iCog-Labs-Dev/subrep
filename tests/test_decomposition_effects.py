"""Tests for paired-replay effect measurement (decomposition/effects.py)."""

import dataclasses

import numpy as np
import pytest

from decomposition import effects
from decomposition.effects import (
    DEFAULT_TABLE_PATH,
    GAMMA,
    IN_FLIGHT,
    EffectTable,
    Sample,
    build_table,
    cell_scale,
    discounted_totals,
    fly_plan,
    group_key,
    improvement,
    lower_bound,
    measure_seed,
    partition,
    replay,
    state,
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


def both_plans(facts, at_start):
    return [CAREFUL, FUEL_SAVING]


def last_two_skills(facts, at_start):
    return [CAREFUL[-2:]]


def switch_to_fuel_saving(facts, at_start):
    """The careful landing at the start, and the rest of the fuel-saving landing in mid-flight."""
    return [CAREFUL] if at_start else [FUEL_SAVING[1:]]


@pytest.fixture
def env():
    environment = SubRepEnv(seed=0)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def small_table():
    return build_table(LUNAR_LANDER, range(4), both_plans, min_samples=1, max_states=1, cell_episodes=None)


@pytest.fixture(scope="module")
def celled_table():
    return build_table(LUNAR_LANDER, range(4), both_plans, min_samples=1, max_states=1, cell_episodes=1)


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
    flight = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL[:3])
    assert len(flight.next_states) == 2
    assert flight.next_states[0] == flight.next_states[1][: len(flight.next_states[0])]


def test_nothing_is_decided_after_the_last_skill_of_a_plan(env):
    assert fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL[:1]).next_states == ()


def test_the_state_after_a_failed_skill_is_decided_at(env, monkeypatch):
    monkeypatch.setitem(SUBSKILLS, "stabilise", dataclasses.replace(SUBSKILLS["stabilise"], max_frames=1))
    flight = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL[:1])
    assert not flight.steps[0].completed and len(flight.next_states) == 1


def test_measuring_a_seed_visits_the_states_along_each_plan(env):
    flights = measure_seed(LUNAR_LANDER, env, TILTED_START_SEED, lambda facts, at_start: [CAREFUL[-2:]])
    assert flights[0].prefix == () and flights[0].skills == tuple(CAREFUL[-2:])
    assert flights[0].next_states
    assert {(), *flights[0].next_states} <= {flight.prefix for flight in flights}
    assert all(flight.steps == () for flight in flights if flight.prefix)


def test_measuring_a_seed_visits_the_states_along_plans_flown_in_mid_flight(env):
    flights = measure_seed(LUNAR_LANDER, env, TILTED_START_SEED, switch_to_fuel_saving)
    after_a_switch = {flight.prefix for flight in flights} - {(), *flights[0].next_states}
    assert after_a_switch
    assert after_a_switch <= {state for flight in flights if flight.prefix for state in flight.next_states}


def test_measuring_a_seed_visits_states_breadth_first_up_to_its_limit(env):
    flights = measure_seed(LUNAR_LANDER, env, TILTED_START_SEED, last_two_skills, max_states=2)
    assert {flight.prefix for flight in flights} == {(), flights[0].next_states[0]}


def test_a_flight_from_mid_flight_measures_only_the_whole_plan(env):
    whole = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL)
    rest = fly_plan(LUNAR_LANDER, env, TILTED_START_SEED, CAREFUL[1:], whole.next_states[0])
    assert rest.steps == ()
    assert len(whole.steps) == len(CAREFUL)


def test_measuring_a_seed_stops_at_its_state_limit(env):
    flights = measure_seed(LUNAR_LANDER, env, TILTED_START_SEED, lambda facts, at_start: [CAREFUL[-2:]], max_states=1)
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


def test_lower_bound_resamples_whole_episodes():
    values = np.random.default_rng(0).normal(5.0, 2.0, size=40)
    assert lower_bound(values, seed=1, episodes=range(40)) == lower_bound(values, seed=1)
    repeated = np.repeat(values, 5)  # every episode measured five times over
    episodes = np.repeat(np.arange(40), 5)
    assert lower_bound(repeated, seed=1, episodes=episodes) < lower_bound(repeated, seed=1)
    with pytest.raises(ValueError, match="episode"):
        lower_bound(values, seed=1, episodes=range(39))


UPRIGHT_OVER_PAD = "upright=1,over_pad=1,near_ground=0"


def sample(group=UPRIGHT_OVER_PAD, seed=0):
    return Sample("brake", group, 1.0, (1.0, 0.0), 5, True, (0.0,) * 8, seed, 0)


def test_summarise_rejects_samples_from_different_groups():
    with pytest.raises(ValueError, match="one group"):
        summarise([sample(), sample("upright=0,over_pad=1,near_ground=0")], LUNAR_LANDER.objectives)


def test_an_entry_counts_the_episodes_its_samples_come_from():
    entry = summarise([sample(seed=0), sample(seed=0), sample(seed=3)], LUNAR_LANDER.objectives)
    assert entry["samples"] == 3 and entry["episodes"] == 2


def test_an_entry_needs_enough_episodes_not_just_samples():
    table = build_table(LUNAR_LANDER, range(2), both_plans, min_samples=1)
    assert any(entry["samples"] >= 3 > entry["episodes"] for entry in table.data["plans"])
    assert all(entry["episodes"] <= 2 for entry in table.data["plans"] + table.data["steps"])
    assert build_table(LUNAR_LANDER, range(2), both_plans, min_samples=3).data["plans"] == []


def blobs(count=100, offset=10.0):
    """Two clusters of start states far apart, one state per episode."""
    rng = np.random.default_rng(0)
    near = rng.normal(0.0, 1.0, size=(count, 8))
    return np.vstack([near, near + offset]), np.arange(2 * count)


def test_cells_gather_similar_states():
    starts, episodes = blobs()
    labels = partition(starts, episodes, cell_episodes=100, min_episodes=30, seed=0)
    assert set(labels[:100]) != set(labels[100:]) and len(set(labels[:100])) == len(set(labels[100:])) == 1


def test_every_cell_is_measured_in_enough_episodes():
    starts, episodes = blobs(count=30)
    labels = partition(starts, episodes, cell_episodes=10, min_episodes=30, seed=0)
    assert labels.max() == 1  # six cells were asked for; only two can hold 30 episodes each
    assert all(len(set(episodes[labels == cell])) >= 30 for cell in range(2))


def test_cells_count_episodes_not_samples():
    starts, _ = blobs(count=50)
    episodes = np.repeat(np.arange(10), 10)  # ten states from each of ten episodes
    labels = partition(starts, episodes, cell_episodes=1, min_episodes=6, seed=0)
    assert all(len(set(episodes[labels == cell])) >= 6 for cell in set(labels))


def test_a_group_is_kept_whole_without_a_cell_size_or_without_enough_episodes():
    starts, episodes = blobs()
    assert not partition(starts, episodes, cell_episodes=None, min_episodes=30, seed=0).any()
    assert not partition(starts[:40], episodes[:40], cell_episodes=10, min_episodes=30, seed=0).any()


def test_cells_are_reproducible():
    starts, episodes = blobs()
    first = partition(starts, episodes, cell_episodes=50, min_episodes=30, seed=3)
    np.testing.assert_array_equal(first, partition(starts, episodes, cell_episodes=50, min_episodes=30, seed=3))


def two_rows_of_three():
    """Six tight clusters in two rows of three, 33 episodes each: splitting the rows apart beats splitting the columns."""
    rng = np.random.default_rng(1)
    centres = [(x, y) for y in (-1.0, 1.0) for x in (-1.0, 0.0, 1.0)]
    states = np.vstack([np.array(centre) + rng.normal(0.0, 0.05, size=(33, 2)) for centre in centres])
    return states, np.arange(len(states))


def test_cells_follow_the_states_whatever_the_seed(monkeypatch):
    states, episodes = two_rows_of_three()
    splits = [partition(states, episodes, cell_episodes=99, min_episodes=30, seed=seed) for seed in range(20)]
    for labels in splits:
        np.testing.assert_array_equal(labels, splits[0])
    assert (splits[0][:99] == 0).all() and (splits[0][99:] == 1).all()  # the bottom row, then the top row
    monkeypatch.setattr(effects, "KMEANS_RESTARTS", 1)
    single = [partition(states, episodes, cell_episodes=99, min_episodes=30, seed=seed) for seed in range(20)]
    assert any(not np.array_equal(labels, splits[0]) for labels in single)  # one start alone can split the columns


def test_cells_are_numbered_in_the_order_their_first_states_appear():
    states, episodes = blobs()
    order = np.r_[np.arange(100, 200), np.arange(100)]  # the far cluster first
    labels = partition(states[order], episodes[order], cell_episodes=100, min_episodes=30, seed=0)
    assert (labels[:100] == 0).all() and (labels[100:] == 1).all()


def test_a_few_outlying_states_join_the_nearest_cell_instead_of_keeping_the_group_whole():
    starts, episodes = blobs()
    outliers = np.full((5, 8), 60.0)
    states = np.vstack([starts, outliers])
    labels = partition(states, np.arange(len(states)), cell_episodes=60, min_episodes=30, seed=0)
    assert labels.max() == 1
    assert len(set(labels[:100])) == len(set(labels[100:200])) == 1 and labels[0] != labels[100]
    assert len(set(labels[200:])) == 1 and labels[200] in (labels[0], labels[100])


def test_cells_tell_early_states_from_late_ones_at_the_same_place():
    same_place = np.zeros((200, 8))
    elapsed = np.repeat([150.0, 800.0], 100)
    labels = partition(np.column_stack([same_place, elapsed]), np.arange(200), 100, 30, seed=0)
    assert len(set(labels[:100])) == len(set(labels[100:])) == 1 and labels[0] != labels[100]


def test_a_partition_checks_its_inputs():
    starts, episodes = blobs()
    with pytest.raises(ValueError, match="episode"):
        partition(starts, episodes[:-1], cell_episodes=50, min_episodes=30, seed=0)
    with pytest.raises(ValueError, match="cell_episodes"):
        partition(starts, episodes, cell_episodes=0, min_episodes=30, seed=0)


def test_a_state_is_the_observation_followed_by_the_frames_already_flown():
    np.testing.assert_array_equal(state(np.array([1.0, 2.0]), 7), [1.0, 2.0, 7.0])
    with pytest.raises(ValueError, match="negative"):
        state(np.array([1.0]), -1)


def test_spread_is_one_where_a_number_never_changes():
    starts = np.array([[0.0, 1.0], [2.0, 1.0]])
    np.testing.assert_array_equal(cell_scale(starts), [1.0, 1.0])


# The table


def test_the_table_is_reproducible_from_its_seeds(small_table, celled_table):
    assert build_table(LUNAR_LANDER, range(4), both_plans, 1, 1, cell_episodes=None).data == small_table.data
    assert build_table(LUNAR_LANDER, range(4), both_plans, 1, 1, cell_episodes=1).data == celled_table.data


def test_a_shared_first_step_is_counted_once_per_seed(small_table):
    stabilise = [entry for entry in small_table.data["steps"] if entry["name"] == "stabilise"]
    assert sum(entry["samples"] for entry in stabilise) == 4


def test_the_table_round_trips_through_json(small_table, tmp_path):
    path = tmp_path / "effects.json"
    small_table.save(path)
    assert EffectTable.load(path, LUNAR_LANDER).data == small_table.data


def test_lookup_returns_the_entry_for_the_state(small_table):
    entry = small_table.data["plans"][0]
    estimate = small_table.plan(entry["skills"], episode_start(entry["group"], range(4)), 0)
    assert estimate.samples == entry["samples"] and estimate.episodes == entry["episodes"]
    assert estimate.mean.delta_r == entry["delta_r"]["mean"]
    assert estimate.lower.delta_n == (entry["delta_safety"]["lower"], entry["delta_fuel"]["lower"])
    assert estimate.interaction.delta_r == entry["interaction"]["delta_r"]["mean"]


def test_a_group_kept_whole_answers_for_the_whole_group(small_table):
    entry = small_table.data["plans"][0]
    assert "cells" not in entry
    assert small_table.plan(entry["skills"], episode_start(entry["group"], range(4)), 0).region == entry["group"]


def test_a_lookup_answers_from_the_cell_nearest_the_state(celled_table):
    entry = next(entry for entry in celled_table.data["plans"] if entry.get("cells"))
    assert all(cell["episodes"] >= 1 for cell in entry["cells"])
    for index, cell in enumerate(entry["cells"]):
        *observation, elapsed = cell["centroid"]
        estimate = celled_table.plan(entry["skills"], np.array(observation), elapsed)
        assert estimate.region == f"{entry['group']}|cell={index}"
        assert (estimate.samples, estimate.episodes) == (cell["samples"], cell["episodes"])
        assert estimate.lower.delta_r == cell["delta_r"]["lower"]
        assert estimate.outcomes == cell["outcomes"]


def test_the_cells_of_a_group_hold_all_its_samples(celled_table):
    for entry in celled_table.data["plans"]:
        if entry.get("cells"):
            assert sum(cell["samples"] for cell in entry["cells"]) == entry["samples"]
            assert len(entry["cell_scale"]) == len(entry["cells"][0]["centroid"])


def test_a_state_outside_the_group_support_is_not_measured_in_any_cell(celled_table):
    for entry in celled_table.data["plans"]:
        if not entry.get("cells"):
            continue
        start = episode_start(entry["group"], range(4))
        lower = np.array(start, dtype=np.float64)
        lower[1] -= 0.5  # same facts, but half a unit lower than any measured start
        if group_key(LUNAR_LANDER, lander_facts(lower)) == entry["group"]:
            assert celled_table.plan(entry["skills"], start, 0) is not None
            assert celled_table.plan(entry["skills"], lower, 0) is None
            return
    pytest.fail("no group split into cells keeps its facts half a unit lower")


def first_steps(table):
    """Step entries for stabilise, which starts every plan at the start of an episode."""
    return [entry for entry in table.data["steps"] if entry["name"] == "stabilise"]


def test_steps_have_no_interaction(small_table):
    entry = first_steps(small_table)[0]
    assert small_table.step("stabilise", episode_start(entry["group"], range(4)), 0).interaction is None


def test_unmeasured_or_thin_groups_are_unknown(small_table):
    start = episode_start(small_table.data["plans"][0]["group"], range(4))
    assert small_table.step("hover", start, 0) is None
    strict = EffectTable({**small_table.data, "min_samples": 1000}, LUNAR_LANDER)
    assert strict.plan(CAREFUL, start, 0) is None


def test_the_support_holds_every_measured_start(small_table, env):
    for entry in small_table.data["plans"]:
        for seed in range(4):
            start = replay(env, seed, ())
            if group_key(LUNAR_LANDER, lander_facts(start)) == entry["group"]:
                assert small_table.supports(entry, start, 0)


def test_a_rough_value_follows_the_weights(committed_table):
    skills = ["brake", "touch_down"]
    assert committed_table.rough_value(skills, (1.0, 0.0)) > committed_table.rough_value(skills, (0.0, 1.0))
    assert committed_table.rough_value(["no_such_step"], (0.5, 0.5)) == 0.0


def test_a_rough_value_uses_the_whole_plan_and_adds_up_steps_only_if_it_was_never_measured(committed_table):
    weights = (0.5, 0.5)
    measured = [entry for entry in committed_table.data["plans"] if entry["name"] == "slow_descent>touch_down"]
    assert committed_table.rough_value(["slow_descent", "touch_down"], weights) == pytest.approx(
        committed_table._mean_value(measured, weights)
    )
    never_flown = ["heuristic_pilot", "heuristic_pilot"]
    assert committed_table.rough_value(never_flown, weights) == pytest.approx(
        2 * committed_table.rough_value(["no_such_plan", "heuristic_pilot"], weights)
    )


def test_a_state_outside_the_support_is_not_measured(small_table):
    entry = small_table.data["plans"][0]
    start = episode_start(entry["group"], range(4))
    lower = np.array(start, dtype=np.float64)
    lower[1] -= 0.5  # same facts, but half a unit lower than any measured start
    assert group_key(LUNAR_LANDER, lander_facts(lower)) == entry["group"]
    assert small_table.plan(entry["skills"], start, 0) is not None
    assert small_table.plan(entry["skills"], lower, 0) is None


def test_a_state_later_in_the_episode_than_any_measured_is_not_measured(small_table):
    entry = small_table.data["plans"][0]  # measured from episode starts only
    start = episode_start(entry["group"], range(4))
    assert small_table.plan(entry["skills"], start, 0) is not None
    assert small_table.plan(entry["skills"], start, 400) is None


def test_the_table_checks_its_format(small_table):
    with pytest.raises(ValueError, match="schema_version"):
        EffectTable({**small_table.data, "schema_version": "other"}, LUNAR_LANDER)
    with pytest.raises(ValueError, match="measures 'other'"):
        EffectTable({**small_table.data, "environment": "other"}, LUNAR_LANDER)
    with pytest.raises(ValueError, match="groups by"):
        EffectTable({**small_table.data, "group_facts": ["upright"]}, LUNAR_LANDER)
    with pytest.raises(ValueError, match="objectives"):
        EffectTable({**small_table.data, "objectives": ["safety"]}, LUNAR_LANDER)
    with pytest.raises(ValueError, match="discounts by"):
        EffectTable({**small_table.data, "gamma": 0.5}, LUNAR_LANDER)
    with pytest.raises(KeyError):
        EffectTable({key: value for key, value in small_table.data.items() if key != "support_margin"}, LUNAR_LANDER)


# The committed table


def test_committed_table_covers_every_plan_from_unseen_episode_starts(committed_table, env):
    plans = plans_for(load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, LUNAR_LANDER.skills))
    unseen = range(committed_table.data["seeds"]["stop"], committed_table.data["seeds"]["stop"] + 30)
    for seed in unseen:
        start = replay(env, seed, ())
        for skills in plans(lander_facts(start)):
            assert committed_table.plan(skills, start, 0) is not None, (seed, skills)


def test_committed_table_answers_from_states_reached_in_mid_flight(committed_table, env):
    unseen = committed_table.data["seeds"]["stop"]
    plans = plans_for(load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, LUNAR_LANDER.skills))
    answered = 0
    for seed in range(unseen, unseen + 30):
        flight = fly_plan(LUNAR_LANDER, env, seed, CAREFUL)
        for prefix in flight.next_states[:-1]:
            reached = replay(env, seed, prefix)
            measured = [committed_table.plan(skills, reached, len(prefix)) for skills in plans(lander_facts(reached))]
            answered += any(estimate is not None for estimate in measured)
    assert answered > 30


def test_committed_table_refuses_a_state_no_flight_reached(committed_table):
    entry = committed_table.data["plans"][0]
    *below, _ = np.asarray(entry["support"]["low"], dtype=np.float64) - 1.0
    assert committed_table.plan(entry["skills"], np.array(below), 0) is None


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
    estimate = small_table.plan(entry["skills"], episode_start(entry["group"], range(4)), 0)
    assert estimate.outcomes == entry["outcomes"]
    step = first_steps(small_table)[0]
    assert small_table.step("stabilise", episode_start(step["group"], range(4)), 0).outcomes is None

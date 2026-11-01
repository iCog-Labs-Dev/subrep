"""Tests for the hierarchical goal network planner (decomposition/planner.py)."""

import ast
import json
import random
from pathlib import Path

import pytest

import decomposition.planner as planner_module
import decomposition.recipe_book as recipe_book_module
from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable, replay
from decomposition.planner import ALREADY_TRUE, DIRECT, SEARCH, TO_ACHIEVE, generate, goal_graph, plans_for, propose
from decomposition.recipe_book import LUNAR_LANDER_BOOK, SCHEMA_VERSION, holds, load_recipe_book, parse_recipe_book
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import FACT_NAMES, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import SUBSKILLS

START = {
    "upright": False,
    "over_pad": False,
    "near_ground": False,
    "slow": False,
    "falling_fast": True,
    "landed": False,
    "descent_underway": True,
}


def toy_book(methods=None, finish_requires=None, finish_effects=None, terminal=False):
    """Goal "g" is reached by getting ready (warm or cool) and then finishing."""
    return parse_recipe_book(
        {
            "schema_version": SCHEMA_VERSION,
            "environment": "test",
            "goals": {"g": {"achieved_when": {"done": True}, "terminal": terminal}},
            "methods": [{"name": "prepare_first", "subgoals": [{"ready": True}, {"done": True}]}]
            if methods is None
            else methods,
            "skills": {
                "warm": {"requires": {}, "effects": {"ready": True}, "may_change": []},
                "cool": {"requires": {}, "effects": {"ready": True, "cold": True}, "may_change": []},
                "finish": {
                    "requires": finish_requires if finish_requires is not None else {"ready": True},
                    "effects": finish_effects if finish_effects is not None else {"done": True},
                    "may_change": [],
                },
            },
        },
        fact_names=("ready", "done", "cold"),
        skill_names=("warm", "cool", "finish"),
    )


@pytest.fixture(scope="module")
def lunar_book():
    return load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, SUBSKILLS)


# Working through subgoals


def test_every_skill_that_reaches_a_subgoal_gives_a_candidate():
    candidates = propose(toy_book(), "g", {})
    assert [(c.option, c.skills) for c in candidates] == [
        ("prepare_first", ("warm", "finish")),
        ("prepare_first", ("cool", "finish")),
    ]
    assert candidates[0].methods == ("prepare_first",)
    assert candidates[1].predicted_facts == {"ready": True, "cold": True, "done": True}


def test_a_subgoal_that_already_holds_is_skipped():
    [candidate] = propose(toy_book(), "g", {"ready": True})
    assert candidate.skills == ("finish",)
    assert candidate.subgoals[0].status == ALREADY_TRUE and candidate.subgoals[0].skills == ()
    assert candidate.subgoals[1].status == TO_ACHIEVE and candidate.subgoals[1].skills == ("finish",)


def test_a_method_applies_only_when_its_requirements_are_known():
    methods = [{"name": "m", "requires": {"cold": False}, "subgoals": [{"ready": True}, {"done": True}]}]
    assert [c.option for c in propose(toy_book(methods), "g", {"cold": False})] == ["m", "m"]
    assert propose(toy_book(methods, finish_requires={"cold": True}), "g", {"cold": True}) != []
    assert "m" not in [c.option for c in propose(toy_book(methods), "g", {"cold": True})]
    assert "m" not in [c.option for c in propose(toy_book(methods), "g", {})]


def test_a_method_for_another_subgoal_is_not_offered():
    methods = [{"name": "other", "subgoals": [{"cold": True}]}]
    assert "other" not in [c.option for c in propose(toy_book(methods), "g", {"ready": True})]


def test_a_skill_applies_directly_without_a_method():
    [candidate] = propose(toy_book(methods=[]), "g", {"ready": True})
    assert candidate.option == DIRECT and candidate.skills == ("finish",)
    assert not candidate.searched


def test_a_skill_whose_requirements_are_not_met_is_reached_through_the_search():
    candidates = propose(toy_book(finish_requires={"cold": True}), "g", {})
    assert ("cool", "finish") in [c.skills for c in candidates]
    assert all(c.skills[-1] == "finish" for c in candidates)


def test_a_skill_that_does_not_reach_the_subgoal_is_not_chosen():
    assert propose(toy_book(finish_effects={}), "g", {"done": False}) == []


def test_methods_that_refer_to_themselves_end():
    methods = [{"name": "loop", "subgoals": [{"done": True}]}, {"name": "ring", "subgoals": [{"ready": True}, {"done": True}]}]
    assert propose(toy_book(methods), "g", {"ready": True})


def test_an_achieved_goal_needs_no_plan():
    assert propose(toy_book(), "g", {"done": True}) == []


def test_a_terminal_goal_is_never_achieved_while_planning():
    [candidate] = propose(toy_book(terminal=True), "g", {"done": True, "ready": True})
    assert candidate.skills == ("finish",)


def test_unknown_goals_are_rejected():
    with pytest.raises(ValueError, match="unknown goal"):
        propose(toy_book(), "fly", {})


def test_the_planner_does_not_depend_on_any_environment():
    for module in (planner_module, recipe_book_module):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        roots = {name.split(".")[0] for name in imported}
        assert not roots & {"env", "demo", "gymnasium", "mo_gymnasium", "numpy"}, module.__name__


# Generating candidates under priorities


def test_without_a_limit_every_candidate_is_kept_in_the_books_order():
    generation = generate(toy_book(), ("g",), {})
    assert [c.skills for c in generation.candidates] == [c.skills for c in propose(toy_book(), "g", {})]
    assert generation.skipped == ()


def test_a_limit_keeps_the_most_promising_plans_and_records_the_rest():
    values = {("warm", "finish"): 1.0, ("cool", "finish"): 2.0}
    generation = generate(toy_book(), ("g",), {}, rough_value=values.get, limit=1)
    assert [c.skills for c in generation.candidates] == [("cool", "finish")]
    assert [c.skills for c in generation.skipped] == [("warm", "finish")]


def test_priorities_change_which_plans_a_limit_keeps():
    weights_to_values = {
        "warm_first": {("warm", "finish"): 2.0, ("cool", "finish"): 1.0},
        "cool_first": {("warm", "finish"): 1.0, ("cool", "finish"): 2.0},
    }
    kept = {
        name: generate(toy_book(), ("g",), {}, rough_value=values.get, limit=1).candidates[0].skills[0]
        for name, values in weights_to_values.items()
    }
    assert kept == {"warm_first": "warm", "cool_first": "cool"}


def test_a_limit_above_the_number_of_plans_changes_nothing():
    assert generate(toy_book(), ("g",), {}, limit=10) == generate(toy_book(), ("g",), {})


def test_a_limit_is_checked_and_needs_a_rough_value():
    with pytest.raises(ValueError, match="at least 1"):
        generate(toy_book(), ("g",), {}, limit=0)
    with pytest.raises(ValueError, match="rough value"):
        generate(toy_book(), ("g",), {}, limit=1)


# The search that fills gaps


def test_the_search_finds_a_plan_when_the_book_has_no_methods():
    candidates = propose(toy_book(methods=[]), "g", {})
    assert {c.skills for c in candidates} == {("warm", "finish"), ("cool", "finish")}
    assert all(c.option == SEARCH and c.searched for c in candidates)


def test_the_search_fills_a_gap_inside_a_method():
    methods = [{"name": "only_finish", "subgoals": [{"done": True}]}]
    [first, *_] = propose(toy_book(methods), "g", {})
    assert first.option == "only_finish" and first.searched
    assert first.skills[-1] == "finish"


def test_the_search_gives_up_when_no_skill_chain_reaches_the_subgoal():
    assert propose(toy_book(methods=[], finish_effects={"cold": True}), "g", {}) == []


def test_the_search_lands_with_no_recipes_at_all(lunar_book):
    data = json.loads(LUNAR_LANDER_BOOK.read_text(encoding="utf-8"))
    data["methods"] = []
    no_recipes = parse_recipe_book(data, FACT_NAMES, SUBSKILLS)
    candidates = propose(no_recipes, "land_on_pad", {**START, "over_pad": True})
    assert candidates and all(c.searched and c.skills[-1] == "touch_down" for c in candidates)


# The LunarLander book


def test_landing_from_the_air_offers_every_recipe(lunar_book):
    candidates = propose(lunar_book, "land_on_pad", START)
    assert [(c.option, c.skills) for c in candidates] == [
        ("careful_landing", ("stabilise", "move_over_pad", "slow_descent", "touch_down")),
        ("fuel_saving_landing", ("stabilise", "move_over_pad", "free_fall", "slow_descent", "touch_down")),
        ("fuel_saving_landing", ("stabilise", "move_over_pad", "free_fall", "brake", "touch_down")),
        ("direct_descent", ("slow_descent", "touch_down")),
    ]
    assert all(c.predicted_facts["landed"] for c in candidates)


def test_subgoals_that_already_hold_drop_their_skills(lunar_book):
    facts = {**START, "upright": True, "over_pad": True, "falling_fast": False, "descent_underway": False}
    candidates = propose(lunar_book, "land_on_pad", facts)
    assert ("slow_descent", "touch_down") in [c.skills for c in candidates]
    assert all("stabilise" not in c.skills and "move_over_pad" not in c.skills for c in candidates)


def test_the_book_no_longer_forbids_descending_before_reaching_the_pad(lunar_book):
    assert holds(lunar_book.skills["slow_descent"].requires, {"over_pad": False})


def test_free_fall_promises_a_condition_the_planner_can_use(lunar_book):
    assert lunar_book.skills["free_fall"].effects == {"descent_underway": True}


@pytest.mark.parametrize("near_ground", [True, None])
def test_fuel_saving_needs_known_height_to_fall(lunar_book, near_ground):
    facts = dict(START)
    if near_ground is None:
        del facts["near_ground"]
    else:
        facts["near_ground"] = near_ground
    assert "fuel_saving_landing" not in [c.option for c in propose(lunar_book, "land_on_pad", facts)]


def test_recovering_stability(lunar_book):
    [candidate] = propose(lunar_book, "recover_stability", START)
    assert candidate.option == DIRECT
    assert candidate.skills == ("stabilise",)
    assert propose(lunar_book, "recover_stability", {**START, "upright": True}) == []


def test_the_goal_graph_shows_goal_subgoals_and_skills(lunar_book):
    facts = {**START, "upright": True}
    candidate = propose(lunar_book, "land_on_pad", facts)[0]
    graph = json.loads(json.dumps(goal_graph(candidate)))
    assert graph["goal"] == "land_on_pad" and graph["option"] == "careful_landing"
    assert [node["conditions"] for node in graph["nodes"]] == [
        {"upright": True},
        {"over_pad": True},
        {"near_ground": True, "slow": True},
        {"landed": True, "over_pad": True},
    ]
    assert [node["status"] for node in graph["nodes"]] == [ALREADY_TRUE, TO_ACHIEVE, TO_ACHIEVE, TO_ACHIEVE]
    assert [node["skills"] for node in graph["nodes"]] == [[], ["move_over_pad"], ["slow_descent"], ["touch_down"]]
    assert graph["edges"] == [[0, 1], [1, 2], [2, 3]]


def test_plans_for_lists_each_distinct_plan_once(lunar_book):
    plans = plans_for(lunar_book)(START)
    assert len(plans) == len(set(plans))
    assert ("stabilise",) in plans and ("slow_descent", "touch_down") in plans


def test_random_facts_give_sound_deterministic_plans(lunar_book):
    rng = random.Random(0)
    for _ in range(500):
        facts = {name: rng.choice([True, False]) for name in FACT_NAMES if rng.random() < 0.8}
        for goal in lunar_book.goals.values():
            candidates = propose(lunar_book, goal.name, facts)
            assert candidates == propose(lunar_book, goal.name, facts)
            if not goal.terminal and holds(goal.achieved_when, facts):
                assert candidates == []
            assert len({c.skills for c in candidates}) == len(candidates)
            for candidate in candidates:
                known = dict(facts)
                for skill in candidate.skills:
                    assert holds(lunar_book.skills[skill].requires, known), (facts, candidate.skills, skill)
                    known = lunar_book.skills[skill].apply(known)
                assert known == candidate.predicted_facts
                assert all(known.get(name, value) == value for name, value in goal.achieved_when.items())


def test_every_plan_proposed_at_an_episode_start_has_measured_effects(lunar_book):
    table = EffectTable.load(DEFAULT_TABLE_PATH, LUNAR_LANDER)
    env = SubRepEnv(seed=0)
    try:
        for seed in range(table.data["seeds"]["stop"], table.data["seeds"]["stop"] + 30):
            start = replay(env, seed, ())
            for goal in lunar_book.goals:
                for candidate in propose(lunar_book, goal, lander_facts(start)):
                    entries = [e for e in table.data["plans"] if tuple(e["skills"]) == candidate.skills]
                    assert entries, (seed, candidate.skills)
    finally:
        env.close()

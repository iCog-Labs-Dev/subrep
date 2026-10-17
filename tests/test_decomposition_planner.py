"""Tests for the HTN planner (decomposition/planner.py)."""

import ast
import random
from pathlib import Path

import pytest

import decomposition.planner as planner_module
import decomposition.recipe_book as recipe_book_module
from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable, replay
from decomposition.planner import propose
from decomposition.recipe_book import LUNAR_LANDER_BOOK, SCHEMA_VERSION, holds, load_recipe_book, parse_recipe_book
from demo.build_lunar_effect_table import PLANS
from env.lunar_facts import FACT_NAMES, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import SUBSKILLS

START = {"upright": False, "over_pad": False, "near_ground": False, "slow": False, "falling_fast": True, "landed": False}


def toy_book(finish_requires=None, finish_effects=None):
    """Goal "g" is reached by preparing (warm or cool) then finishing, or by finishing when ready."""
    return parse_recipe_book(
        {
            "schema_version": SCHEMA_VERSION,
            "environment": "test",
            "goals": {"g": {"task": "root", "achieved_when": {"done": True}}},
            "tasks": {
                "root": {
                    "methods": [
                        {"name": "prepare_first", "subtasks": ["prepare", "finish"]},
                        {"name": "finish_now", "requires": {"ready": True}, "subtasks": ["finish"]},
                    ]
                },
                "prepare": {
                    "methods": [
                        {"name": "warm_up", "subtasks": ["warm"]},
                        {"name": "cool_down", "subtasks": ["cool"]},
                    ]
                },
            },
            "primitives": {
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


# Decomposition rules


def test_every_choice_of_methods_is_a_candidate():
    candidates = propose(toy_book(), "g", {})
    assert [(c.option, c.skills) for c in candidates] == [
        ("prepare_first+warm_up", ("warm", "finish")),
        ("prepare_first+cool_down", ("cool", "finish")),
    ]
    assert candidates[0].methods == (("root", "prepare_first"), ("prepare", "warm_up"))
    assert candidates[1].predicted_facts == {"ready": True, "cold": True, "done": True}


def test_a_method_applies_only_when_its_requirements_are_known():
    options = [c.option for c in propose(toy_book(), "g", {"ready": True})]
    assert options == ["prepare_first+warm_up", "prepare_first+cool_down", "finish_now"]
    assert "finish_now" not in [c.option for c in propose(toy_book(), "g", {"ready": False})]


def test_a_primitive_whose_requirements_cannot_be_met_prunes_the_branch():
    candidates = propose(toy_book(finish_requires={"cold": True}), "g", {})
    assert [c.skills for c in candidates] == [("cool", "finish")]


def test_plans_that_do_not_achieve_the_goal_are_dropped():
    assert propose(toy_book(finish_effects={}), "g", {}) == []


def test_an_achieved_goal_needs_no_plan():
    assert propose(toy_book(), "g", {"done": True}) == []


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


# The LunarLander book


def test_landing_from_the_air_offers_a_careful_and_a_fuel_saving_plan(lunar_book):
    candidates = propose(lunar_book, "land_on_pad", START)
    assert [c.option for c in candidates] == ["careful_landing", "fuel_saving_landing"]
    assert [list(c.skills) for c in candidates] == PLANS[:2]
    assert all(c.predicted_facts["landed"] for c in candidates)


@pytest.mark.parametrize("near_ground", [True, None])
def test_fuel_saving_needs_known_height_to_fall(lunar_book, near_ground):
    facts = dict(START)
    if near_ground is None:
        del facts["near_ground"]
    else:
        facts["near_ground"] = near_ground
    assert [c.option for c in propose(lunar_book, "land_on_pad", facts)] == ["careful_landing"]


def test_recovering_stability(lunar_book):
    [candidate] = propose(lunar_book, "recover_stability", START)
    assert candidate.option == "stabilise_in_place"
    assert candidate.skills == ("stabilise",)
    assert propose(lunar_book, "recover_stability", {**START, "upright": True}) == []


def test_random_facts_give_sound_deterministic_plans(lunar_book):
    rng = random.Random(0)
    measured = {tuple(plan) for plan in PLANS}
    for _ in range(500):
        facts = {name: rng.choice([True, False]) for name in FACT_NAMES if rng.random() < 0.8}
        for goal in lunar_book.goals.values():
            candidates = propose(lunar_book, goal.name, facts)
            assert candidates == propose(lunar_book, goal.name, facts)
            if holds(goal.achieved_when, facts):
                assert candidates == []
            assert len({c.option for c in candidates}) == len(candidates)
            for candidate in candidates:
                assert candidate.skills in measured
                known = dict(facts)
                for skill in candidate.skills:
                    assert holds(lunar_book.primitives[skill].requires, known), (facts, candidate.skills, skill)
                    known = lunar_book.primitives[skill].apply(known)
                assert known == candidate.predicted_facts
                assert holds(goal.achieved_when, known)


def test_every_plan_proposed_at_an_episode_start_has_measured_effects(lunar_book):
    table = EffectTable.load(DEFAULT_TABLE_PATH)
    env = SubRepEnv(seed=0)
    try:
        for seed in range(table.data["seeds"]["stop"], table.data["seeds"]["stop"] + 30):
            facts = lander_facts(replay(env, seed, ()))
            for goal in lunar_book.goals:
                for candidate in propose(lunar_book, goal, facts):
                    assert table.plan(candidate.skills, facts) is not None, (seed, candidate.skills)
    finally:
        env.close()

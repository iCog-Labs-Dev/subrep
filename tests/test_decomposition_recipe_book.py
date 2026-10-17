"""Tests for recipe books (decomposition/recipe_book.py) and the LunarLander book."""

import pytest

from decomposition.recipe_book import (
    LUNAR_LANDER_BOOK,
    SCHEMA_VERSION,
    Primitive,
    RecipeBookError,
    holds,
    load_recipe_book,
    parse_recipe_book,
)
from demo.build_lunar_effect_table import PLANS
from env.lunar_facts import FACT_NAMES, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import SUBSKILLS, run_subskill

# The book's declarations were read off seeds 0-299; these seeds check them.
HELD_OUT_SEEDS = range(300, 400)

FACTS = ("there", "tired")
SKILLS = ("step",)


def minimal_book():
    return {
        "schema_version": SCHEMA_VERSION,
        "environment": "test",
        "goals": {"reach": {"task": "go", "achieved_when": {"there": True}}},
        "tasks": {"go": {"methods": [{"name": "walk", "subtasks": ["step"]}]}},
        "primitives": {"step": {"requires": {}, "effects": {"there": True}, "may_change": ["tired"]}},
    }


@pytest.fixture(scope="module")
def lunar_book():
    return load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, SUBSKILLS)


@pytest.fixture(scope="module")
def flown_steps():
    """(sub-skill, facts before, facts after, completed) for every step the plans reach."""
    steps = []
    env = SubRepEnv(seed=HELD_OUT_SEEDS.start)
    try:
        for seed in HELD_OUT_SEEDS:
            for plan in PLANS:
                obs, _ = env.reset(seed=seed)
                for name in plan:
                    before = lander_facts(obs)
                    run = run_subskill(env, SUBSKILLS[name], obs)
                    obs = run.observation
                    steps.append((name, before, lander_facts(obs), run.completed))
                    if run.terminated or run.truncated or not run.completed:
                        break
    finally:
        env.close()
    return steps


# Facts and primitives


def test_holds_treats_missing_facts_as_unknown():
    assert holds({"upright": True}, {"upright": True, "slow": False})
    assert not holds({"upright": True}, {"upright": False})
    assert not holds({"upright": True}, {})
    assert holds({}, {})


def test_a_primitive_sets_its_effects_forgets_what_it_may_change_and_keeps_the_rest():
    primitive = Primitive("brake", requires={}, effects={"slow": True}, may_change=("upright",))
    after = primitive.apply({"slow": False, "upright": True, "over_pad": True})
    assert after == {"slow": True, "over_pad": True}


# Checking a book


def test_a_minimal_book_loads():
    book = parse_recipe_book(minimal_book(), FACTS, SKILLS)
    assert book.goals["reach"].task == "go"
    assert book.tasks["go"].methods[0].subtasks == ("step",)
    assert book.tasks["go"].methods[0].requires == {}
    assert book.primitives["step"].may_change == ("tired",)


def _set(path, value):
    def change(book):
        target = book
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


@pytest.mark.parametrize(
    "change, message",
    [
        (_set(["schema_version"], "other"), "schema_version"),
        (_set(["primitives", "step", "requires"], {"hungry": True}), "unknown facts"),
        (_set(["primitives", "step", "effects"], {"there": 1}), "true or false"),
        (_set(["primitives", "step", "may_change"], ["there"]), "both an effect"),
        (_set(["primitives", "step", "require"], {}), "unexpected keys"),
        (_set(["primitives", "jump"], {"requires": {}, "effects": {}, "may_change": []}), "not sub-skills"),
        (_set(["tasks", "go", "methods", 0, "subtasks"], ["fly"]), "unknown subtasks"),
        (_set(["tasks", "go", "methods", 0, "subtasks"], ["go"]), "cycle"),
        (_set(["tasks", "go", "methods"], []), "at least one method"),
        (lambda book: book["tasks"]["go"]["methods"].append({"name": "walk", "subtasks": []}), "same name"),
        (_set(["tasks", "step"], {"methods": [{"name": "m", "subtasks": []}]}), "both a task and a primitive"),
        (_set(["goals", "reach", "task"], "step"), "unknown task"),
        (_set(["goals", "reach", "achieved_when"], {}), "at least one fact"),
        (lambda book: book["goals"]["reach"].pop("task"), "missing"),
    ],
)
def test_malformed_books_are_rejected(change, message):
    book = minimal_book()
    change(book)
    with pytest.raises(RecipeBookError, match=message):
        parse_recipe_book(book, FACTS, SKILLS)


def test_a_cycle_through_several_tasks_is_rejected():
    book = minimal_book()
    book["tasks"]["go"]["methods"][0]["subtasks"] = ["prepare"]
    book["tasks"]["prepare"] = {"methods": [{"name": "again", "subtasks": ["go"]}]}
    with pytest.raises(RecipeBookError, match="cycle: go -> prepare -> go"):
        parse_recipe_book(book, FACTS, SKILLS)


# The LunarLander book


def test_lunar_book_covers_every_sub_skill(lunar_book):
    assert set(lunar_book.primitives) == set(SUBSKILLS)
    assert set(lunar_book.goals) == {"land_on_pad", "recover_stability"}


def test_lunar_book_predicts_the_facts_after_every_completed_step(lunar_book, flown_steps):
    completed = [(name, before, after) for name, before, after, done in flown_steps if done]
    assert completed
    for name, before, after in completed:
        predicted = lunar_book.primitives[name].apply(before)
        assert holds(predicted, after), (name, before, after)


def test_lunar_book_requirements_hold_whenever_a_plan_reaches_a_step(lunar_book, flown_steps):
    for name, before, _, _ in flown_steps:
        assert holds(lunar_book.primitives[name].requires, before), (name, before)

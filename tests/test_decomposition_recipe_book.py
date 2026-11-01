"""Tests for recipe books (decomposition/recipe_book.py) and the LunarLander book."""

import itertools

import pytest

from decomposition.recipe_book import (
    LUNAR_LANDER_BOOK,
    SCHEMA_VERSION,
    RecipeBookError,
    Skill,
    holds,
    load_recipe_book,
    parse_recipe_book,
)
from demo.run_lunar_subskill_reliability import SEQUENCES
from env.lunar_facts import FACT_NAMES, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import SUBSKILLS, run_subskill

# The book's declarations were read off seeds 0-299; these seeds check them.
HELD_OUT_SEEDS = range(300, 400)

FACTS = ("there", "tired")
SKILLS = ("step",)
PLANS = [*SEQUENCES.values(), ["stabilise"], ["slow_descent", "touch_down"]]


def minimal_book():
    return {
        "schema_version": SCHEMA_VERSION,
        "environment": "test",
        "goals": {"reach": {"achieved_when": {"there": True}}},
        "methods": [{"name": "walk", "subgoals": [{"there": True}]}],
        "skills": {"step": {"requires": {}, "effects": {"there": True}, "may_change": ["tired"]}},
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


# Facts and skills


def test_holds_treats_missing_facts_as_unknown():
    assert holds({"upright": True}, {"upright": True, "slow": False})
    assert not holds({"upright": True}, {"upright": False})
    assert not holds({"upright": True}, {})
    assert holds({}, {})


def test_a_skill_sets_its_effects_forgets_what_it_may_change_and_keeps_the_rest():
    skill = Skill("brake", requires={}, effects={"slow": True}, may_change=("upright",))
    after = skill.apply({"slow": False, "upright": True, "over_pad": True})
    assert after == {"slow": True, "over_pad": True}


# Checking a book


def test_a_minimal_book_loads():
    book = parse_recipe_book(minimal_book(), FACTS, SKILLS)
    assert book.goals["reach"].achieved_when == {"there": True}
    assert book.methods[0].subgoals == ({"there": True},)
    assert book.methods[0].requires == {}
    assert book.skills["step"].may_change == ("tired",)
    assert book.goals["reach"].terminal is False


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
        (_set(["skills", "step", "requires"], {"hungry": True}), "unknown facts"),
        (_set(["skills", "step", "effects"], {"there": 1}), "true or false"),
        (_set(["skills", "step", "may_change"], ["there"]), "both an effect"),
        (_set(["skills", "step", "require"], {}), "unexpected keys"),
        (_set(["skills", "jump"], {"requires": {}, "effects": {}, "may_change": []}), "not skills"),
        (_set(["methods"], {}), "must be a list"),
        (_set(["methods", 0, "subgoals"], []), "at least one subgoal"),
        (_set(["methods", 0, "subgoals"], [{}]), "no conditions"),
        (_set(["methods", 0, "subgoals"], [{"hungry": True}]), "unknown facts"),
        (_set(["methods", 0, "requires"], {"hungry": True}), "unknown facts"),
        (lambda book: book["methods"].append({"name": "walk", "subgoals": [{"tired": True}]}), "different names"),
        (_set(["goals", "reach", "achieved_when"], {}), "at least one fact"),
        (lambda book: book["goals"]["reach"].pop("achieved_when"), "missing"),
        (_set(["goals", "reach", "terminal"], "yes"), "terminal must be true or false"),
    ],
)
def test_malformed_books_are_rejected(change, message):
    book = minimal_book()
    change(book)
    with pytest.raises(RecipeBookError, match=message):
        parse_recipe_book(book, FACTS, SKILLS)


# The LunarLander book


def test_lunar_book_covers_every_sub_skill(lunar_book):
    assert set(lunar_book.skills) == set(SUBSKILLS)
    assert set(lunar_book.goals) == {"land_on_pad", "recover_stability"}


def test_landing_on_the_pad_is_a_terminal_goal_that_needs_the_pad(lunar_book):
    goal = lunar_book.goals["land_on_pad"]
    assert goal.terminal
    assert goal.achieved_when == {"landed": True, "over_pad": True}
    assert not lunar_book.goals["recover_stability"].terminal


def test_lunar_book_predicts_the_facts_after_every_completed_step(lunar_book, flown_steps):
    completed = [(name, before, after) for name, before, after, done in flown_steps if done]
    assert completed
    for name, before, after in completed:
        predicted = lunar_book.skills[name].apply(before)
        assert holds(predicted, after), (name, before, after)


def test_lunar_book_requirements_hold_whenever_a_plan_reaches_a_step(lunar_book, flown_steps):
    for name, before, _, _ in flown_steps:
        assert holds(lunar_book.skills[name].requires, before), (name, before)


def possible_facts():
    """Every combination of facts except those the fact definitions rule out.

    ``slow`` (|vy| <= 0.2) and ``falling_fast`` (vy <= -0.4) cannot both hold,
    and ``descent_underway`` means falling fast or near the ground.
    """
    for values in itertools.product((False, True), repeat=len(FACT_NAMES)):
        facts = dict(zip(FACT_NAMES, values))
        consistent = facts["descent_underway"] == (facts["falling_fast"] or facts["near_ground"])
        if consistent and not (facts["slow"] and facts["falling_fast"]):
            yield facts


def test_lunar_book_effects_follow_from_how_each_sub_skill_stops(lunar_book):
    for name, skill in lunar_book.skills.items():
        stops = [facts for facts in possible_facts() if SUBSKILLS[name].is_done(facts)]
        assert stops, name
        for facts in stops:
            assert holds(skill.effects, facts), (name, facts)


def test_free_fall_may_stop_near_the_ground_and_brake_still_applies(lunar_book):
    stopped_near_ground = {
        **dict.fromkeys(FACT_NAMES, False),
        "near_ground": True,
        "over_pad": True,
        "descent_underway": True,
    }
    assert SUBSKILLS["free_fall"].is_done(stopped_near_ground)
    after = lunar_book.skills["free_fall"].apply(stopped_near_ground)
    assert holds(lunar_book.skills["brake"].requires, after)

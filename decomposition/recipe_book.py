"""Recipe books: goals, the subgoals that lead to them, and what each skill needs and does.

A recipe book is a versioned JSON file in the style of a hierarchical goal
network (HGN):

- a goal is a set of conditions on facts. A terminal goal is achieved only by how
  the episode ends, so its conditions are confirmed on the final result rather
  than mid-episode;
- a method is one way to reach a subgoal: an ordered list of subgoals, each a set
  of conditions, and the facts that must hold before it is chosen. Its last
  subgoal is what it achieves;
- a skill has the facts it requires, the facts it makes true or false
  (``effects``), and the facts it may change unpredictably (``may_change``).
  Facts a skill does not mention keep their value.

The book is checked against the fact and skill names the caller provides, so this
module does not depend on any environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Collection, Dict, Mapping, Set, Tuple, Union

SCHEMA_VERSION = "subrep.recipe_book.v2"
LUNAR_LANDER_BOOK = Path(__file__).resolve().parent / "data" / "lunar_recipe_book.json"

Facts = Mapping[str, bool]


class RecipeBookError(ValueError):
    """The recipe book is malformed or does not match the environment's names."""


def holds(conditions: Facts, facts: Facts) -> bool:
    """Whether every condition is known to hold in ``facts``; a missing fact is unknown."""
    return all(name in facts and facts[name] == value for name, value in conditions.items())


@dataclass(frozen=True)
class Skill:
    """A skill as the planner sees it: what it requires, what it achieves and what it may disturb."""

    name: str
    requires: Dict[str, bool]
    effects: Dict[str, bool]
    may_change: Tuple[str, ...]

    def apply(self, facts: Facts) -> Dict[str, bool]:
        """The facts known after this skill completes from a state with ``facts``."""
        after = {name: value for name, value in facts.items() if name not in self.may_change}
        after.update(self.effects)
        return after


@dataclass(frozen=True)
class Method:
    """One way to reach a subgoal: ordered subgoals and when the way applies."""

    name: str
    subgoals: Tuple[Dict[str, bool], ...]
    requires: Dict[str, bool]


@dataclass(frozen=True)
class Goal:
    """A goal: the facts that show it is achieved."""

    name: str
    achieved_when: Dict[str, bool]
    terminal: bool = False  # achieved only by how the episode ends


@dataclass(frozen=True)
class RecipeBook:
    """A checked recipe book."""

    environment: str
    goals: Dict[str, Goal]
    methods: Tuple[Method, ...]
    skills: Dict[str, Skill]


def load_recipe_book(
    path: Union[str, Path], fact_names: Collection[str], skill_names: Collection[str]
) -> RecipeBook:
    """Load and check a recipe book against the environment's fact and skill names."""
    return parse_recipe_book(json.loads(Path(path).read_text(encoding="utf-8")), fact_names, skill_names)


def parse_recipe_book(data: Mapping, fact_names: Collection[str], skill_names: Collection[str]) -> RecipeBook:
    """Check a decoded recipe book and build it."""
    _check_keys(data, "recipe book", required={"schema_version", "environment", "goals", "methods", "skills"})
    if data["schema_version"] != SCHEMA_VERSION:
        raise RecipeBookError(f"expected schema_version {SCHEMA_VERSION!r}, got {data['schema_version']!r}")
    facts = set(fact_names)

    skills = {name: _skill(name, spec, facts) for name, spec in data["skills"].items()}
    not_skills = sorted(set(skills) - set(skill_names))
    if not_skills:
        raise RecipeBookError(f"skills {not_skills} are not skills of the environment")

    if not isinstance(data["methods"], list):
        raise RecipeBookError("methods must be a list")
    methods = tuple(_method(spec, facts) for spec in data["methods"])
    names = [method.name for method in methods]
    if len(set(names)) != len(names):
        raise RecipeBookError("methods must have different names")

    goals = {name: _goal(name, spec, facts) for name, spec in data["goals"].items()}
    return RecipeBook(environment=str(data["environment"]), goals=goals, methods=methods, skills=skills)


def _check_keys(
    spec: Mapping, where: str, required: AbstractSet[str], optional: AbstractSet[str] = frozenset()
) -> None:
    if not isinstance(spec, Mapping):
        raise RecipeBookError(f"{where} must be an object")
    missing = sorted(required - set(spec))
    unexpected = sorted(set(spec) - required - set(optional))
    if missing:
        raise RecipeBookError(f"{where} is missing {missing}")
    if unexpected:
        raise RecipeBookError(f"{where} has unexpected keys {unexpected}")


def _fact_names(names: object, facts: Set[str], where: str) -> Tuple[str, ...]:
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise RecipeBookError(f"{where} must be a list of fact names")
    unknown = sorted(set(names) - facts)
    if unknown:
        raise RecipeBookError(f"{where} uses unknown facts {unknown}")
    return tuple(names)


def _conditions(spec: object, facts: Set[str], where: str) -> Dict[str, bool]:
    if not isinstance(spec, Mapping):
        raise RecipeBookError(f"{where} must map fact names to true or false")
    _fact_names(list(spec), facts, where)
    if not all(isinstance(value, bool) for value in spec.values()):
        raise RecipeBookError(f"{where} must map fact names to true or false")
    return dict(spec)


def _skill(name: str, spec: Mapping, facts: Set[str]) -> Skill:
    where = f"skill {name}"
    _check_keys(spec, where, required={"requires", "effects", "may_change"})
    effects = _conditions(spec["effects"], facts, f"{where} effects")
    may_change = _fact_names(spec["may_change"], facts, f"{where} may_change")
    both = sorted(set(effects) & set(may_change))
    if both:
        raise RecipeBookError(f"{where} lists {both} as both an effect and a fact it may change")
    return Skill(name, _conditions(spec["requires"], facts, f"{where} requires"), effects, may_change)


def _method(spec: Mapping, facts: Set[str]) -> Method:
    _check_keys(spec, "a method", required={"name", "subgoals"}, optional={"requires"})
    where = f"method {spec['name']}"
    subgoals = spec["subgoals"]
    if not isinstance(subgoals, list) or not subgoals:
        raise RecipeBookError(f"{where} needs at least one subgoal")
    conditions = tuple(_conditions(subgoal, facts, f"{where} subgoal") for subgoal in subgoals)
    if not all(conditions):
        raise RecipeBookError(f"{where} has a subgoal with no conditions")
    return Method(str(spec["name"]), conditions, _conditions(spec.get("requires", {}), facts, where))


def _goal(name: str, spec: Mapping, facts: Set[str]) -> Goal:
    _check_keys(spec, f"goal {name}", required={"achieved_when"}, optional={"terminal"})
    achieved_when = _conditions(spec["achieved_when"], facts, f"goal {name} achieved_when")
    if not achieved_when:
        raise RecipeBookError(f"goal {name} needs at least one fact in achieved_when")
    terminal = spec.get("terminal", False)
    if not isinstance(terminal, bool):
        raise RecipeBookError(f"goal {name} terminal must be true or false")
    return Goal(name, achieved_when, terminal)

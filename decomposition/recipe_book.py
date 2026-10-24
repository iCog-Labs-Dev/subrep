"""Recipe books: goals, the methods that break them into steps, and what each step needs and does.

A recipe book is a versioned JSON file in the style of a hierarchical task
network (HTN):

- a goal names the task that achieves it and the facts that show it is
  achieved; a terminal goal is achieved only by how the episode ends, so its
  facts are confirmed on the final result rather than mid-episode;
- a task has one or more methods; a method is an ordered list of subtasks
  (tasks or primitives) and the facts that must hold before it is chosen;
- a primitive is a sub-skill with the facts it requires, the facts it makes
  true or false (``effects``), and the facts it may change unpredictably
  (``may_change``). Facts a primitive does not mention keep their value.

The book is checked against the fact and sub-skill names the caller provides,
so this module does not depend on any environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import AbstractSet, Collection, Dict, Mapping, Set, Tuple, Union

SCHEMA_VERSION = "subrep.recipe_book.v1"
LUNAR_LANDER_BOOK = Path(__file__).resolve().parent / "data" / "lunar_recipe_book.json"

Facts = Mapping[str, bool]


class RecipeBookError(ValueError):
    """The recipe book is malformed or does not match the environment's names."""


def holds(conditions: Facts, facts: Facts) -> bool:
    """Whether every condition is known to hold in ``facts``; a missing fact is unknown."""
    return all(name in facts and facts[name] == value for name, value in conditions.items())


@dataclass(frozen=True)
class Primitive:
    """A sub-skill: what it requires, what it achieves and what it may disturb."""

    name: str
    requires: Dict[str, bool]
    effects: Dict[str, bool]
    may_change: Tuple[str, ...]

    def apply(self, facts: Facts) -> Dict[str, bool]:
        """The facts known after this primitive completes from a state with ``facts``."""
        after = {name: value for name, value in facts.items() if name not in self.may_change}
        after.update(self.effects)
        return after


@dataclass(frozen=True)
class Method:
    """One way to carry out a task: ordered subtasks and when the way applies."""

    name: str
    subtasks: Tuple[str, ...]
    requires: Dict[str, bool]


@dataclass(frozen=True)
class Task:
    """A task that is carried out by one of its methods."""

    name: str
    methods: Tuple[Method, ...]


@dataclass(frozen=True)
class Goal:
    """A goal: the task that achieves it and the facts that show it is achieved."""

    name: str
    task: str
    achieved_when: Dict[str, bool]
    terminal: bool = False  # achieved only by how the episode ends


@dataclass(frozen=True)
class RecipeBook:
    """A checked recipe book."""

    environment: str
    goals: Dict[str, Goal]
    tasks: Dict[str, Task]
    primitives: Dict[str, Primitive]


def load_recipe_book(
    path: Union[str, Path], fact_names: Collection[str], skill_names: Collection[str]
) -> RecipeBook:
    """Load and check a recipe book against the environment's fact and sub-skill names."""
    return parse_recipe_book(json.loads(Path(path).read_text(encoding="utf-8")), fact_names, skill_names)


def parse_recipe_book(data: Mapping, fact_names: Collection[str], skill_names: Collection[str]) -> RecipeBook:
    """Check a decoded recipe book and build it."""
    _check_keys(data, "recipe book", required={"schema_version", "environment", "goals", "tasks", "primitives"})
    if data["schema_version"] != SCHEMA_VERSION:
        raise RecipeBookError(f"expected schema_version {SCHEMA_VERSION!r}, got {data['schema_version']!r}")
    facts = set(fact_names)

    primitives = {name: _primitive(name, spec, facts) for name, spec in data["primitives"].items()}
    not_skills = sorted(set(primitives) - set(skill_names))
    if not_skills:
        raise RecipeBookError(f"primitives {not_skills} are not sub-skills of the environment")

    tasks = {name: _task(name, spec, facts) for name, spec in data["tasks"].items()}
    clashes = sorted(set(tasks) & set(primitives))
    if clashes:
        raise RecipeBookError(f"names {clashes} are used for both a task and a primitive")
    for task in tasks.values():
        for method in task.methods:
            unknown = [name for name in method.subtasks if name not in tasks and name not in primitives]
            if unknown:
                raise RecipeBookError(f"method {task.name}/{method.name} uses unknown subtasks {unknown}")
    _check_acyclic(tasks)

    goals = {name: _goal(name, spec, facts) for name, spec in data["goals"].items()}
    for goal in goals.values():
        if goal.task not in tasks:
            raise RecipeBookError(f"goal {goal.name} names unknown task {goal.task!r}")

    return RecipeBook(environment=str(data["environment"]), goals=goals, tasks=tasks, primitives=primitives)


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


def _primitive(name: str, spec: Mapping, facts: Set[str]) -> Primitive:
    where = f"primitive {name}"
    _check_keys(spec, where, required={"requires", "effects", "may_change"})
    effects = _conditions(spec["effects"], facts, f"{where} effects")
    may_change = _fact_names(spec["may_change"], facts, f"{where} may_change")
    both = sorted(set(effects) & set(may_change))
    if both:
        raise RecipeBookError(f"{where} lists {both} as both an effect and a fact it may change")
    return Primitive(name, _conditions(spec["requires"], facts, f"{where} requires"), effects, may_change)


def _task(name: str, spec: Mapping, facts: Set[str]) -> Task:
    _check_keys(spec, f"task {name}", required={"methods"})
    if not isinstance(spec["methods"], list) or not spec["methods"]:
        raise RecipeBookError(f"task {name} needs at least one method")
    methods = []
    for method in spec["methods"]:
        _check_keys(method, f"a method of task {name}", required={"name", "subtasks"}, optional={"requires"})
        where = f"method {name}/{method['name']}"
        subtasks = method["subtasks"]
        if not isinstance(subtasks, list) or not all(isinstance(subtask, str) for subtask in subtasks):
            raise RecipeBookError(f"{where} subtasks must be a list of names")
        methods.append(
            Method(str(method["name"]), tuple(subtasks), _conditions(method.get("requires", {}), facts, where))
        )
    names = [method.name for method in methods]
    if len(set(names)) != len(names):
        raise RecipeBookError(f"task {name} has methods with the same name")
    return Task(name, tuple(methods))


def _goal(name: str, spec: Mapping, facts: Set[str]) -> Goal:
    _check_keys(spec, f"goal {name}", required={"task", "achieved_when"}, optional={"terminal"})
    achieved_when = _conditions(spec["achieved_when"], facts, f"goal {name} achieved_when")
    if not achieved_when:
        raise RecipeBookError(f"goal {name} needs at least one fact in achieved_when")
    terminal = spec.get("terminal", False)
    if not isinstance(terminal, bool):
        raise RecipeBookError(f"goal {name} terminal must be true or false")
    return Goal(name, str(spec["task"]), achieved_when, terminal)


def _check_acyclic(tasks: Mapping[str, Task]) -> None:
    finished: Set[str] = set()

    def visit(name: str, path: Tuple[str, ...]) -> None:
        if name in path:
            raise RecipeBookError(f"tasks form a cycle: {' -> '.join(path + (name,))}")
        if name in finished or name not in tasks:
            return
        for method in tasks[name].methods:
            for subtask in method.subtasks:
                visit(subtask, path + (name,))
        finished.add(name)

    for name in tasks:
        visit(name, ())

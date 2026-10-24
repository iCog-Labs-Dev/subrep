"""Propose plans for a goal by decomposing it with a recipe book.

This is a total-order HTN planner in the style of SHOP. Starting from the
goal's task, it replaces the first task on the agenda with the subtasks of one
of its methods, left to right, while tracking which facts are known:

- a method can be chosen only when its requirements are known to hold;
- a primitive can be added only when its requirements are known to hold, and
  its effects and ``may_change`` list then update the known facts.

Every choice of methods that ends in a list of sub-skills is a candidate
unless the book predicts that it leaves one of the goal's facts false. A fact a
sub-skill may change stays open: the book cannot promise, for example, that a
touch-down ends on the pad, so whether a goal was really achieved is confirmed
from the final result, not predicted here. Recipe books are checked to be free
of cycles when they are loaded, so the search always ends.

The planner only proposes. Whether a candidate is worth running is decided
against measured effects by the verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Tuple

from decomposition.recipe_book import RecipeBook, holds

# (sub-skills, facts known afterwards, chosen methods as (task, method) pairs)
_Expansion = Tuple[Tuple[str, ...], Dict[str, bool], Tuple[Tuple[str, str], ...]]


@dataclass(frozen=True)
class CandidatePlan:
    """A way that may achieve a goal, as an ordered list of sub-skills."""

    goal: str
    option: str  # the goal task's method, then "+" and any other chosen alternative
    methods: Tuple[Tuple[str, str], ...]  # every (task, method) chosen, in order
    skills: Tuple[str, ...]
    predicted_facts: Dict[str, bool]  # facts known once the plan completes


def propose(book: RecipeBook, goal: str, facts: Mapping[str, bool]) -> List[CandidatePlan]:
    """Every distinct plan that may achieve ``goal`` from ``facts``.

    Returns an empty list when no method applies, or when a goal that is not
    terminal already holds; a terminal goal is achieved only by how the episode
    ends, so it never counts as achieved while planning. Candidates are in the
    order of the book's methods.
    """
    if goal not in book.goals:
        raise ValueError(f"unknown goal {goal!r}; the book has {sorted(book.goals)}")
    target = book.goals[goal]
    if not target.terminal and holds(target.achieved_when, facts):
        return []

    candidates: List[CandidatePlan] = []
    seen = set()
    for skills, predicted, methods in _expand(book, (target.task,), dict(facts)):
        if not skills or skills in seen or _contradicts(target.achieved_when, predicted):
            continue
        seen.add(skills)
        candidates.append(CandidatePlan(goal, _option(book, methods), methods, skills, dict(predicted)))
    return candidates


def _expand(book: RecipeBook, agenda: Tuple[str, ...], facts: Dict[str, bool]) -> Iterator[_Expansion]:
    if not agenda:
        yield (), facts, ()
        return
    head, rest = agenda[0], agenda[1:]

    if head in book.primitives:
        primitive = book.primitives[head]
        if holds(primitive.requires, facts):
            for skills, predicted, methods in _expand(book, rest, primitive.apply(facts)):
                yield (head,) + skills, predicted, methods
        return

    task = book.tasks[head]
    for method in task.methods:
        if holds(method.requires, facts):
            for skills, predicted, methods in _expand(book, method.subtasks + rest, facts):
                yield skills, predicted, ((task.name, method.name),) + methods


def _contradicts(conditions: Mapping[str, bool], facts: Mapping[str, bool]) -> bool:
    """Whether some condition is known to be false in ``facts``."""
    return any(name in facts and facts[name] != value for name, value in conditions.items())


def _option(book: RecipeBook, methods: Tuple[Tuple[str, str], ...]) -> str:
    """Name a candidate by its goal method and the alternatives chosen below it."""
    root, below = methods[0], methods[1:]
    alternatives = [method for task, method in below if len(book.tasks[task].methods) > 1]
    return "+".join([root[1], *alternatives])

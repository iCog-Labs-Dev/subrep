"""Propose plans for a goal by decomposing it into subgoals with a recipe book.

This is a hierarchical goal network (HGN) planner in the style of GDP and GoDeL
(Shivashankar et al., 2013). It works on an ordered list of subgoals, each a set
of conditions on facts, while tracking which facts are known:

- a subgoal that already holds is skipped;
- a method applies to a subgoal when its last subgoal is that subgoal and its
  requirements are known to hold, and its subgoals are then pursued in order;
- a skill applies when its requirements are known to hold and its effects make
  all of the subgoal true. One that makes only part of it true, without
  contradicting the rest, applies only when no method or whole skill does;
- when nothing applies, a short search over the skills chains from the known
  facts to the subgoal. This is GoDeL's action chaining; its landmark inference
  is not implemented.

Paper GDP treats any method whose last subgoal shares a fact with the subgoal as
relevant. Here it must be the same subgoal, so that a recipe for landing is not
offered as a way to get over the pad.

Every way of working through the subgoals that ends in a list of skills is a
candidate. A fact a skill may change stays open: the book cannot promise, for
example, that a touch-down ends on the pad, so whether a goal was really achieved
is confirmed from the final result, not predicted here. The planner only
proposes. Whether a candidate is worth running is decided against measured
effects by the verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

from decomposition.recipe_book import Facts, Goal, RecipeBook, Skill, holds

ALREADY_TRUE = "already_true"
TO_ACHIEVE = "to_achieve"
DIRECT = "direct"
SEARCH = "search"
MAX_PLAN_SKILLS = 10  # bounds the planner when a book's skills undo each other
MAX_SEARCH_SKILLS = 5

_Subgoal = Tuple[Tuple[str, bool], ...]  # conditions in a fixed order, so that they compare and hash


@dataclass(frozen=True)
class _MethodEnd:
    """Marks the end of a method's subgoals on the agenda."""

    subgoal: _Subgoal


@dataclass(frozen=True)
class SubgoalStep:
    """One subgoal of a plan: what must hold, and which skills make it hold."""

    conditions: Dict[str, bool]
    status: str  # ALREADY_TRUE when it held at the time, otherwise TO_ACHIEVE
    skills: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidatePlan:
    """A way that may achieve a goal: its subgoals, and the skills that make them hold."""

    goal: str
    option: str  # the method chosen for the goal, DIRECT if a skill applied, SEARCH if the search found it
    methods: Tuple[str, ...]  # every method chosen, in order
    skills: Tuple[str, ...]
    predicted_facts: Dict[str, bool]  # facts known once the plan completes
    subgoals: Tuple[SubgoalStep, ...] = ()
    searched: bool = False  # whether the search filled a gap the book's methods left
    success_outcomes: Tuple[str, ...] = ()  # the endings of an episode that achieve the goal


@dataclass(frozen=True)
class _Partial:
    skills: Tuple[str, ...]
    facts: Dict[str, bool]
    methods: Tuple[str, ...]
    steps: Tuple[SubgoalStep, ...]
    searched: bool


def propose(book: RecipeBook, goal: str, facts: Facts) -> List[CandidatePlan]:
    """Every distinct plan that may achieve ``goal`` from ``facts``.

    Returns an empty list when nothing applies, or when a goal that is not
    terminal already holds; a terminal goal is achieved only by how the episode
    ends, so it never counts as achieved while planning. Candidates follow the
    order of the book's methods, then of its skills.
    """
    if goal not in book.goals:
        raise ValueError(f"unknown goal {goal!r}; the book has {sorted(book.goals)}")
    target = book.goals[goal]
    if not target.terminal and holds(target.achieved_when, facts):
        return []

    subgoal = _key(target.achieved_when)
    candidates: List[CandidatePlan] = []
    seen = set()
    for partial in _Search(book, target).solve((subgoal,), dict(facts), frozenset()):
        if not partial.skills or partial.skills in seen:
            continue
        seen.add(partial.skills)
        candidates.append(
            CandidatePlan(
                goal=goal,
                option=partial.methods[0] if partial.methods else SEARCH if partial.searched else DIRECT,
                methods=partial.methods,
                skills=partial.skills,
                predicted_facts=partial.facts,
                subgoals=partial.steps,
                searched=partial.searched,
                success_outcomes=target.success_outcomes,
            )
        )
    return candidates


def plans_for(book: RecipeBook, goals: Optional[Sequence[str]] = None):
    """A function from facts to the distinct skill lists the planner proposes for ``goals``."""
    names = list(book.goals) if goals is None else list(goals)

    def plans(facts: Facts) -> List[Tuple[str, ...]]:
        found = (candidate.skills for goal in names for candidate in propose(book, goal, facts))
        return list(dict.fromkeys(found))

    return plans


def goal_graph(candidate: CandidatePlan) -> Dict[str, object]:
    """The plan as a JSON-ready goal graph: the goal, its ordered subgoals and the skills under each."""
    nodes = [
        {"id": index, "conditions": dict(step.conditions), "status": step.status, "skills": list(step.skills)}
        for index, step in enumerate(candidate.subgoals)
    ]
    return {
        "goal": candidate.goal,
        "option": candidate.option,
        "searched": candidate.searched,
        "nodes": nodes,
        "edges": [[index, index + 1] for index in range(len(nodes) - 1)],  # each subgoal comes before the next
    }


def _key(conditions: Mapping[str, bool]) -> _Subgoal:
    return tuple(sorted(conditions.items()))


def _not_known_true(subgoal: _Subgoal, facts: Facts) -> Tuple[Tuple[str, bool], ...]:
    return tuple((name, value) for name, value in subgoal if facts.get(name) is not value)


def _may_hold_after(skill: Skill, subgoal: _Subgoal, after: Facts) -> bool:
    """Whether the subgoal may hold once ``skill`` has run: each condition is known true, or the skill may have made it so."""
    return all(after.get(name) is value or (name not in after and name in skill.may_change) for name, value in subgoal)


def _contributes(skill: Skill, subgoal: _Subgoal, facts: Facts) -> bool:
    """Whether the skill makes an unmet condition true and contradicts no other.

    A terminal subgoal may already look true while the episode runs; a skill that
    could have made it true is then what carries the episode to its end.
    """
    effects = skill.effects
    unmet = _not_known_true(subgoal, facts) or subgoal
    return all(effects.get(name, value) == value for name, value in subgoal) and any(
        effects.get(name) is value for name, value in unmet
    )


class _Search:
    """Works through an agenda of subgoals for one goal of one book."""

    def __init__(self, book: RecipeBook, target: Goal) -> None:
        self.book = book
        self.terminal = _key(target.achieved_when) if target.terminal else None

    def solve(
        self, agenda: Tuple[Union[_Subgoal, _MethodEnd], ...], facts: Dict[str, bool], active: FrozenSet[_Subgoal]
    ) -> Iterator[_Partial]:
        if not agenda:
            yield _Partial((), facts, (), (), False)
            return
        head, rest = agenda[0], agenda[1:]

        if isinstance(head, _MethodEnd):
            yield from self.solve(rest, facts, active - {head.subgoal})
            return
        conditions = dict(head)
        if head != self.terminal and holds(conditions, facts):
            step = SubgoalStep(conditions, ALREADY_TRUE)
            for partial in self.solve(rest, facts, active):
                yield replace(partial, steps=(step,) + partial.steps)
            return

        found = False
        for partial in self._by_method(head, rest, facts, active):
            found = True
            yield partial
        for partial in self._by_skill(head, rest, facts, active, whole_only=found):
            found = True
            yield partial
        if not found:
            yield from self._by_chaining(head, rest, facts, active)

    def _by_method(
        self, head: _Subgoal, rest: tuple, facts: Dict[str, bool], active: FrozenSet[_Subgoal]
    ) -> Iterator[_Partial]:
        if head in active:  # a method is working towards this subgoal, so only a skill can finish it
            return
        for method in self.book.methods:
            if method.subgoals and _key(method.subgoals[-1]) == head and holds(method.requires, facts):
                subgoals = tuple(_key(subgoal) for subgoal in method.subgoals)
                for partial in self.solve(subgoals + (_MethodEnd(head),) + rest, facts, active | {head}):
                    yield replace(partial, methods=(method.name,) + partial.methods)

    def _by_skill(
        self, head: _Subgoal, rest: tuple, facts: Dict[str, bool], active: FrozenSet[_Subgoal], whole_only: bool
    ) -> Iterator[_Partial]:
        unmet = _not_known_true(head, facts)
        usable = [
            skill
            for skill in self.book.skills.values()
            if holds(skill.requires, facts) and _contributes(skill, head, facts)
        ]
        whole = [skill for skill in usable if all(skill.effects.get(name) is value for name, value in unmet)]
        for skill in whole if whole or whole_only else usable:
            after = skill.apply(facts)
            if _may_hold_after(skill, head, after):
                remaining = rest
            elif len(_not_known_true(head, after)) < len(unmet):
                remaining = (head,) + rest
            else:
                continue
            yield from self._after_skills((skill.name,), head, remaining, after, active, searched=False)

    def _by_chaining(
        self, head: _Subgoal, rest: tuple, facts: Dict[str, bool], active: FrozenSet[_Subgoal]
    ) -> Iterator[_Partial]:
        frontier = [((), facts)]
        visited = {_key(facts)}
        for _ in range(MAX_SEARCH_SKILLS):
            reached = False
            next_frontier = []
            for skills, known in frontier:
                for skill in self.book.skills.values():
                    if not holds(skill.requires, known):
                        continue
                    after = skill.apply(known)
                    if _contributes(skill, head, known) and _may_hold_after(skill, head, after):
                        reached = True
                        yield from self._after_skills(skills + (skill.name,), head, rest, after, active, searched=True)
                    elif _key(after) not in visited:
                        visited.add(_key(after))
                        next_frontier.append((skills + (skill.name,), after))
            if reached or not next_frontier:
                return
            frontier = next_frontier

    def _after_skills(
        self,
        skills: Tuple[str, ...],
        head: _Subgoal,
        rest: tuple,
        facts: Dict[str, bool],
        active: FrozenSet[_Subgoal],
        searched: bool,
    ) -> Iterator[_Partial]:
        step = SubgoalStep(dict(head), TO_ACHIEVE, skills)
        for partial in self.solve(rest, facts, active):
            if len(skills) + len(partial.skills) <= MAX_PLAN_SKILLS:
                yield replace(
                    partial,
                    skills=skills + partial.skills,
                    steps=(step,) + partial.steps,
                    searched=searched or partial.searched,
                )

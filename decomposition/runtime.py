"""Pursue goals: plan and certify from the real state before every skill.

Each decision reads the facts of the current state and the priorities for it,
asks the planner for candidate plans for every goal on offer, certifies them with
the verifier, and flies only the first skill of the plan it chooses. The next
decision does the same from wherever that skill really left the lander, so the
rest of a plan is re-certified at every subgoal and a subgoal that already holds
is skipped without being told. The run ends when the goal of the plan being flown
is achieved, no plan can be flown, the episode ends, or too many skills fail.

Priorities never name a goal. When several goals are on offer, their plans compete
under the same gates and the same score, and the goal pursued is the goal of the
plan that wins, so it can change when the priorities do.

The plan in progress is kept unless another admitted plan beats it by more than
``SWITCH_MARGIN``, so that small differences between estimates do not make the
lander change its mind at every subgoal.

Every skill flies under a certificate admitted from the state it starts in. When
nothing is admitted, the run stops, unless the caller chooses the continuation
policy ``Continuation.KEEP_UNREJECTED_PLAN``. The plan in progress then goes on
without a new certificate, but only if priorities were given, its last skill
completed, and the rest of it was not rejected from this state: nothing was
measured from states like this one, or it passed the gates but reaches its goal
less often than a new plan must. A rest the gates reject, a rest the planner no
longer proposes, and refused priorities stop the run under either policy.
Decisions taken under the continuation policy are recorded as continued, never
as re-certified.

A new plan must reach its goal at least ``DEFAULT_MIN_GOAL_RATE`` of the time
with 95% confidence, so that a plan cannot be chosen because it ends the episode
quickly while often missing the goal. The rate is a policy choice for the team.

Whether a goal is achieved is confirmed, never assumed:

- once the episode is over, the final result decides. The goal counts as
  achieved only if the domain confirms the ending and the goal's facts hold in
  the final state, so landing on the pad needs the lander to rest on the pad.
  Legs touching the ground before a crash, or a landing beside the pad, do not
  count;
- while the episode runs, a goal that is not terminal is achieved when its facts
  hold. A terminal goal, such as landing on the pad, is achieved only by how the
  episode ends.

By default the runtime never acts without a current certificate: when no plan can
be flown it stops and leaves the lander to the caller. Given a feedback log, it
also skips plans whose certificates the log has revoked. A source of priorities
may refuse to give any, as an MDN whose weights lie outside its own region does;
nothing new is then certified.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from decomposition.domain import Domain
from decomposition.effects import EffectTable
from decomposition.executor import Execution, execute
from decomposition.feedback import FeedbackLog
from decomposition.planner import CandidatePlan, Generation, generate, goal_graph
from decomposition.recipe_book import Goal, RecipeBook, holds
from decomposition.verifier import (
    GOAL_TOO_RARE,
    NOT_MEASURED,
    PlanVerdict,
    Priorities,
    best_plan,
    plan_skill_id,
    verify,
)

DEFAULT_MAX_ATTEMPTS = 3  # skills that may fail before the run gives up
DEFAULT_MIN_GOAL_RATE = 0.9
MAX_DECISIONS = 30
SWITCH_MARGIN = 0.1  # another plan must score this fraction higher to replace the plan in progress

PrioritySource = Callable[[np.ndarray], Optional[Priorities]]


class GoalStatus(str, Enum):
    """How the pursuit of a goal ended."""

    ACHIEVED = "achieved"
    ALREADY_ACHIEVED = "already_achieved"
    NO_CERTIFIED_PLAN = "no_certified_plan"
    PRIORITIES_REFUSED = "priorities_refused"
    FAILED = "failed"


class Continuation(str, Enum):
    """What the runtime does when no plan is admitted from the current state."""

    STOP = "stop"  # every skill needs a certificate admitted from the state it starts in
    KEEP_UNREJECTED_PLAN = "keep_unrejected_plan"  # the plan in progress may go on, as the module describes


@dataclass(frozen=True)
class Decision:
    """One round of planning and certifying, and the skill flown because of it."""

    facts: Dict[str, bool]
    verdicts: Tuple[PlanVerdict, ...]
    chosen: Optional[PlanVerdict]  # the certified plan flown; None when nothing could be flown
    skills: Tuple[str, ...]  # the part of the chosen plan still to fly, first skill included
    execution: Optional[Execution]
    recertified: bool  # whether the verifier admitted this plan from this very state
    skipped: Tuple[CandidatePlan, ...] = ()  # candidates a measuring limit left out, kept for the record
    weights: Optional[Tuple[float, ...]] = None  # the priorities at this decision; None if refused
    continued: bool = False  # flown under the continuation policy, on a certificate from an earlier state


@dataclass(frozen=True)
class GoalRun:
    """Everything that happened while pursuing goals."""

    goal: str  # the goal of the last plan flown, or the first goal offered if none was
    status: GoalStatus
    decisions: Tuple[Decision, ...]
    rewards: np.ndarray  # (frames, objectives) over every decision
    observation: np.ndarray
    episode_over: bool
    info: Optional[Dict[str, Any]]

    @property
    def frames(self) -> int:
        return len(self.rewards)

    @property
    def flown(self) -> Tuple[Decision, ...]:
        return tuple(decision for decision in self.decisions if decision.execution is not None)

    @property
    def actions(self) -> Tuple[int, ...]:
        return tuple(action for decision in self.flown for action in decision.execution.actions)

    @property
    def rechecks(self) -> int:
        """Decisions taken after the first, that is, mid-flight."""
        return max(0, len(self.decisions) - 1)

    @property
    def recertified_rechecks(self) -> int:
        """Mid-flight decisions at which the verifier admitted the plan from the real state."""
        return sum(decision.recertified for decision in self.decisions[1:])

    @property
    def continued(self) -> int:
        """Decisions flown under the continuation policy, without current approval."""
        return sum(decision.continued for decision in self.decisions)

    @property
    def switches(self) -> int:
        """Times a skill finished and the next one flown was not the next skill of the plan."""
        return sum(
            before.execution.completed and after.skills != before.skills[1:]
            for before, after in zip(self.flown, self.flown[1:])
        )

    @property
    def replans(self) -> int:
        """Times a skill failed and the run went on with a new plan."""
        return sum(before.execution.stopped_at is not None for before, _ in zip(self.flown, self.flown[1:]))

    def record(self) -> Dict[str, object]:
        """A JSON-ready summary of the run."""
        return {
            "goal": self.goal,
            "status": self.status.value,
            "frames": self.frames,
            "rechecks": self.rechecks,
            "recertified_rechecks": self.recertified_rechecks,
            "continued": self.continued,
            "switches": self.switches,
            "replans": self.replans,
            "episode_over": self.episode_over,
            "decisions": [_decision_record(decision) for decision in self.decisions],
        }


def pursue(
    domain: Domain,
    env: Any,
    observation: np.ndarray,
    goals: Union[str, Sequence[str]],
    book: RecipeBook,
    table: EffectTable,
    priorities: Union[Priorities, PrioritySource],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    feedback: Optional[FeedbackLog] = None,
    timestamp: Optional[str] = None,
    min_goal_rate: float = DEFAULT_MIN_GOAL_RATE,
    limit: Optional[int] = None,
    continuation: Continuation = Continuation.STOP,
) -> GoalRun:
    """Pursue ``goals`` from ``observation``, which ``env`` must have just produced.

    ``priorities`` are fixed, or a source asked at every decision. ``limit`` caps
    how many candidates are measured at each decision, most promising first under
    the priorities. It is off by default. ``continuation`` says what happens when
    nothing is admitted; by default the run stops.
    """
    offered = (goals,) if isinstance(goals, str) else tuple(goals)
    unknown = [goal for goal in offered if goal not in book.goals]
    if unknown or not offered:
        raise ValueError(f"unknown goal {unknown or offered!r}; the book has {sorted(book.goals)}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
    obs = np.asarray(observation, dtype=np.float32)
    decisions: List[Decision] = []
    rewards: List[np.ndarray] = []
    info: Optional[Dict[str, Any]] = None
    episode_over = False
    in_progress: Optional[PlanVerdict] = None  # the certified plan being flown
    remaining: Tuple[str, ...] = ()  # the part of it still to fly
    failures = 0
    achieved = next((g for g in offered if goal_achieved(domain, book.goals[g], obs, episode_over, info)), None)

    if achieved is not None:
        status = GoalStatus.ALREADY_ACHIEVED
    else:
        status = GoalStatus.FAILED
        for _ in range(MAX_DECISIONS):
            facts = domain.facts(obs)
            current = priorities(obs) if callable(priorities) else priorities
            generation = Generation(())
            verdicts: Tuple[PlanVerdict, ...] = ()
            if current is not None:
                weights = current.weights
                generation = generate(
                    book, offered, facts, lambda skills: table.rough_value(skills, weights), limit, not decisions
                )
                verdicts = tuple(verify(generation.candidates, obs, table, current, timestamp, min_goal_rate))
                if feedback is not None:
                    verdicts = feedback.screen(verdicts, facts)
            weights = None if current is None else current.weights
            skipped = generation.skipped

            chosen = _choose(verdicts, remaining)
            recertified = chosen is not None
            continued = (
                not recertified
                and continuation is Continuation.KEEP_UNREJECTED_PLAN
                and current is not None
                and _may_continue(verdicts, remaining)
            )
            if recertified:
                in_progress, remaining = chosen, chosen.candidate.skills
            elif not continued:
                decisions.append(Decision(facts, verdicts, None, (), None, False, skipped, weights))
                status = GoalStatus.NO_CERTIFIED_PLAN if current is not None else GoalStatus.PRIORITIES_REFUSED
                break

            execution = execute(domain, env, book, remaining[:1], obs, settle=len(remaining) == 1)
            decisions.append(
                Decision(facts, verdicts, in_progress, remaining, execution, recertified, skipped, weights, continued)
            )
            rewards.extend(execution.rewards)
            obs, episode_over = execution.observation, execution.episode_over
            if execution.info is not None:
                info = execution.info
            if goal_achieved(domain, book.goals[in_progress.candidate.goal], obs, episode_over, info):
                status = GoalStatus.ACHIEVED
                break
            if episode_over:
                break
            if execution.completed:
                remaining = remaining[1:]
            else:
                remaining = ()
                failures += 1
                if failures >= max_attempts:
                    break

    return GoalRun(
        goal=achieved or (in_progress.candidate.goal if in_progress is not None else offered[0]),
        status=status,
        decisions=tuple(decisions),
        rewards=np.asarray(rewards, dtype=np.float64).reshape(-1, domain.num_objectives),
        observation=obs,
        episode_over=episode_over,
        info=info,
    )


def _verdict_for(verdicts: Sequence[PlanVerdict], skills: Tuple[str, ...]) -> Optional[PlanVerdict]:
    return next((verdict for verdict in verdicts if verdict.candidate.skills == skills), None)


def _choose(verdicts: Sequence[PlanVerdict], remaining: Tuple[str, ...]) -> Optional[PlanVerdict]:
    """The best admitted plan, unless the plan in progress is admitted and not clearly beaten."""
    best = best_plan(verdicts)
    current = _verdict_for(verdicts, remaining)
    if best is None or current is None or not current.admitted:
        return best
    return best if best.score > current.score + SWITCH_MARGIN * abs(current.score) else current


def _may_continue(verdicts: Sequence[PlanVerdict], remaining: Tuple[str, ...]) -> bool:
    """Whether the continuation policy lets the plan in progress go on although nothing was admitted.

    Only a rest the planner still proposes and the gates did not reject may go on.
    """
    current = _verdict_for(verdicts, remaining) if remaining else None
    return current is not None and current.reason in (NOT_MEASURED, GOAL_TOO_RARE)


def goal_achieved(
    domain: Domain, target: Goal, observation: np.ndarray, episode_over: bool, info: Optional[Mapping[str, Any]]
) -> bool:
    """Whether ``target`` is achieved in the state ``observation``, confirmed as the module describes."""
    if episode_over:
        final = domain.final_facts(info, observation)
        return final is not None and holds(target.achieved_when, final)
    return not target.terminal and holds(target.achieved_when, domain.facts(observation))


def _decision_record(decision: Decision) -> Dict[str, object]:
    chosen, execution = decision.chosen, decision.execution
    return {
        "facts": dict(decision.facts),
        "weights": None if decision.weights is None else list(decision.weights),
        "plan": None if chosen is None else plan_skill_id(chosen.candidate),
        "goal_graph": None if chosen is None else goal_graph(chosen.candidate),
        "skills_left": list(decision.skills),
        "recertified": decision.recertified,
        "continued": decision.continued,
        "gate": None if chosen is None else chosen.entry.gate_type,
        "score": None if chosen is None else chosen.score,
        "rejected": [
            {"plan": plan_skill_id(v.candidate), "reason": v.reason} for v in decision.verdicts if not v.admitted
        ],
        "not_measured_by_limit": [plan_skill_id(candidate) for candidate in decision.skipped],
        "steps": []
        if execution is None
        else [
            {
                "skill": step.skill,
                "status": step.status.value,
                "frames": step.frames,
                "prediction_held": step.prediction_held,
            }
            for step in execution.steps
        ],
        "settling_frames": 0 if execution is None else execution.settling_frames,
    }

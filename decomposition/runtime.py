"""Pursue a goal: plan, certify, fly, and replan from the real state when a step fails.

Each attempt reads the facts of the current state, asks the planner for
candidate plans, certifies them with the verifier, and flies the best admitted
plan with the executor. When a step stops the plan and the episode goes on,
the next attempt plans again from the state the lander is really in. The run
ends when the goal is achieved, no plan is admitted, the episode ends, or the
attempts run out.

While the episode runs, the goal's facts say whether it is achieved. Once the
episode is over, the environment's verdict decides: the goal counts as achieved
only if the episode ended in a successful landing. Leg contact flickers when
the lander comes to rest, so the last frame's facts alone can miss a landing.

The runtime never acts without a certificate: when no plan is admitted it
stops and leaves the lander to the caller. Given a feedback log, it also skips
plans whose certificates the log has revoked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from decomposition.effects import EffectTable
from decomposition.executor import Execution, execute
from decomposition.feedback import FeedbackLog
from decomposition.planner import propose
from decomposition.recipe_book import Goal, RecipeBook, holds
from decomposition.verifier import PlanVerdict, Priorities, best_plan, plan_skill_id, verify
from env.lunar_facts import lander_facts

DEFAULT_MAX_ATTEMPTS = 3


class GoalStatus(str, Enum):
    """How the pursuit of a goal ended."""

    ACHIEVED = "achieved"
    ALREADY_ACHIEVED = "already_achieved"
    NO_CERTIFIED_PLAN = "no_certified_plan"
    FAILED = "failed"


@dataclass(frozen=True)
class Attempt:
    """One round of planning, certifying and flying."""

    facts: Dict[str, bool]
    verdicts: Tuple[PlanVerdict, ...]
    execution: Optional[Execution]  # None when no plan was admitted

    @property
    def chosen(self) -> Optional[PlanVerdict]:
        return best_plan(self.verdicts)


@dataclass(frozen=True)
class GoalRun:
    """Everything that happened while pursuing one goal."""

    goal: str
    status: GoalStatus
    attempts: Tuple[Attempt, ...]
    rewards: np.ndarray  # (frames, 2) over every attempt
    observation: np.ndarray
    episode_over: bool
    info: Optional[Dict[str, Any]]

    @property
    def frames(self) -> int:
        return len(self.rewards)

    @property
    def replans(self) -> int:
        """Attempts flown after the first."""
        return max(0, sum(attempt.execution is not None for attempt in self.attempts) - 1)

    def record(self) -> Dict[str, object]:
        """A JSON-ready summary of the run."""
        return {
            "goal": self.goal,
            "status": self.status.value,
            "frames": self.frames,
            "replans": self.replans,
            "episode_over": self.episode_over,
            "attempts": [_attempt_record(attempt) for attempt in self.attempts],
        }


def pursue(
    env: Any,
    observation: np.ndarray,
    goal: str,
    book: RecipeBook,
    table: EffectTable,
    priorities: Priorities,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    feedback: Optional[FeedbackLog] = None,
    timestamp: Optional[str] = None,
) -> GoalRun:
    """Pursue ``goal`` from ``observation``, which ``env`` must have just produced."""
    if goal not in book.goals:
        raise ValueError(f"unknown goal {goal!r}; the book has {sorted(book.goals)}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
    target = book.goals[goal]
    obs = np.asarray(observation, dtype=np.float32)
    attempts: List[Attempt] = []
    rewards: List[np.ndarray] = []
    info: Optional[Dict[str, Any]] = None
    episode_over = False

    if _achieved(target, obs, episode_over, info):
        status = GoalStatus.ALREADY_ACHIEVED
    else:
        status = GoalStatus.FAILED
        for _ in range(max_attempts):
            facts = lander_facts(obs)
            verdicts = tuple(verify(propose(book, goal, facts), facts, table, priorities, timestamp))
            if feedback is not None:
                verdicts = feedback.screen(verdicts, facts)
            chosen = best_plan(verdicts)
            if chosen is None:
                attempts.append(Attempt(facts, verdicts, None))
                status = GoalStatus.NO_CERTIFIED_PLAN
                break

            execution = execute(env, book, chosen.candidate.skills, obs)
            attempts.append(Attempt(facts, verdicts, execution))
            rewards.extend(execution.rewards)
            obs, episode_over = execution.observation, execution.episode_over
            if execution.info is not None:
                info = execution.info
            if _achieved(target, obs, episode_over, info):
                status = GoalStatus.ACHIEVED
                break
            if episode_over:
                break

    return GoalRun(
        goal=goal,
        status=status,
        attempts=tuple(attempts),
        rewards=np.asarray(rewards, dtype=np.float64).reshape(-1, 2),
        observation=obs,
        episode_over=episode_over,
        info=info,
    )


def _achieved(
    target: Goal, observation: np.ndarray, episode_over: bool, info: Optional[Mapping[str, Any]]
) -> bool:
    if episode_over:
        return bool(info and info.get("landing_success"))
    return holds(target.achieved_when, lander_facts(observation))


def _attempt_record(attempt: Attempt) -> Dict[str, object]:
    chosen, execution = attempt.chosen, attempt.execution
    return {
        "facts": dict(attempt.facts),
        "plan": None if chosen is None else plan_skill_id(chosen.candidate),
        "gate": None if chosen is None else chosen.entry.gate_type,
        "score": None if chosen is None else chosen.score,
        "rejected": [
            {"plan": plan_skill_id(v.candidate), "reason": v.reason} for v in attempt.verdicts if not v.admitted
        ],
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

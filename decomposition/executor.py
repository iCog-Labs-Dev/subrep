"""Fly a plan in the environment, re-checking the recipe book at every step.

Before each sub-skill runs, its requirements are checked against the real
state; after it completes, the facts the book predicts are checked against the
real state too. A step whose requirements do not hold, a sub-skill that runs
out of frames, or the end of the episode stops the plan, and the execution
says which step stopped it and why, so the caller can replan from where the
lander really is. A prediction that does not hold is recorded but does not stop
the plan: the next step's requirement check decides whether the plan can go on.

A plan that leaves the lander standing on its legs keeps running its last
sub-skill until the episode ends, as in the effect measurement, so the landing
reward is collected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from decomposition.recipe_book import RecipeBook, holds
from env.lunar_facts import lander_facts
from env.lunar_subskills import SUBSKILLS, run_subskill


class StepStatus(str, Enum):
    """How one step of a plan ended."""

    COMPLETED = "completed"
    REQUIREMENTS_NOT_MET = "requirements_not_met"
    OUT_OF_FRAMES = "out_of_frames"
    EPISODE_ENDED = "episode_ended"


@dataclass(frozen=True)
class StepRecord:
    """What happened when one step of a plan ran."""

    skill: str
    status: StepStatus
    frames: int
    facts_before: Dict[str, bool]
    facts_after: Dict[str, bool]
    prediction_held: Optional[bool]  # whether the book's predicted facts held; None if the step did not complete


@dataclass(frozen=True)
class Execution:
    """The result of flying a plan from one state."""

    skills: Tuple[str, ...]
    steps: Tuple[StepRecord, ...]
    actions: Tuple[int, ...]
    rewards: np.ndarray  # (frames, 2): every frame flown, settling included
    settling_frames: int
    observation: np.ndarray
    episode_over: bool
    info: Optional[Dict[str, Any]]  # the environment's info from the last frame

    @property
    def completed(self) -> bool:
        """Whether every step of the plan completed."""
        return len(self.steps) == len(self.skills) and all(s.status is StepStatus.COMPLETED for s in self.steps)

    @property
    def stopped_at(self) -> Optional[StepRecord]:
        """The step that stopped the plan, or None if every step completed."""
        return next((step for step in self.steps if step.status is not StepStatus.COMPLETED), None)

    @property
    def frames(self) -> int:
        return len(self.actions)


def execute(env: Any, book: RecipeBook, skills: Sequence[str], observation: np.ndarray) -> Execution:
    """Fly ``skills`` in order from ``observation``, which ``env`` must have just produced."""
    if not skills:
        raise ValueError("a plan needs at least one sub-skill")
    obs = np.asarray(observation, dtype=np.float32)
    steps: List[StepRecord] = []
    actions: List[int] = []
    rewards: List[np.ndarray] = []
    info: Optional[Dict[str, Any]] = None
    episode_over = False

    for name in skills:
        primitive = book.primitives[name]
        before = lander_facts(obs)
        if not holds(primitive.requires, before):
            steps.append(StepRecord(name, StepStatus.REQUIREMENTS_NOT_MET, 0, before, before, None))
            break

        run = run_subskill(env, SUBSKILLS[name], obs)
        actions.extend(run.actions)
        rewards.extend(run.rewards)
        obs = run.observation
        if run.info is not None:
            info = run.info
        episode_over = run.terminated or run.truncated
        after = lander_facts(obs)

        if run.completed:
            status = StepStatus.COMPLETED
            prediction_held: Optional[bool] = holds(primitive.apply(before), after)
        else:
            status = StepStatus.EPISODE_ENDED if episode_over else StepStatus.OUT_OF_FRAMES
            prediction_held = None
        steps.append(StepRecord(name, status, run.frames, before, after, prediction_held))
        if episode_over or status is not StepStatus.COMPLETED:
            break

    settling_frames = 0
    plan_completed = len(steps) == len(skills) and steps[-1].status is StepStatus.COMPLETED
    if plan_completed and not episode_over and lander_facts(obs)["landed"]:
        last = SUBSKILLS[skills[-1]]
        while not episode_over:
            action = int(last.policy(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(np.asarray(reward, dtype=np.float64))
            settling_frames += 1
            episode_over = terminated or truncated

    return Execution(
        skills=tuple(skills),
        steps=tuple(steps),
        actions=tuple(actions),
        rewards=np.asarray(rewards, dtype=np.float64).reshape(-1, 2),
        settling_frames=settling_frames,
        observation=np.asarray(obs, dtype=np.float32),
        episode_over=episode_over,
        info=info,
    )

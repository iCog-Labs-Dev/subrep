"""Short scripted sub-skills that fly the MO-LunarLander lander.

Each sub-skill steers like Gymnasium's built-in landing controller
(``gymnasium.envs.box2d.lunar_lander.heuristic``, MIT License): it fires the
engines to move the lander towards an angle target and a height target. Every
sub-skill uses its own targets and stops as soon as its completion check holds,
or when it reaches its frame limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from env.lunar_facts import lander_facts

NO_ENGINE, LEFT_ENGINE, MAIN_ENGINE, RIGHT_ENGINE = 0, 1, 2, 3

# Steering gains, dead band and tilt limit, as in Gymnasium's heuristic.
_ANGLE_GAIN = 0.5
_SPIN_GAIN = 1.0
_HEIGHT_GAIN = 0.5
_VERTICAL_SPEED_GAIN = 0.5
_DEAD_BAND = 0.05
_MAX_TILT = 0.4

# Height offsets used by the descent sub-skills, in observation units.
_DESCENT_STEP = 0.25
_BRAKE_LIFT = 0.3


def steer(observation: np.ndarray, angle_target: float, height_target: float) -> int:
    """Pick the engine that moves the lander towards the two targets."""
    s = observation
    angle_todo = (angle_target - s[4]) * _ANGLE_GAIN - s[5] * _SPIN_GAIN
    height_todo = (height_target - s[1]) * _HEIGHT_GAIN - s[3] * _VERTICAL_SPEED_GAIN
    if s[6] or s[7]:
        # A leg is down: stop rotating and only soften the vertical speed.
        angle_todo = 0.0
        height_todo = -s[3] * _VERTICAL_SPEED_GAIN

    if height_todo > abs(angle_todo) and height_todo > _DEAD_BAND:
        return MAIN_ENGINE
    if angle_todo < -_DEAD_BAND:
        return RIGHT_ENGINE
    if angle_todo > _DEAD_BAND:
        return LEFT_ENGINE
    return NO_ENGINE


def angle_towards_pad(observation: np.ndarray) -> float:
    """The tilt that carries the lander back over the pad, as in the heuristic."""
    target = observation[0] * 0.5 + observation[2] * 1.0
    return float(np.clip(target, -_MAX_TILT, _MAX_TILT))


@dataclass(frozen=True)
class SubSkill:
    """A scripted policy with a completion check and a frame limit."""

    name: str
    policy: Callable[[np.ndarray], int]
    is_done: Callable[[Dict[str, bool]], bool]
    max_frames: int


@dataclass(frozen=True)
class SubSkillRun:
    """What happened when a sub-skill was run once.

    ``rewards`` holds one reward vector per frame, so callers can sum or
    discount them as they need.
    """

    skill: str
    completed: bool
    frames: int
    actions: tuple
    rewards: np.ndarray
    observation: np.ndarray
    terminated: bool
    truncated: bool
    info: Optional[Dict[str, Any]]


SUBSKILLS: Dict[str, SubSkill] = {
    skill.name: skill
    for skill in (
        SubSkill(
            name="stabilise",
            policy=lambda s: steer(s, 0.0, s[1]),
            is_done=lambda f: f["upright"],
            max_frames=120,
        ),
        SubSkill(
            name="move_over_pad",
            policy=lambda s: steer(s, angle_towards_pad(s), s[1]),
            is_done=lambda f: f["over_pad"],
            max_frames=300,
        ),
        SubSkill(
            name="slow_descent",
            policy=lambda s: steer(s, angle_towards_pad(s), max(s[1] - _DESCENT_STEP, 0.0)),
            is_done=lambda f: f["near_ground"] and f["slow"],
            max_frames=400,
        ),
        SubSkill(
            name="free_fall",
            policy=lambda s: NO_ENGINE,
            is_done=lambda f: f["falling_fast"] or f["near_ground"],
            max_frames=300,
        ),
        SubSkill(
            name="brake",
            policy=lambda s: steer(s, angle_towards_pad(s), s[1] + _BRAKE_LIFT),
            is_done=lambda f: f["slow"],
            max_frames=150,
        ),
        SubSkill(
            name="touch_down",
            policy=lambda s: steer(s, angle_towards_pad(s), 0.0),
            is_done=lambda f: f["landed"],
            max_frames=300,
        ),
    )
}


def run_subskill(env: Any, skill: SubSkill, observation: np.ndarray) -> SubSkillRun:
    """Run ``skill`` from ``observation`` until it completes, the episode ends,
    or its frame limit is reached.

    ``env`` must be the environment that produced ``observation``.
    """
    obs = np.asarray(observation, dtype=np.float32)
    actions = []
    rewards = []
    terminated = truncated = False
    info: Optional[Dict[str, Any]] = None

    while not skill.is_done(lander_facts(obs)) and len(actions) < skill.max_frames:
        action = int(skill.policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        rewards.append(np.asarray(reward, dtype=np.float64))
        if terminated or truncated:
            break

    return SubSkillRun(
        skill=skill.name,
        completed=skill.is_done(lander_facts(obs)),
        frames=len(actions),
        actions=tuple(actions),
        rewards=np.asarray(rewards, dtype=np.float64).reshape(-1, 2),
        observation=np.asarray(obs, dtype=np.float32),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=info,
    )

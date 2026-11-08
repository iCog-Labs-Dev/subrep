"""Scan a beacon, then land: a LunarLander mission where the order of steps matters.

A beacon floats away from the pad. To scan it the lander must stay near it, moving
slowly, for ``SCAN_FRAMES`` frames in a row. The mission is complete only when the
beacon has been scanned and the lander then rests on the pad, so no single
landing controller can complete it by landing well.

``BeaconMissionEnv`` adds a third objective, Mission, to the reward vector of
``SubRepEnv``: a reward when the scan completes and another when the mission does.
The physics and the Safety and Fuel rewards do not change. Observations gain the
offset to the beacon and the scan's progress, so the state stays fully observed.
"""

from __future__ import annotations

from collections import ChainMap
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from decomposition.domain import Domain, Facts
from env.lunar_facts import FACT_NAMES, OUTCOMES, PAD_HALF_WIDTH, landing_outcome, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_pilots import mission_pilots
from env.lunar_subskills import _MAX_TILT, NO_ENGINE, SUBSKILLS, SubSkill, angle_towards_pad, run_subskill, steer

DEFAULT_BEACON = (-0.5, 0.7)
SCAN_RADIUS = 0.15
SCAN_MAX_SPEED = 0.25
SCAN_FRAMES = 15
# Discounted over a flight of about 600 frames, these are worth about 25 by the time they are earned.
SCAN_REWARD = 150.0
MISSION_REWARD = 150.0
MISSION_GAMMA = 0.999  # a mission takes about 500 frames, far beyond the 100-frame horizon of a discount of 0.99
CENTRED_HALF_WIDTH = 0.1
CENTRED_MAX_DRIFT = 0.1
CRUISE_HEIGHT = 0.5

MISSION_COMPLETE = "mission_complete"
BEACON_FACT_NAMES = FACT_NAMES + ("beacon_scanned", "centred_over_pad")
BEACON_OUTCOMES = (MISSION_COMPLETE,) + OUTCOMES
OBJECTIVES = ("safety", "fuel", "mission")

_LANDER_NUMBERS = 8
_BEACON_DX, _BEACON_DY, _SCAN_PROGRESS = 8, 9, 10


class BeaconMissionEnv:
    """``SubRepEnv`` with a beacon to scan and a Mission objective."""

    def __init__(self, seed: int = 42, beacon: Tuple[float, float] = DEFAULT_BEACON) -> None:
        self.base = SubRepEnv(seed=seed)
        self.beacon = (float(beacon[0]), float(beacon[1]))
        self._scan_frames = 0
        self._scanned = False

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        observation, info = self.base.reset(seed=seed)
        self._scan_frames = 0
        self._scanned = False
        return self._observation(observation), info

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.base.step(action)
        mission = self._update_scan(observation)
        landed_on_pad = bool(info["landing_success"] and abs(observation[0]) <= PAD_HALF_WIDTH)
        info["mission_complete"] = landed_on_pad and self._scanned
        if info["mission_complete"]:
            mission += MISSION_REWARD
        info["beacon_scanned"] = self._scanned
        return (
            self._observation(observation),
            np.array([reward[0], reward[1], mission], dtype=np.float32),
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        self.base.close()

    def _update_scan(self, observation: np.ndarray) -> float:
        x, y, vx, vy = observation[:4]
        near = abs(x - self.beacon[0]) < SCAN_RADIUS and abs(y - self.beacon[1]) < SCAN_RADIUS
        steady = abs(vx) < SCAN_MAX_SPEED and abs(vy) < SCAN_MAX_SPEED
        self._scan_frames = self._scan_frames + 1 if near and steady else 0
        if not self._scanned and self._scan_frames >= SCAN_FRAMES:
            self._scanned = True
            return SCAN_REWARD
        return 0.0

    def _observation(self, observation: np.ndarray) -> np.ndarray:
        progress = 1.0 if self._scanned else self._scan_frames / SCAN_FRAMES
        extra = (observation[0] - self.beacon[0], observation[1] - self.beacon[1], progress)
        return np.concatenate([observation, np.asarray(extra, dtype=np.float32)], dtype=np.float32)


def beacon_facts(observation: np.ndarray) -> Facts:
    """The lander's facts, with two more.

    - ``beacon_scanned``: the scan has completed.
    - ``centred_over_pad``: near the middle of the pad and hardly drifting, so a
      lander that has flown in from far away has room to come down on the pad.
    """
    values = np.asarray(observation, dtype=np.float64).reshape(-1)
    return {
        **lander_facts(values[:_LANDER_NUMBERS]),
        "beacon_scanned": bool(values[_SCAN_PROGRESS] >= 1.0),
        "centred_over_pad": bool(abs(values[0]) <= CENTRED_HALF_WIDTH and abs(values[2]) <= CENTRED_MAX_DRIFT),
    }


def mission_outcome(info: Mapping[str, Any], observation: np.ndarray) -> str:
    """How an episode ended; a landing on the pad after the scan completes the mission."""
    if info.get("mission_complete"):
        return MISSION_COMPLETE
    return landing_outcome(info, np.asarray(observation)[:_LANDER_NUMBERS])


def _confirmed_final_facts(info: Optional[Mapping[str, Any]], observation: np.ndarray) -> Optional[Facts]:
    if not (info and info.get("landing_success")):
        return None
    return {**beacon_facts(observation), "landed": True}


def _towards_beacon(observation: np.ndarray) -> int:
    """Tilt to carry the lander to the beacon and hold its height, as the pad's steering does."""
    s = observation
    angle = float(np.clip(s[_BEACON_DX] * 0.5 + s[2] * 1.0, -_MAX_TILT, _MAX_TILT))
    return steer(s, angle, s[1] - s[_BEACON_DY])


GO_TO_BEACON = SubSkill(
    name="go_to_beacon",
    policy=_towards_beacon,
    is_done=lambda f: f["beacon_scanned"],
    max_frames=500,
)
# ``move_over_pad`` is made for short hops: over a long flight it lets the lander
# sink, and it stops at the edge of the pad. This one holds a height and stops in the middle.
CRUISE_TO_PAD = SubSkill(
    name="cruise_to_pad",
    policy=lambda s: steer(s, angle_towards_pad(s), CRUISE_HEIGHT),
    is_done=lambda f: f["centred_over_pad"],
    max_frames=400,
)
MISSION_SKILLS = {skill.name: skill for skill in (GO_TO_BEACON, CRUISE_TO_PAD)}


def _run_skill(env: Any, skill: SubSkill, observation: np.ndarray) -> Any:
    return run_subskill(env, skill, observation, facts=beacon_facts, objectives=len(OBJECTIVES))


def beacon_domain(
    beacon: Tuple[float, float] = DEFAULT_BEACON, pilots: Optional[Mapping[str, Any]] = None
) -> Domain:
    """The mission as a goal decomposition domain, for a beacon at ``beacon``."""
    return Domain(
        environment=f"mo-lunar-lander-v3+beacon({beacon[0]},{beacon[1]})",
        objectives=OBJECTIVES,
        fact_names=BEACON_FACT_NAMES,
        group_facts=("upright", "over_pad", "near_ground", "slow", "falling_fast", "beacon_scanned"),
        outcomes=BEACON_OUTCOMES,
        idle_action=NO_ENGINE,
        skills=ChainMap(SUBSKILLS, MISSION_SKILLS, dict(pilots or {})),
        settling_fact="landed",
        facts=beacon_facts,
        run_skill=_run_skill,
        outcome=mission_outcome,
        final_facts=_confirmed_final_facts,
        make_env=lambda seed: BeaconMissionEnv(seed=seed, beacon=beacon),
        gamma=MISSION_GAMMA,
    )


BEACON_MISSION = beacon_domain(DEFAULT_BEACON, mission_pilots())

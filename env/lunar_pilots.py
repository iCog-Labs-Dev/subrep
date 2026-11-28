"""Whole-episode controllers offered to the planner as single skills.

A pilot flies from wherever it starts until the lander is down, so a plan can be
one skill long. Pilots go through the same measurements and gates as plans of
several skills, which lets plans and single skills compete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Union

import numpy as np

from env.lunar_subskills import SubSkill

PPO_CHECKPOINT = Path("models/pilot_ppo.pt")
MISSION_PPO_CHECKPOINT = Path("models/mission_pilot_ppo.pt")
MAX_PILOT_FRAMES = 1000  # a LunarLander episode lasts at most this long
_LANDER_NUMBERS = 8


class _CheckpointPolicy:
    """A saved PPO pilot, loaded when first asked for an action, reading as many numbers as it was trained on."""

    def __init__(self, checkpoint: Union[str, Path]) -> None:
        self.checkpoint = Path(checkpoint)
        self._pilot = None

    def __call__(self, observation: np.ndarray) -> int:
        if self._pilot is None:
            from pilot.rl_pilot import RLPilot

            self._pilot = RLPilot.load(self.checkpoint, map_location="cpu")
        inputs = np.asarray(observation, dtype=np.float32)[: self._pilot.observation_dim]
        return int(self._pilot.predict(inputs, deterministic=True, return_probability=False))


def _heuristic(observation: np.ndarray) -> int:
    from pilot.train_pilot import heuristic_action

    return heuristic_action(np.asarray(observation)[:_LANDER_NUMBERS])


def _pilot(name: str, policy: Callable[[np.ndarray], int]) -> SubSkill:
    return SubSkill(name=name, policy=policy, is_done=lambda f: f["landed"], max_frames=MAX_PILOT_FRAMES)


def landing_pilots(checkpoint: Optional[Union[str, Path]] = None) -> Dict[str, SubSkill]:
    """The heuristic and PPO landing controllers, as skills."""
    return {
        "heuristic_pilot": _pilot("heuristic_pilot", _heuristic),
        "ppo_pilot": _pilot("ppo_pilot", _CheckpointPolicy(checkpoint or PPO_CHECKPOINT)),
    }


def mission_pilots(checkpoint: Optional[Union[str, Path]] = None) -> Dict[str, SubSkill]:
    """The landing controllers and the PPO pilot trained on the mission, as skills."""
    return {
        **landing_pilots(),
        "mission_pilot": _pilot("mission_pilot", _CheckpointPolicy(checkpoint or MISSION_PPO_CHECKPOINT)),
    }

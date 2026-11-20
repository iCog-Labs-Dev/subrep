"""Safety-Gymnasium adapter for SubRep SafeRL pilots.

The wrapper maps a Safety-Gymnasium environment into the same 2-objective
contract used by the rest of SubRep:

    reward -> task payoff objective
    cost   -> safety objective, with larger values meaning safer behavior
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from gymnasium.spaces import Box
from utils.safety_objectives import OBJECTIVE_NAMES, control_effort


class SafeRLGymnasiumEnv:
    """Wrap a Safety-Gymnasium env with SubRep's 2D reward interface."""

    def __init__(
        self,
        env_id: str = "SafetyPointGoal1-v0",
        seed: int = 42,
        render_mode: Optional[str] = None,
        make_env: Optional[Callable] = None,
        include_control_efficiency: bool = False,
    ) -> None:
        self.include_control_efficiency = bool(include_control_efficiency)
        self.objective_names = OBJECTIVE_NAMES if include_control_efficiency else OBJECTIVE_NAMES[:2]
        self.num_objectives = len(self.objective_names)
        self.env_id = env_id
        self.seed = int(seed)
        self.env = self._create_env(env_id, render_mode=render_mode, make_env=make_env)
        self.env.reset(seed=self.seed)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.reward_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_objectives,),
            dtype=np.float32,
        )

    @staticmethod
    def _create_env(env_id: str, *, render_mode: Optional[str], make_env: Optional[Callable]):
        if make_env is not None:
            try:
                return make_env(env_id, render_mode=render_mode)
            except TypeError:
                return make_env(env_id)

        try:
            import safety_gymnasium
        except ImportError as exc:
            raise ImportError(
                "Safety-Gymnasium is optional. Install it in a Python 3.10 "
                "environment with: python -m pip install -r requirements-safety.txt"
            ) from exc

        if render_mode is None:
            return safety_gymnasium.make(env_id)
        return safety_gymnasium.make(env_id, render_mode=render_mode)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
            return self.env.reset(seed=self.seed)
        return self.env.reset()

    def step(self, action):
        effort = control_effort(action, self.action_space.low, self.action_space.high) if self.include_control_efficiency else None
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        reward_value = float(reward)
        cost_value = float(np.asarray(cost, dtype=np.float64).reshape(-1)[0])
        reward_vector = self._map_reward_and_cost(reward_value, cost_value)

        if self.include_control_efficiency:
            reward_vector = np.append(reward_vector, -effort).astype(np.float32)
            info["control_effort"] = effort
            info["control_efficiency_motive"] = -effort
        info["objective_names"] = list(self.objective_names)
        info["raw_objective_values"] = reward_vector.copy()
        info["objective_measurement_version"] = 1
        info["task_reward"] = reward_value
        info["safety_cost"] = cost_value
        info["safety_motive"] = float(reward_vector[0])
        info["task_motive"] = float(reward_vector[1])
        info["subrep_reward"] = reward_vector.copy()

        return obs, reward_vector, terminated, truncated, info

    @staticmethod
    def _map_reward_and_cost(reward: float, cost: float) -> np.ndarray:
        """Return motives ordered as [Safety, Task]. Larger is better."""
        safety = -float(cost)
        task = float(reward)
        return np.array([safety, task], dtype=np.float32)

    def close(self) -> None:
        self.env.close()

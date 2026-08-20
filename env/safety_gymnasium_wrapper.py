"""Safety-Gymnasium adapter for SubRep SafeRL pilots.

The wrapper maps Safety-Gymnasium reward/cost into SubRep motives. By default
it preserves the original 2-objective contract:

    [Safety, Task] = [-cost, reward]

For stronger benchmarks it can expose 3D/4D motives by adding control
efficiency and action smoothness objectives, both with larger-is-better signs.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from gymnasium.spaces import Box


class SafeRLGymnasiumEnv:
    """Wrap a Safety-Gymnasium env with SubRep's 2D reward interface."""

    def __init__(
        self,
        env_id: str = "SafetyPointGoal1-v0",
        seed: int = 42,
        render_mode: Optional[str] = None,
        make_env: Optional[Callable] = None,
        objective_mode: str = "2d",
        control_scale: float = 0.01,
        smoothness_scale: float = 0.01,
    ) -> None:
        self.env_id = env_id
        self.seed = int(seed)
        self.objective_mode = self._validate_objective_mode(objective_mode)
        self.control_scale = self._validate_scale("control_scale", control_scale)
        self.smoothness_scale = self._validate_scale("smoothness_scale", smoothness_scale)
        self.env = self._create_env(env_id, render_mode=render_mode, make_env=make_env)
        self.env.reset(seed=self.seed)
        self._previous_action: np.ndarray | None = None

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.num_objectives = {"2d": 2, "3d": 3, "4d": 4}[self.objective_mode]
        self.reward_space = Box(
            low=np.full(self.num_objectives, -np.inf, dtype=np.float32),
            high=np.full(self.num_objectives, np.inf, dtype=np.float32),
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
        self._previous_action = None
        if seed is not None:
            self.seed = int(seed)
            return self.env.reset(seed=self.seed)
        return self.env.reset()

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        reward_value = float(reward)
        cost_value = float(np.asarray(cost, dtype=np.float64).reshape(-1)[0])
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        control_cost = float(np.linalg.norm(action_array, ord=2))
        if self._previous_action is None:
            smoothness_cost = 0.0
        else:
            smoothness_cost = float(np.linalg.norm(action_array - self._previous_action, ord=2))
        self._previous_action = action_array.copy()

        reward_vector = self._map_reward_and_cost(
            reward_value,
            cost_value,
            control_cost=control_cost,
            smoothness_cost=smoothness_cost,
            objective_mode=self.objective_mode,
            control_scale=self.control_scale,
            smoothness_scale=self.smoothness_scale,
        )

        info["task_reward"] = reward_value
        info["safety_cost"] = cost_value
        info["control_cost"] = control_cost
        info["smoothness_cost"] = smoothness_cost
        info["control_scale"] = self.control_scale
        info["smoothness_scale"] = self.smoothness_scale
        info["safety_motive"] = float(reward_vector[0])
        info["task_motive"] = float(reward_vector[1])
        info["subrep_reward"] = reward_vector.copy()

        return obs, reward_vector, terminated, truncated, info

    @staticmethod
    def _validate_objective_mode(objective_mode: str) -> str:
        normalized = str(objective_mode).strip().lower()
        if normalized not in {"2d", "3d", "4d"}:
            raise ValueError(
                f"objective_mode must be one of '2d', '3d', '4d', got {objective_mode!r}"
            )
        return normalized

    @staticmethod
    def _validate_scale(name: str, value: float) -> float:
        scale = float(value)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
        return scale

    @staticmethod
    def _map_reward_and_cost(
        reward: float,
        cost: float,
        *,
        control_cost: float = 0.0,
        smoothness_cost: float = 0.0,
        objective_mode: str = "2d",
        control_scale: float = 0.01,
        smoothness_scale: float = 0.01,
    ) -> np.ndarray:
        """Return motives ordered as [Safety, Task, Control?, Smoothness?].

        Larger is better for every component, so costs are negated.
        """
        safety = -float(cost)
        task = float(reward)
        values = [safety, task]
        if objective_mode in {"3d", "4d"}:
            values.append(-float(control_scale) * float(control_cost))
        if objective_mode == "4d":
            values.append(-float(smoothness_scale) * float(smoothness_cost))
        return np.array(values, dtype=np.float32)

    def close(self) -> None:
        self.env.close()

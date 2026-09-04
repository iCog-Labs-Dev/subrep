"""A deterministic 6-objective Minecraft-shaped stub environment.

This is NOT Minecraft. It is a stand-in that exercises the full m=6
certification loop end to end while the real Minecraft/AIRIS stack does not
exist in this repository. Pure numpy: no network, no game client, no mod.

It matches the `SubRepEnv` contract (env/lunar_lander_wrapper.py:12-117) that
`IdlePolicy` and the certification pipeline already expect:
    reset(seed=None) -> (obs, info)
    step(action)     -> (obs, reward_vector, terminated, truncated, info)
    close()

Objective order is SubRep's Minecraft phi(x), per CLAUDE.md:
    [Safety, Reputation, DeadlineSlack, InventoryValue, Sustainability,
     Infrastructure]

The environment carries a `threat` level that rises and falls over an episode.
Threat degrades the Safety payoff of each action in proportion to how exposed
that action is, so options that look attractive in calm conditions become
genuinely costly under pressure. That gives a motivational governor something
real to react to rather than stationary noise.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium.spaces import Box, Discrete

OBJECTIVE_NAMES: Tuple[str, ...] = (
    "Safety",
    "Reputation",
    "DeadlineSlack",
    "InventoryValue",
    "Sustainability",
    "Infrastructure",
)

# Action archetypes, named after the paper's execution trace (doc:520-556).
SKILL_NAMES: Tuple[str, ...] = (
    "Idle",
    "TorchCorridor",
    "IronGolemSpawn",
    "ArcherKite",
    "SwingGateBarricade",
    "DiscountChain",
)

# Mean per-step payoff for each action, one row per action, columns in
# OBJECTIVE_NAMES order.
#                       Safe   Rep    Dead   Inv    Sust   Infra
_BASE_REWARDS = np.array([
    [0.00, 0.00, -0.05, 0.00, 0.00, 0.00],   # Idle
    [0.30, 0.05, -0.10, -0.10, 0.05, 0.25],  # TorchCorridor
    [0.45, 0.10, -0.20, -0.35, -0.10, 0.15],  # IronGolemSpawn
    [0.25, 0.00, 0.10, 0.05, -0.15, -0.05],  # ArcherKite
    [0.35, 0.05, -0.05, -0.15, 0.00, 0.30],  # SwingGateBarricade
    [-0.10, 0.45, 0.05, 0.20, 0.10, 0.00],   # DiscountChain
], dtype=np.float32)

# How badly rising threat hurts each action's Safety payoff. Trading while
# under attack is the most exposed thing you can do; spawning a golem is the
# least.
_THREAT_VULNERABILITY = np.array(
    [0.35, 0.20, 0.05, 0.30, 0.10, 0.60], dtype=np.float32
)

_NUM_OBJECTIVES = len(OBJECTIVE_NAMES)
_NUM_ACTIONS = len(SKILL_NAMES)
_DEFAULT_EPISODE_LENGTH = 24


class MinecraftStubEnv:
    """Six-objective stand-in for a Minecraft SubRep environment."""

    def __init__(
        self,
        seed: int = 42,
        *,
        episode_length: int = _DEFAULT_EPISODE_LENGTH,
        noise_scale: float = 0.02,
        render_mode: Optional[str] = None,
    ) -> None:
        if episode_length <= 0:
            raise ValueError(f"episode_length must be positive, got {episode_length}")
        if noise_scale < 0.0:
            raise ValueError(f"noise_scale must be non-negative, got {noise_scale}")

        self.seed = int(seed)
        self.episode_length = int(episode_length)
        self.noise_scale = float(noise_scale)
        self.render_mode = render_mode

        self.action_space = Discrete(_NUM_ACTIONS)
        self.action_space.seed(self.seed)

        # Observation: [threat, progress, 6 running motive totals].
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 + _NUM_OBJECTIVES,),
            dtype=np.float32,
        )
        self.reward_space = Box(
            low=-5.0,
            high=5.0,
            shape=(_NUM_OBJECTIVES,),
            dtype=np.float32,
        )

        # `env.env` lets this drop into code written against SubRepEnv, which
        # exposes the underlying gym env that way (demo/run_full_pipeline.py:95).
        self.env = self

        self._rng = np.random.default_rng(self.seed)
        self._t = 0
        self._threat = 0.0
        self._totals = np.zeros(_NUM_OBJECTIVES, dtype=np.float32)

    # -- helpers -----------------------------------------------------------

    @property
    def num_objectives(self) -> int:
        return _NUM_OBJECTIVES

    def _threat_at(self, t: int) -> float:
        """Threat rises to a mid-episode peak, then falls.

        Mirrors the paper's dusk -> night -> post-skirmish arc.
        """
        phase = np.pi * t / max(1, self.episode_length - 1)
        # Clip BEFORE the fractional power: sin(pi) lands slightly negative in
        # floating point, and a negative base to a fractional exponent is NaN.
        return float(np.clip(np.sin(phase), 0.0, 1.0) ** 1.5)

    def _observation(self) -> np.ndarray:
        return np.concatenate(
            [
                np.array(
                    [self._threat, self._t / self.episode_length], dtype=np.float32
                ),
                self._totals.astype(np.float32),
            ]
        ).astype(np.float32)

    # -- gym-style API -----------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the episode. Passing a seed makes the episode reproducible."""
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.default_rng(self.seed)
            self.action_space.seed(self.seed)

        self._t = 0
        self._threat = self._threat_at(0)
        self._totals = np.zeros(_NUM_OBJECTIVES, dtype=np.float32)

        info = {"threat": self._threat, "skill_names": SKILL_NAMES}
        return self._observation(), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        """Apply one action and return the 6-objective reward vector."""
        action = int(action)
        if not (0 <= action < _NUM_ACTIONS):
            raise ValueError(
                f"action must be in [0, {_NUM_ACTIONS - 1}], got {action}"
            )

        reward = _BASE_REWARDS[action].copy()

        # Threat erodes Safety in proportion to how exposed the action is.
        reward[0] -= self._threat * _THREAT_VULNERABILITY[action]

        # Sustained threat also depresses Reputation: villagers scatter.
        reward[1] -= 0.15 * self._threat

        if self.noise_scale > 0.0:
            reward = reward + self._rng.normal(
                0.0, self.noise_scale, size=_NUM_OBJECTIVES
            ).astype(np.float32)

        reward = reward.astype(np.float32)
        self._totals += reward

        self._t += 1
        self._threat = self._threat_at(self._t)
        terminated = False
        truncated = self._t >= self.episode_length

        info: Dict[str, Any] = {
            "threat": self._threat,
            "skill_name": SKILL_NAMES[action],
            "step": self._t,
            "subrep_reward": reward.copy(),
            "motive_totals": self._totals.copy(),
        }

        if reward.shape != (_NUM_OBJECTIVES,):
            raise ValueError(
                f"Reward vector shape mismatch: expected ({_NUM_OBJECTIVES},), "
                f"got {reward.shape}"
            )

        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        """No resources to release; present for API parity."""
        return None

"""Deterministic idle baseline policy for SubRep certification."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class IdlePolicy:
    """A simple baseline policy that always selects the idle action."""

    def __init__(self, env: Any, idle_action: int = 0, gamma: float = 0.99) -> None:
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")

        self.env = env
        self.idle_action = int(idle_action)
        self.gamma = float(gamma)

    def get_action(self, obs) -> int:
        """Return the deterministic do-nothing action."""
        return self.idle_action

    def run_baseline_episodes(self, num_episodes: int = 50, seed: int = 42) -> Dict[str, Any]:
        """Run baseline episodes and aggregate discounted payoff/motive statistics."""
        if num_episodes <= 0:
            raise ValueError("num_episodes must be positive")

        episode_payoffs: List[float] = []
        episode_motives: List[np.ndarray] = []

        for episode_index in range(num_episodes):
            obs, _ = self._reset_env(seed + episode_index)
            discount = 1.0
            total_payoff = 0.0
            motive_deltas = None

            while True:
                action = self.get_action(obs)
                obs, reward_vec, terminated, truncated, _ = self.env.step(action)
                reward_vec = np.asarray(reward_vec, dtype=np.float32)

                if reward_vec.ndim != 1:
                    raise ValueError(f"reward vector must be 1D, got shape {reward_vec.shape}")

                if motive_deltas is None:
                    motive_deltas = np.zeros_like(reward_vec, dtype=np.float32)

                total_payoff += discount * float(np.sum(reward_vec))
                motive_deltas += discount * reward_vec

                if terminated or truncated:
                    break

                discount *= self.gamma

            episode_payoffs.append(float(total_payoff))
            episode_motives.append(np.asarray(motive_deltas, dtype=np.float32))

        payoff_array = np.asarray(episode_payoffs, dtype=np.float32)
        motive_array = np.asarray(episode_motives, dtype=np.float32)

        baseline_payoff = float(np.mean(payoff_array))
        baseline_motives = np.mean(motive_array, axis=0).astype(np.float32)

        return {
            "baseline_payoff": baseline_payoff,
            "baseline_motives": baseline_motives,
            "episode_payoffs": payoff_array,
            "episode_motives": motive_array,
            "payoff_std": float(np.std(payoff_array)),
            "motives_std": np.std(motive_array, axis=0).astype(np.float32),
            "num_episodes": int(num_episodes),
            "seed": int(seed),
            "gamma": float(self.gamma),
            "idle_action": int(self.idle_action),
        }

    def _reset_env(self, seed: int):
        """Reset the environment, seeding it when the API supports it."""
        try:
            return self.env.reset(seed=seed)
        except TypeError:
            if hasattr(self.env, "seed"):
                try:
                    seed_attr = getattr(self.env, "seed")
                    if callable(seed_attr):
                        seed_attr(seed)
                    else:
                        self.env.seed = seed
                except Exception:
                    pass
            return self.env.reset()
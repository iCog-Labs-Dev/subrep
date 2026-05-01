"""
Skill execution loop for SubRep.

Runs a policy in the wrapped MO-LunarLander environment and collects
discounted rollout summaries needed by later certification stages.
"""

from __future__ import annotations # Used for forward type references in Python 3.7+ without string literals.
from typing import Callable, Optional
import numpy as np

class SkillExecutor:
    """
    Execute one rollout and summarize discounted outcomes.
    - `total_payoff` defaults to discounted sum of reward_vector.sum().
    This scalarization is a practical default and can be overridden via
      `payoff_fn` without changing the return format.
    """

    def __init__(
        self,
        env,
        policy_fn: Callable[[np.ndarray], int],
        gamma: float = 0.99,
        max_steps: Optional[int] = None,
        payoff_fn: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        """
        Initialize executor configuration.
        Args:
            env: Environment instance exposing reset() and step().
            policy_fn: Callable mapping observation -> action.
            gamma: Discount factor used for payoff and motive totals.
            max_steps: Optional rollout cap. If None, run until env ends.
            payoff_fn: Optional scalarization function for 2D reward vectors.
        """
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")

        self.env = env
        self.policy_fn = policy_fn
        self.gamma = gamma
        self.max_steps = max_steps
        self.payoff_fn = payoff_fn or (lambda reward_vec: float(np.sum(reward_vec)))
        self.last_run_info = None         # Holds diagnostics from the most recent run for downstream debugging.

    def run_episode(self):
        """
        Run one rollout from reset and return:
        (total_payoff, motive_deltas, terminated)
        """
        obs, _ = self.env.reset()
        initial_obs = np.array(obs, copy=True)
        total_payoff = 0.0
        motive_deltas = np.zeros(2, dtype=np.float32)
        discount = 1.0
        steps = 0
        terminated = False
        truncated = False
        final_reward = np.zeros(2, dtype=np.float32)
        stop_reason = "unknown"

        while True:
            if self.max_steps is not None and steps >= self.max_steps:
                stop_reason = "max_steps"
                break

            # Query action from caller-provided policy.
            action = self.policy_fn(obs)
            obs, reward_vec, terminated, truncated, _ = self.env.step(action)
            reward_vec = np.asarray(reward_vec, dtype=np.float32)

            # Apply discounting to both scalar payoff and 2D motive totals.
            total_payoff += discount * float(self.payoff_fn(reward_vec))
            motive_deltas += discount * reward_vec
            final_reward = reward_vec
            steps += 1

            # Stop conditions: true terminal state, timeout truncation, or external max-step cap.
            if terminated:
                stop_reason = "terminated"
                break
            if truncated:
                stop_reason = "truncated"
                break

            discount *= self.gamma

        # Console summary required by task spec.
        print("Episode summary:")
        print(f"  steps: {steps}")
        print(f"  total_payoff: {total_payoff:.6f}")
        print(f"  motive_deltas: {motive_deltas}")
        print(f"  final_reward: {final_reward}")
        print(f"  end_reason: {stop_reason}")

        # Extra run metadata kept without changing public return tuple.
        self.last_run_info = {
            "initial_obs": initial_obs,
            "steps": steps,
            "truncated": bool(truncated),
            "stop_reason": stop_reason,
            "final_reward": final_reward.copy(),
            "gamma": float(self.gamma),
            "max_steps": self.max_steps,
        }

        return float(total_payoff), motive_deltas, bool(terminated)
"""
Skill execution loop for SubRep.

Runs a policy in the wrapped MO-LunarLander environment and collects
discounted rollout summaries needed by later certification stages.
"""

from __future__ import annotations # Used for forward type references in Python 3.7+ without string literals.
import copy
from typing import Callable, Optional
import warnings
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
            payoff_fn: Optional scalarization function for reward vectors.
        """
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")

        self.env = env
        self.policy_fn = policy_fn
        self.gamma = gamma
        self.max_steps = max_steps
        self.payoff_fn = payoff_fn or (lambda reward_vec: float(np.sum(reward_vec)))
        self.last_run_info = None         # Holds diagnostics from the most recent run for downstream debugging.

    @classmethod
    def from_pilot_checkpoint(
        cls,
        env,
        checkpoint_path: str = "models/pilot_ppo.pt",
        gamma: float = 0.99,
        max_steps: Optional[int] = None,
        payoff_fn: Optional[Callable[[np.ndarray], float]] = None,
        deterministic: bool = True,
        map_location: str = "cpu",
    ) -> "SkillExecutor":
        """Create an executor backed by a saved RLPilot checkpoint."""
        from pilot.rl_pilot import RLPilot

        pilot = RLPilot.load(checkpoint_path, map_location=map_location)

        def policy_fn(obs):
            # SkillExecutor already supports arbitrary policy callables. This
            # adapter keeps that contract intact while replacing random action
            # sampling with the trained PPO pilot for certification rollouts.
            return pilot.predict(
                obs,
                deterministic=deterministic,
                return_probability=True,
            )

        executor = cls(
            env=env,
            policy_fn=policy_fn,
            gamma=gamma,
            max_steps=max_steps,
            payoff_fn=payoff_fn,
        )
        executor.pilot = pilot
        executor.pilot_checkpoint_path = checkpoint_path
        return executor

    def run_episode(self, initial_obs: Optional[np.ndarray] = None):
        """
        Run one rollout and return (total_payoff, motive_deltas, terminated).
        Args:
            initial_obs: If provided, skip env.reset() and start from this state.
        """
        # Determine fallback motive dimension if available
        fallback_dim = 2
        if hasattr(self.env, "metadata") and isinstance(self.env.metadata, dict):
            m_names = self.env.metadata.get("motive_names")
            if isinstance(m_names, (list, tuple)) and len(m_names) > 0:
                fallback_dim = len(m_names)
        elif hasattr(self.env, "reward_space") and hasattr(self.env.reward_space, "shape"):
            if self.env.reward_space.shape:
                fallback_dim = self.env.reward_space.shape[0]

        if initial_obs is not None:
            # Use provided observation instead of resetting.
            obs = np.array(initial_obs, copy=True) if isinstance(initial_obs, (np.ndarray, list, tuple)) else copy.copy(initial_obs)
        else:
            reset_out = self.env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, _ = reset_out
            else:
                obs = reset_out

        initial_obs = np.array(obs, copy=True) if isinstance(obs, (np.ndarray, list, tuple)) else copy.copy(obs)
        total_payoff = 0.0
        motive_deltas = None
        discount = 1.0
        steps = 0
        terminated = False
        truncated = False
        final_reward = None
        stop_reason = "unknown"
        behavior_probability = None

        while True:
            if self.max_steps is not None and steps >= self.max_steps:
                stop_reason = "max_steps"
                break

            # Query action from caller-provided policy.
            action_output = self.policy_fn(obs)
            action, behavior_probability = self._parse_policy_output(action_output)
            step_out = self.env.step(action)
            if len(step_out) == 5:
                obs, reward_vec, terminated, truncated, info = step_out
            elif len(step_out) == 4:
                obs, reward_vec, terminated, info = step_out
                truncated = False
            else:
                raise ValueError(f"Expected step() to return 4 or 5 elements, got {len(step_out)}")

            info = dict(info) if isinstance(info, dict) else {}
            reward_vec = np.asarray(reward_vec, dtype=np.float32)

            if motive_deltas is None:
                motive_deltas = np.zeros_like(reward_vec, dtype=np.float32)

            # Preference: task_payoff in info, fallback to payoff_fn(reward_vec)
            if "task_payoff" in info:
                step_payoff = float(info["task_payoff"])
            else:
                warnings.warn(
                    "Environment step info missing 'task_payoff'; falling back to payoff_fn",
                    RuntimeWarning,
                    stacklevel=2,
                )
                step_payoff = float(self.payoff_fn(reward_vec))

            total_payoff += discount * step_payoff
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

        if motive_deltas is None:
            motive_deltas = np.zeros(fallback_dim, dtype=np.float32)
        if final_reward is None:
            final_reward = np.zeros(fallback_dim, dtype=np.float32)

        motive_names = None
        if hasattr(self.env, "metadata") and isinstance(self.env.metadata, dict):
            motive_names = self.env.metadata.get("motive_names")

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
            "behavior_probability": behavior_probability,
            "motive_names": motive_names,
        }

        return float(total_payoff), motive_deltas, bool(terminated)

    @staticmethod
    def _parse_policy_output(action_output):
        """Allow policies to optionally return the chosen-action probability.
        """
        if isinstance(action_output, tuple):
            if len(action_output) != 2:
                raise ValueError("policy_fn tuple output must be (action, behavior_probability)")
            action, behavior_probability = action_output
            if behavior_probability is None:
                return action, None
            behavior_probability = float(behavior_probability)
            if not np.isfinite(behavior_probability) or behavior_probability <= 0.0 or behavior_probability > 1.0:
                raise ValueError(
                    f"behavior_probability must be finite and lie in (0, 1], got {behavior_probability}"
                )
            return action, behavior_probability
        return action_output, None

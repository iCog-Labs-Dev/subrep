"""Collect SafeRL candidate rollouts for SubRep certification pilots.

Usage:
    python -m data_collector.collect_safety_gymnasium_rollouts \
      --contexts 25 --env-id SafetyPointGoal1-v0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from env.safety_gymnasium_wrapper import SafeRLGymnasiumEnv
from env.skill_executor import SkillExecutor
from pilot.safety_gymnasium_ppo import SafetyPPOPilot


PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class SafetyCandidatePolicy:
    skill_id: str
    policy_fn: PolicyFn


def _zero_policy(action_space) -> PolicyFn:
    def policy_fn(_obs: np.ndarray) -> np.ndarray:
        return np.zeros(action_space.shape, dtype=np.float32)

    return policy_fn


def _random_policy(action_space, *, seed: int) -> PolicyFn:
    action_space.seed(seed)

    def policy_fn(_obs: np.ndarray) -> np.ndarray:
        return np.asarray(action_space.sample(), dtype=np.float32)

    return policy_fn


def _scaled_random_policy(action_space, *, seed: int, scale: float) -> PolicyFn:
    rng = np.random.default_rng(seed)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)

    def policy_fn(_obs: np.ndarray) -> np.ndarray:
        raw = rng.uniform(low=low, high=high).astype(np.float32)
        return np.clip(scale * raw, low, high).astype(np.float32)

    return policy_fn


def _constant_axis_policy(action_space, *, axis: int, value: float) -> PolicyFn:
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)

    def policy_fn(_obs: np.ndarray) -> np.ndarray:
        action = np.zeros(action_space.shape, dtype=np.float32)
        if action.size:
            flat = action.reshape(-1)
            flat[int(axis) % len(flat)] = float(value)
        return np.clip(action, low, high).astype(np.float32)

    return policy_fn


def _ppo_policy(checkpoint_path: str | Path) -> PolicyFn:
    pilot = SafetyPPOPilot.load(checkpoint_path, map_location="cpu")

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        return pilot.predict(obs, deterministic=True)

    return policy_fn


def build_default_safety_candidate_policies(
    env: SafeRLGymnasiumEnv,
    *,
    ppo_checkpoint: str | Path | None = None,
    ppo_lagrangian_checkpoint: str | Path | None = None,
) -> tuple[SafetyCandidatePolicy, ...]:
    """Return simple continuous-control candidates for a first SafeRL pilot."""
    action_space = env.action_space
    policies = [
        SafetyCandidatePolicy("zero_action", _zero_policy(action_space)),
        SafetyCandidatePolicy("small_random", _scaled_random_policy(action_space, seed=17, scale=0.25)),
        SafetyCandidatePolicy("random", _random_policy(action_space, seed=23)),
        SafetyCandidatePolicy("axis0_positive", _constant_axis_policy(action_space, axis=0, value=0.35)),
        SafetyCandidatePolicy("axis0_negative", _constant_axis_policy(action_space, axis=0, value=-0.35)),
        SafetyCandidatePolicy("axis1_positive", _constant_axis_policy(action_space, axis=1, value=0.35)),
        SafetyCandidatePolicy("axis1_negative", _constant_axis_policy(action_space, axis=1, value=-0.35)),
    ]
    if ppo_checkpoint is not None:
        checkpoint = Path(ppo_checkpoint)
        if checkpoint.exists():
            policies.append(SafetyCandidatePolicy("ppo_deterministic", _ppo_policy(checkpoint)))
        else:
            print(
                f"[Warning] PPO checkpoint not found: {checkpoint}. Skipping PPO candidate.",
                flush=True,
            )
    if ppo_lagrangian_checkpoint is not None:
        checkpoint = Path(ppo_lagrangian_checkpoint)
        if checkpoint.exists():
            policies.append(
                SafetyCandidatePolicy(
                    "ppo_lagrangian_deterministic",
                    _ppo_policy(checkpoint),
                )
            )
        else:
            print(
                f"[Warning] PPO-Lagrangian checkpoint not found: {checkpoint}. "
                "Skipping PPO-Lagrangian candidate.",
                flush=True,
            )
    return tuple(policies)


class SafetyGymnasiumRolloutCollector:
    """Collect same-context SafeRL candidate outcomes."""

    def __init__(
        self,
        *,
        env_id: str = "SafetyPointGoal1-v0",
        seed: int = 42,
        save_dir: str = "data/safety_gymnasium_rollouts",
        max_steps: int = 200,
        gamma: float = 0.99,
        ppo_checkpoint: str | Path | None = None,
        ppo_lagrangian_checkpoint: str | Path | None = None,
        env_factory: Optional[Callable[..., SafeRLGymnasiumEnv]] = None,
    ) -> None:
        self.env_id = env_id
        self.seed = int(seed)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = int(max_steps)
        self.gamma = float(gamma)
        self.ppo_checkpoint = ppo_checkpoint
        self.ppo_lagrangian_checkpoint = ppo_lagrangian_checkpoint
        factory = env_factory or SafeRLGymnasiumEnv
        self.env = factory(env_id=env_id, seed=seed)
        self.candidate_policies = build_default_safety_candidate_policies(
            self.env,
            ppo_checkpoint=ppo_checkpoint,
            ppo_lagrangian_checkpoint=ppo_lagrangian_checkpoint,
        )

    def collect_context(self, context_index: int) -> dict[str, object]:
        context_seed = self.seed + int(context_index)
        context, _ = self.env.reset(seed=context_seed)
        context = np.asarray(context, dtype=np.float32)

        skill_ids: list[str] = []
        payoffs: list[float] = []
        motives: list[np.ndarray] = []
        costs: list[float] = []
        task_returns: list[float] = []
        terminated_flags: list[bool] = []
        step_counts: list[int] = []
        stop_reasons: list[str] = []

        for candidate in self.candidate_policies:
            obs, _ = self.env.reset(seed=context_seed)
            obs = np.asarray(obs, dtype=np.float32)
            if not np.allclose(obs, context, rtol=1e-6, atol=1e-6):
                raise RuntimeError("Reset seed did not reproduce the SafeRL context")

            executor = SkillExecutor(
                env=self.env,
                policy_fn=candidate.policy_fn,
                gamma=self.gamma,
                max_steps=self.max_steps,
                payoff_fn=lambda reward_vec: float(np.asarray(reward_vec, dtype=np.float32)[1]),
            )
            payoff, motive_values, terminated = executor.run_episode(initial_obs=obs)
            info = executor.last_run_info or {}
            motive_array = np.asarray(motive_values, dtype=np.float32).reshape(-1)

            skill_ids.append(candidate.skill_id)
            payoffs.append(float(payoff))
            motives.append(motive_array)
            costs.append(float(-motive_array[0]))
            task_returns.append(float(motive_array[1]))
            terminated_flags.append(bool(terminated))
            step_counts.append(int(info.get("steps", 0)))
            stop_reasons.append(str(info.get("stop_reason", "unknown")))

        return {
            "env_id": np.asarray(self.env_id),
            "context": context,
            "context_seed": np.asarray(context_seed, dtype=np.int32),
            "candidate_skill_ids": np.asarray(skill_ids),
            "candidate_payoffs": np.asarray(payoffs, dtype=np.float32),
            "candidate_motives": np.stack(motives, axis=0).astype(np.float32),
            "candidate_safety_costs": np.asarray(costs, dtype=np.float32),
            "candidate_task_returns": np.asarray(task_returns, dtype=np.float32),
            "terminated_flags": np.asarray(terminated_flags, dtype=np.bool_),
            "step_counts": np.asarray(step_counts, dtype=np.int32),
            "stop_reasons": np.asarray(stop_reasons),
        }

    def save_context(self, record: dict[str, object], context_index: int, prefix: str = "safety_rollout") -> str:
        path = self.save_dir / f"{prefix}_{context_index:05d}.npz"
        np.savez(path, **record)
        return str(path)

    def collect(self, contexts: int, *, prefix: str = "safety_rollout") -> list[dict[str, object]]:
        if int(contexts) <= 0:
            raise ValueError("contexts must be positive")
        records: list[dict[str, object]] = []
        for index in range(1, int(contexts) + 1):
            record = self.collect_context(index)
            self.save_context(record, index, prefix=prefix)
            records.append(record)
            print(
                f"[{index:05d}/{contexts:05d}] saved {len(record['candidate_skill_ids'])} "
                f"SafeRL candidate outcomes for seed {record['context_seed']}",
                flush=True,
            )
        return records

    def close(self) -> None:
        self.env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Safety-Gymnasium candidate rollouts.")
    parser.add_argument("--contexts", type=int, default=25)
    parser.add_argument("--env-id", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument("--save-dir", type=str, default="data/safety_gymnasium_rollouts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="safety_rollout")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument(
        "--ppo-checkpoint",
        type=str,
        default=None,
        help="Optional SafetyPPOPilot checkpoint to include as ppo_deterministic candidate.",
    )
    parser.add_argument(
        "--ppo-lagrangian-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional SafetyPPOPilot checkpoint trained with Lagrangian cost "
            "updates to include as ppo_lagrangian_deterministic candidate."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collector = SafetyGymnasiumRolloutCollector(
        env_id=args.env_id,
        seed=args.seed,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        gamma=args.gamma,
        ppo_checkpoint=args.ppo_checkpoint,
        ppo_lagrangian_checkpoint=args.ppo_lagrangian_checkpoint,
    )
    try:
        collector.collect(args.contexts, prefix=args.prefix)
    finally:
        collector.close()
    print(f"[Done] Safety-Gymnasium rollout files saved to '{args.save_dir}/'", flush=True)


if __name__ == "__main__":
    main()

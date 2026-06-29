"""Collect same-context candidate outcome sets for MDN training.

Usage:
    python -m data_collector.collect_candidate_sets --contexts 500 --save-dir data/mdn_candidate_sets --seed 42

Each saved file contains one starting context and the actual rollout outcomes
for several candidate policies from that same reset seed. This is stronger MDN
training data than isolated rollouts because the trainer can compare candidate
skills under the same context.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor
from pilot.rl_pilot import RLPilot


PolicyFn = Callable[[np.ndarray], int | tuple[int, float]]


@dataclass(frozen=True)
class CandidatePolicy:
    skill_id: str
    policy_fn: PolicyFn


def _fixed_action_policy(action: int) -> PolicyFn:
    def policy_fn(_obs: np.ndarray) -> tuple[int, float]:
        return int(action), 1.0

    return policy_fn


def _random_policy(env: SubRepEnv) -> PolicyFn:
    action_count = int(env.env.action_space.n)

    def policy_fn(_obs: np.ndarray) -> tuple[int, float]:
        return int(env.env.action_space.sample()), 1.0 / float(action_count)

    return policy_fn


def build_default_candidate_policies(
    env: SubRepEnv,
    *,
    pilot_checkpoint: str = "models/pilot_ppo.pt",
    map_location: str = "cpu",
) -> tuple[CandidatePolicy, ...]:
    """Build a small diverse candidate set for LunarLander MDN data collection."""
    pilot = RLPilot.load(pilot_checkpoint, map_location=map_location)
    return (
        CandidatePolicy(
            "ppo_deterministic",
            lambda obs, _pilot=pilot: _pilot.predict(obs, deterministic=True, return_probability=True),
        ),
        CandidatePolicy(
            "ppo_stochastic",
            lambda obs, _pilot=pilot: _pilot.predict(obs, deterministic=False, return_probability=True),
        ),
        CandidatePolicy("noop", _fixed_action_policy(0)),
        CandidatePolicy("left_engine", _fixed_action_policy(1)),
        CandidatePolicy("main_engine", _fixed_action_policy(2)),
        CandidatePolicy("right_engine", _fixed_action_policy(3)),
        CandidatePolicy("random", _random_policy(env)),
    )


class CandidateSetCollector:
    """Collect candidate outcomes from identical reset contexts."""

    def __init__(
        self,
        *,
        seed: int = 42,
        save_dir: str = "data/mdn_candidate_sets",
        max_steps: int | None = 200,
        gamma: float = 0.99,
        pilot_checkpoint: str = "models/pilot_ppo.pt",
        map_location: str = "cpu",
    ) -> None:
        self.seed = int(seed)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.gamma = float(gamma)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        self.env = SubRepEnv(seed=self.seed)
        self.env.env.action_space.seed(self.seed)
        self.candidate_policies = build_default_candidate_policies(
            self.env,
            pilot_checkpoint=pilot_checkpoint,
            map_location=map_location,
        )

    def collect_context(self, context_index: int) -> dict[str, object]:
        context_seed = self.seed + int(context_index)
        context, _ = self.env.reset(seed=context_seed)
        context = np.asarray(context, dtype=np.float32)

        skill_ids: list[str] = []
        payoffs: list[float] = []
        motives: list[np.ndarray] = []
        terminated_flags: list[bool] = []
        behavior_probabilities: list[float] = []
        step_counts: list[int] = []
        stop_reasons: list[str] = []

        for candidate in self.candidate_policies:
            obs, _ = self.env.reset(seed=context_seed)
            obs = np.asarray(obs, dtype=np.float32)
            if not np.allclose(obs, context, rtol=1e-6, atol=1e-6):
                raise RuntimeError("Reset seed did not reproduce the candidate-set context")

            executor = SkillExecutor(
                env=self.env,
                policy_fn=candidate.policy_fn,
                gamma=self.gamma,
                max_steps=self.max_steps,
            )
            payoff, motive_values, terminated = executor.run_episode(initial_obs=obs)
            info = executor.last_run_info or {}
            behavior_probability = info.get("behavior_probability")

            skill_ids.append(candidate.skill_id)
            payoffs.append(float(payoff))
            motives.append(np.asarray(motive_values, dtype=np.float32).reshape(-1))
            terminated_flags.append(bool(terminated))
            behavior_probabilities.append(
                float(behavior_probability) if behavior_probability is not None else np.nan
            )
            step_counts.append(int(info.get("steps", 0)))
            stop_reasons.append(str(info.get("stop_reason", "unknown")))

        return {
            "context": context,
            "context_seed": context_seed,
            "candidate_skill_ids": np.asarray(skill_ids),
            "candidate_payoffs": np.asarray(payoffs, dtype=np.float32),
            "candidate_motives": np.stack(motives, axis=0).astype(np.float32),
            "terminated_flags": np.asarray(terminated_flags, dtype=np.bool_),
            "behavior_probabilities": np.asarray(behavior_probabilities, dtype=np.float32),
            "step_counts": np.asarray(step_counts, dtype=np.int32),
            "stop_reasons": np.asarray(stop_reasons),
        }

    def save_context(self, record: dict[str, object], context_index: int, prefix: str = "candidate_set") -> str:
        path = self.save_dir / f"{prefix}_{context_index:05d}.npz"
        np.savez(path, **record)
        return str(path)

    def collect(self, contexts: int, *, prefix: str = "candidate_set") -> list[dict[str, object]]:
        if contexts <= 0:
            raise ValueError("contexts must be positive")

        records: list[dict[str, object]] = []
        for index in range(1, int(contexts) + 1):
            record = self.collect_context(index)
            self.save_context(record, index, prefix=prefix)
            records.append(record)
            print(
                f"[{index:05d}/{contexts:05d}] saved {len(record['candidate_skill_ids'])} "
                f"candidate outcomes for seed {record['context_seed']}"
            )
        return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect same-context candidate sets for MDN training.")
    parser.add_argument("--contexts", type=int, default=500, help="Number of starting contexts to collect")
    parser.add_argument("--save-dir", type=str, default="data/mdn_candidate_sets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="candidate_set")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--pilot-checkpoint", type=str, default="models/pilot_ppo.pt")
    parser.add_argument("--map-location", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collector = CandidateSetCollector(
        seed=args.seed,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        gamma=args.gamma,
        pilot_checkpoint=args.pilot_checkpoint,
        map_location=args.map_location,
    )
    collector.collect(args.contexts, prefix=args.prefix)
    print(f"[Done] Candidate-set files saved to '{args.save_dir}/'")


if __name__ == "__main__":
    main()

"""Collect real probability-aware runtime logs for IPS/DR MDN training.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baseline.idle_policy import IdlePolicy
from data_collector.collect_candidate_sets import CandidatePolicy, build_default_candidate_policies
from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor
from utils.mdn_checkpoint_loader import load_mdn_checkpoint
from utils.mdn_record_builder import build_candidate_skill_records
from utils.mdn_selection import alpha_to_mean_weights, score_candidate
from utils.probability_aware_logs import (
    epsilon_softmax_candidate_probabilities,
    sample_candidate_index,
    save_probability_aware_log,
    serialize_candidate_records,
)


@dataclass(frozen=True)
class BehaviorPolicyConfig:
    """Configuration for candidate-level runtime logging behavior."""

    epsilon: float = 0.2
    temperature: float = 1.0
    weights: tuple[float, float] = (0.5, 0.5)
    mdn_checkpoint: str | None = None


class ProbabilityAwareRuntimeLogCollector:
    """Collect probability-aware logged decisions from real environment rollouts."""

    def __init__(
        self,
        *,
        seed: int = 42,
        save_dir: str = "data/mdn_probability_aware_logs",
        max_steps: int | None = 200,
        gamma: float = 0.99,
        baseline_episodes: int = 20,
        pilot_checkpoint: str = "models/pilot_ppo.pt",
        map_location: str = "cpu",
        behavior_config: BehaviorPolicyConfig | None = None,
    ) -> None:
        self.seed = int(seed)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.gamma = float(gamma)
        self.baseline_episodes = int(baseline_episodes)
        self.rng = np.random.default_rng(self.seed)
        self.behavior_config = behavior_config or BehaviorPolicyConfig()

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.env = SubRepEnv(seed=self.seed)
        self.env.env.action_space.seed(self.seed)
        self.baseline_stats = IdlePolicy(env=self.env, gamma=self.gamma).run_baseline_episodes(
            num_episodes=self.baseline_episodes,
            seed=self.seed,
        )
        self.candidate_policies = build_default_candidate_policies(
            self.env,
            pilot_checkpoint=pilot_checkpoint,
            map_location=map_location,
        )
        self.behavior_model = None
        if self.behavior_config.mdn_checkpoint is not None:
            self.behavior_model = load_mdn_checkpoint(self.behavior_config.mdn_checkpoint, map_location="cpu")
            self.behavior_model.eval()

    def collect_decision(self, context_index: int) -> dict[str, Any] | None:
        """Collect one logged decision; return None when no candidate certifies."""
        context_seed = self.seed + int(context_index)
        context, _ = self.env.reset(seed=context_seed)
        context = np.asarray(context, dtype=np.float32)

        outcomes = self._evaluate_candidates(context=context, context_seed=context_seed)
        candidate_records = build_candidate_skill_records(
            skill_outcomes=outcomes,
            baseline_stats=self.baseline_stats,
        )
        if not any(candidate.is_certified for candidate in candidate_records):
            return None

        alpha, support_values, weights = self._behavior_weights(context)
        probabilities = epsilon_softmax_candidate_probabilities(
            candidate_records,
            weights,
            epsilon=self.behavior_config.epsilon,
            temperature=self.behavior_config.temperature,
        )
        selected_index, behavior_probability = sample_candidate_index(probabilities, rng=self.rng)
        selected_candidate = candidate_records[selected_index]
        selected_outcome = outcomes[selected_index]
        selected_score = score_candidate(selected_candidate, weights)

        metadata = {
            "collector_seed": self.seed,
            "context_seed": context_seed,
            "gamma": self.gamma,
            "baseline_episodes": self.baseline_episodes,
            "environment": "MO-LunarLander-v3",
            "behavior_policy": "epsilon_softmax_certified_candidates",
            "behavior_epsilon": self.behavior_config.epsilon,
            "behavior_temperature": self.behavior_config.temperature,
            "behavior_weight_source": "mdn_checkpoint" if self.behavior_model is not None else "fixed_weights",
            "behavior_mdn_checkpoint": self.behavior_config.mdn_checkpoint,
        }
        record = {
            "context": context,
            "alpha": alpha,
            "support_values": support_values,
            "weights_used": weights,
            "selected_candidate_index": np.asarray(selected_index, dtype=np.int32),
            "selected_skill_id": np.asarray(selected_candidate.skill_id),
            "selected_score": np.asarray(selected_score, dtype=np.float32),
            "behavior_probability": np.asarray(behavior_probability, dtype=np.float32),
            "actual_payoff": np.asarray(selected_outcome["payoff"], dtype=np.float32),
            "actual_motives": np.asarray(selected_outcome["motives"], dtype=np.float32),
            "behavior_probabilities": self._probability_array(len(candidate_records), probabilities),
            "metadata": metadata,
        }
        record.update(serialize_candidate_records(candidate_records))
        return record

    def save_decision(self, record: dict[str, Any], context_index: int, *, prefix: str = "runtime_log") -> str:
        path = self.save_dir / f"{prefix}_{context_index:05d}.npz"
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing runtime log {path}. "
                "Use --resume to continue an interrupted collection."
            )
        return save_probability_aware_log(path, **record)

    def collect(
        self,
        decisions: int,
        *,
        prefix: str = "runtime_log",
        max_attempts: int | None = None,
        resume: bool = False,
    ) -> list[str]:
        if decisions <= 0:
            raise ValueError("decisions must be positive")

        existing_count = 0
        next_save_index = 1
        next_context_index = 1
        if resume:
            existing_count, next_save_index, next_context_index = self._resume_state(prefix=prefix)
            if existing_count >= int(decisions):
                print(
                    f"[Done] Found {existing_count} existing logs for prefix {prefix!r}; "
                    f"target is {decisions}."
                )
                return []

        remaining_decisions = int(decisions) - existing_count
        max_attempts = int(max_attempts or remaining_decisions * 5)
        if max_attempts < remaining_decisions:
            raise ValueError("max_attempts must be at least the remaining decisions")

        saved_paths: list[str] = []
        attempts = 0
        while len(saved_paths) < remaining_decisions and attempts < max_attempts:
            attempts += 1
            context_index = next_context_index + attempts - 1
            record = self.collect_decision(context_index)
            if record is None:
                print(f"[{context_index:05d}] skipped context with no certified candidates")
                continue
            save_index = next_save_index + len(saved_paths)
            path = self.save_decision(record, save_index, prefix=prefix)
            saved_paths.append(path)
            total_saved = existing_count + len(saved_paths)
            print(
                f"[{total_saved:05d}/{decisions:05d}] saved logged decision "
                f"from context attempt {context_index:05d}"
            )

        if len(saved_paths) < remaining_decisions:
            raise RuntimeError(
                f"Collected {len(saved_paths)} additional decisions after {attempts} attempts; "
                f"increase max_attempts or inspect candidate certification rate"
            )
        return saved_paths

    def _resume_state(self, *, prefix: str) -> tuple[int, int, int]:
        files = sorted(self.save_dir.glob(f"{prefix}_*.npz"))
        if not files:
            return 0, 1, 1

        max_file_index = 0
        max_context_index = 0
        for path in files:
            try:
                max_file_index = max(max_file_index, int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
            try:
                data = np.load(path, allow_pickle=True)
                metadata_json = data.get("metadata_json")
                if metadata_json is None:
                    continue
                metadata = json.loads(str(np.asarray(metadata_json).reshape(()).item()))
                context_seed = metadata.get("context_seed")
                if context_seed is not None:
                    max_context_index = max(max_context_index, int(context_seed) - self.seed)
            except (OSError, ValueError, json.JSONDecodeError):
                continue

        next_save_index = max_file_index + 1
        next_context_index = max(max_context_index, max_file_index) + 1
        print(
            f"[Resume] Found {len(files)} existing logs; next file index "
            f"{next_save_index:05d}, next context attempt {next_context_index:05d}."
        )
        return len(files), next_save_index, next_context_index

    def _evaluate_candidates(self, *, context: np.ndarray, context_seed: int) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        for candidate in self.candidate_policies:
            outcomes.append(self._evaluate_candidate(candidate, context=context, context_seed=context_seed))
        return outcomes

    def _evaluate_candidate(
        self,
        candidate: CandidatePolicy,
        *,
        context: np.ndarray,
        context_seed: int,
    ) -> dict[str, Any]:
        obs, _ = self.env.reset(seed=context_seed)
        obs = np.asarray(obs, dtype=np.float32)
        if not np.allclose(obs, context, rtol=1e-6, atol=1e-6):
            raise RuntimeError("Reset seed did not reproduce the runtime logging context")
        executor = SkillExecutor(
            env=self.env,
            policy_fn=candidate.policy_fn,
            gamma=self.gamma,
            max_steps=self.max_steps,
        )
        payoff, motives, terminated = executor.run_episode(initial_obs=obs)
        info = executor.last_run_info or {}
        return {
            "context": context,
            "skill_id": candidate.skill_id,
            "payoff": float(payoff),
            "motives": tuple(float(v) for v in np.asarray(motives, dtype=np.float32).reshape(-1)),
            "metadata": {
                "terminated": bool(terminated),
                "steps": int(info.get("steps", 0)),
                "stop_reason": str(info.get("stop_reason", "unknown")),
            },
        }

    def _behavior_weights(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.behavior_model is not None:
            with torch.no_grad():
                alpha_tensor, support_tensor = self.behavior_model.forward_inference(
                    torch.as_tensor(context, dtype=torch.float32)
                )
            alpha = alpha_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)
            support_values = support_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32)
            weights = alpha_to_mean_weights(alpha).astype(np.float32)
            return alpha, support_values, weights

        weights = np.asarray(self.behavior_config.weights, dtype=np.float32).reshape(-1)
        if weights.shape != (2,) or np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 1.0, atol=1e-6):
            raise ValueError("fixed behavior weights must be a positive 2D simplex vector")
        alpha = np.maximum(weights, 1e-6).astype(np.float32)
        support_values = np.ones_like(alpha, dtype=np.float32)
        return alpha, support_values, weights

    @staticmethod
    def _probability_array(candidate_count: int, probabilities: dict[int, float]) -> np.ndarray:
        values = np.zeros(candidate_count, dtype=np.float32)
        for index, probability in probabilities.items():
            values[int(index)] = float(probability)
        return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect probability-aware runtime logs for IPS/DR MDN training.")
    parser.add_argument("--decisions", type=int, default=1000, help="Number of logged decisions to save")
    parser.add_argument("--save-dir", type=str, default="data/mdn_probability_aware_logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", type=str, default="runtime_log")
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted collection without overwriting files")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--pilot-checkpoint", type=str, default="models/pilot_ppo.pt")
    parser.add_argument("--map-location", type=str, default="cpu")
    parser.add_argument("--behavior-epsilon", type=float, default=0.2)
    parser.add_argument("--behavior-temperature", type=float, default=1.0)
    parser.add_argument("--behavior-weights", type=float, nargs=2, default=(0.5, 0.5))
    parser.add_argument("--behavior-mdn-checkpoint", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    behavior_weights = np.asarray(args.behavior_weights, dtype=np.float32)
    behavior_weights = behavior_weights / np.sum(behavior_weights)
    collector = ProbabilityAwareRuntimeLogCollector(
        seed=args.seed,
        save_dir=args.save_dir,
        max_steps=args.max_steps,
        gamma=args.gamma,
        baseline_episodes=args.baseline_episodes,
        pilot_checkpoint=args.pilot_checkpoint,
        map_location=args.map_location,
        behavior_config=BehaviorPolicyConfig(
            epsilon=args.behavior_epsilon,
            temperature=args.behavior_temperature,
            weights=tuple(float(v) for v in behavior_weights),
            mdn_checkpoint=args.behavior_mdn_checkpoint,
        ),
    )
    collector.collect(args.decisions, prefix=args.prefix, max_attempts=args.max_attempts, resume=args.resume)
    print(f"[Done] Probability-aware runtime logs saved to '{args.save_dir}/'")


if __name__ == "__main__":
    main()

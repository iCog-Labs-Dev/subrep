"""
Mixed Generator Data Collector.

Runs a diverse set of candidate policies (PPO variations, fixed-action, random)
from the same starting contexts and saves flat .npz records compatible with
generator/train_generator.py. This provides the SkillGenerator with a realistic
candidate distribution (matching the admission pipeline) without changing
certification semantics.

Usage:
    python -m data_collector.collect_mixed_generator_data \
      --episodes 1000 --save-dir data/raw_mixed --seed 42
"""
import argparse
import os
import random

import numpy as np
import torch

from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor
from data_collector.collect_candidate_sets import (
    CandidatePolicy,
    build_extended_candidate_policies,
)


class MixedGeneratorDataCollector:
    """Collects outcomes for all mixed candidates and saves individual .npz files."""

    def __init__(
        self,
        seed: int = 42,
        save_dir: str = "data/raw_mixed",
        pilot_checkpoint: str = "models/pilot_ppo.pt",
        map_location: str = "cpu",
    ) -> None:
        self.seed = seed
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        self.env = SubRepEnv(seed=seed)
        self.env.env.action_space.seed(seed)

        # Build the 9-candidate extended pool
        print(f"[Init] Loading policies (pilot from {pilot_checkpoint})...")
        self.candidate_policies = build_extended_candidate_policies(
            self.env,
            pilot_checkpoint=pilot_checkpoint,
            map_location=map_location,
        )
        print(f"[Init] Active candidates: {[p.skill_id for p in self.candidate_policies]}")

    def collect(self, num_episodes: int) -> None:
        """Run episodes and save one .npz file per rollout."""
        print(f"[Run] Collecting data for {num_episodes} starting contexts...")

        total_saved = 0

        for episode_idx in range(1, num_episodes + 1):
            context_seed = self.seed + episode_idx
            
            # Reset environment to establish the shared context
            context, _ = self.env.reset(seed=context_seed)
            context = np.asarray(context, dtype=np.float32)

            for candidate in self.candidate_policies:
                # 1. Reset stateful policies if they support it
                if hasattr(candidate.policy_fn, "reset"):
                    candidate.policy_fn.reset(seed=context_seed)

                # 2. Reset env exactly to the context seed
                obs, _ = self.env.reset(seed=context_seed)
                if not np.allclose(obs, context, rtol=1e-6, atol=1e-6):
                    raise RuntimeError("Reset seed did not reproduce the shared context")

                # 3. Execute rollout
                executor = SkillExecutor(
                    env=self.env,
                    policy_fn=candidate.policy_fn,
                    max_steps=200,
                )
                payoff, motives, terminated = executor.run_episode(initial_obs=obs)

                # 4. Extract proper initial_obs
                initial_obs = executor.last_run_info.get("initial_obs")
                if initial_obs is None:
                    initial_obs = context

                # 5. Format record identical to utils.data_collector structure
                recordkeys = {
                    "obs": np.asarray(initial_obs, dtype=np.float32),
                    "payoff": float(payoff),
                    "motives": np.asarray(motives, dtype=np.float32),
                    "skill_id": candidate.skill_id,
                    "terminated": bool(terminated),
                }

                behavior_probability = executor.last_run_info.get("behavior_probability")
                if behavior_probability is not None:
                    recordkeys["behavior_probability"] = float(behavior_probability)

                # 6. Save to disk
                total_saved += 1
                filename = f"{candidate.skill_id}_ep{total_saved:05d}.npz"
                filepath = os.path.join(self.save_dir, filename)
                np.savez(filepath, **recordkeys)

            if episode_idx % 10 == 0:
                print(f"      Completed {episode_idx}/{num_episodes} contexts "
                      f"({total_saved} files saved)...")

        print(f"[Done] Success. Total .npz files saved: {total_saved}")
        print(f"[Done] Output directory: {self.save_dir}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect mixed candidate outcomes for SkillGenerator training."
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default="data/raw_mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-checkpoint", type=str, default="models/pilot_ppo.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print("  Mixed Generator Data Collection")
    print("=" * 60)
    
    collector = MixedGeneratorDataCollector(
        seed=args.seed,
        save_dir=args.save_dir,
        pilot_checkpoint=args.pilot_checkpoint,
    )
    collector.collect(args.episodes)


if __name__ == "__main__":
    main()

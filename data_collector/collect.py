"""
Data Collection Entrypoint for SubRep.

Usage:
    python -m data_collector.collect
    python -m data_collector.collect --episodes 200 --save-dir data/raw --seed 42

Collects random rollouts from MO-LunarLander and saves them as .npz files
into data/raw/. These files are required to train the SkillGenerator:
    python -m generator.train_generator
"""

import argparse
from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor
from utils.data_collector import DataCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect MO-LunarLander rollouts and save as .npz files for generator training."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of episodes to collect (default: 500)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="data/raw",
        help="Directory to save .npz files (default: data/raw)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="random",
        help="Filename prefix for saved episodes (default: random)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  SubRep Data Collection")
    print("=" * 60)
    print(f"  Episodes  : {args.episodes}")
    print(f"  Save dir  : {args.save_dir}")
    print(f"  Seed      : {args.seed}")
    print(f"  Prefix    : {args.prefix}")
    print("-" * 60)

    # Set up the environment
    env = SubRepEnv(seed=args.seed)

    # The Team Lead added a trained RLPilot checkpoint. We use it here so the 
    # collected baseline data consists of meaningful flights instead of random crashes.
    executor = SkillExecutor.from_pilot_checkpoint(env=env)

    # Wire up the DataCollector backed by utils/data_collector.py
    collector = DataCollector(
        executor=executor,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    print(f"[Collect] Running {args.episodes} episodes with the trained RL pilot...")
    collector.collect_n_episodes(
        n=args.episodes,
        print_summary=True,
        skill_prefix=args.prefix,
    )

    print(f"[Done] .npz files saved to '{args.save_dir}/'")
    print("       You can now run: python -m generator.train_generator")


if __name__ == "__main__":
    main()

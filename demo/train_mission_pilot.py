"""Train a PPO pilot on the scan-then-land mission, for a fair comparison with plans.

The landing pilot never saw the beacon, so it cannot complete the mission. This
pilot follows the recipe of ``pilot.train_pilot``: it first copies a demonstration
of the mission, flown with the mission's scripted skills, and then fine-tunes the
copy with PPO on the mission's reward.

    python -m demo.train_mission_pilot
    python -m demo.train_mission_pilot --ppo-updates 2 --demonstrations 5 --output /tmp/pilot.pt
"""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List, Tuple

import numpy as np

from env.lunar_beacon import CRUISE_TO_PAD, DEFAULT_BEACON, GO_TO_BEACON, OBJECTIVES, BeaconMissionEnv, beacon_facts
from env.lunar_pilots import MISSION_PPO_CHECKPOINT
from env.lunar_subskills import SUBSKILLS
from pilot.rl_pilot import PPOConfig, RewardShapingConfig, RLPilot, _seed_everything
from pilot.train_pilot import behavior_clone_warm_start

OBSERVATION_NUMBERS = 11
MAX_STEPS = 1000
# Training seeds stay below the seeds the plans are evaluated on.
TRAINING_SEED = 50_000
CHECK_SEED = 3_000


class TrainingEnv:
    """The mission as ``RLPilot.train_ppo`` expects it.

    The trainer shapes its reward from the raw LunarLander rewards and counts a
    success from ``landing_success``, so here the Mission reward is added to the
    shaping slot and a success means the mission is complete. Rewards stay two
    numbers because that is what the trainer averages.
    """

    def __init__(self, seed: int, beacon: Tuple[float, float] = DEFAULT_BEACON) -> None:
        self.env = BeaconMissionEnv(seed=seed, beacon=beacon)

    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self.env.reset(seed=seed)

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        raw = np.array(info["raw_rewards"], dtype=np.float32)
        raw[1] += reward[OBJECTIVES.index("mission")]
        info = {**info, "raw_rewards": raw, "landing_success": info["mission_complete"]}
        return observation, reward[:2], terminated, truncated, info

    def close(self) -> None:
        self.env.close()


def collect_demonstrations(episodes: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Observation and action pairs from flying the mission with its scripted skills, one after another."""
    env = BeaconMissionEnv(seed=seed)
    sequence = (GO_TO_BEACON, CRUISE_TO_PAD, SUBSKILLS["slow_descent"], SUBSKILLS["touch_down"])
    observations: List[np.ndarray] = []
    actions: List[int] = []
    completed = 0
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            stage = frames_in_stage = 0
            for _ in range(MAX_STEPS):
                skill = sequence[stage]
                if stage + 1 < len(sequence) and (
                    skill.is_done(beacon_facts(observation)) or frames_in_stage >= skill.max_frames
                ):
                    stage, frames_in_stage = stage + 1, 0
                    skill = sequence[stage]
                action = int(skill.policy(observation))
                observations.append(observation)
                actions.append(action)
                frames_in_stage += 1
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    completed += bool(info["mission_complete"])
                    break
    finally:
        env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        {"episodes": episodes, "samples": len(actions), "completion_rate": completed / episodes},
    )


def train(args: argparse.Namespace) -> Dict[str, Any]:
    _seed_everything(args.seed)
    pilot = RLPilot(observation_dim=OBSERVATION_NUMBERS)
    observations, actions, dataset = collect_demonstrations(args.demonstrations, TRAINING_SEED + args.seed)
    cloning = behavior_clone_warm_start(
        pilot, observations, actions, seed=args.seed, epochs=args.bc_epochs, batch_size=256, learning_rate=1e-3, device="cpu"
    )
    config = PPOConfig(
        seed=args.seed + TRAINING_SEED,
        checkpoint_path=str(args.output),
        total_updates=args.ppo_updates,
        rollout_steps=2048,
        update_epochs=4,
        learning_rate=args.ppo_learning_rate,
        eval_episodes=args.eval_episodes,
    )
    make_env = lambda: TrainingEnv(seed=args.seed)  # noqa: E731
    cloned_state = copy.deepcopy(pilot.state_dict())
    cloned = pilot.evaluate_policy(make_env, episodes=args.check_episodes, seed=TRAINING_SEED, device="cpu")
    results = pilot.train_ppo(make_env, config=config, reward_config=RewardShapingConfig(), device="cpu")
    best = RLPilot.load(args.output, map_location="cpu")
    tuned = best.evaluate_policy(make_env, episodes=args.check_episodes, seed=TRAINING_SEED, device="cpu")
    source = "behavior_cloning_plus_ppo"
    if tuned["success_rate"] < cloned["success_rate"]:
        best.load_state_dict(cloned_state)  # PPO did not help, so the copy of the demonstrations stands
        source = "behavior_cloning"
    check = best.evaluate_policy(make_env, episodes=args.check_episodes, seed=CHECK_SEED, device="cpu")
    summary = {
        "training_source": source,
        "demonstrations": dataset,
        "cloning_final_loss": cloning["final_loss"],
        "cloned_success_rate": cloned["success_rate"],
        "ppo_best_success_rate": results["best_metrics"]["success_rate"],
        "tuned_success_rate": tuned["success_rate"],
        "development_seed_success_rate": check["success_rate"],
    }
    best.save(args.output, metadata=summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO pilot on the scan-then-land mission.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=str(MISSION_PPO_CHECKPOINT))
    parser.add_argument("--demonstrations", type=int, default=300)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--ppo-updates", type=int, default=60)
    parser.add_argument("--ppo-learning-rate", type=float, default=1e-4)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--check-episodes", type=int, default=100)
    args = parser.parse_args()
    print(train(args))


if __name__ == "__main__":
    main()

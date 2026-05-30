"""Train and evaluate the SubRep PPO pilot for MO-LunarLander.

The training recipe warm-starts the actor from deterministic LunarLander
heuristic demonstrations, then fine-tunes the neural policy with PPO using
SubRep-aligned safety and fuel reward shaping. The saved checkpoint includes
the configuration and evaluation metadata needed to reproduce pilot competence.

Example:
    python -m pilot.train_pilot --seed 7 --output models/pilot_ppo.pt
"""

from __future__ import annotations

import argparse
import platform
import sys
from dataclasses import asdict
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from baseline.idle_policy import IdlePolicy
from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor
from pilot.rl_pilot import PPOConfig, RewardShapingConfig, RLPilot, _seed_everything


DEFAULT_OUTPUT = Path("models/pilot_ppo.pt")
DEFAULT_EVAL_EPISODES = 100
DEFAULT_GATE_EPISODES = 10
DEFAULT_MAX_STEPS = 1000
DEFAULT_GAMMA = 0.99


def heuristic_action(obs: np.ndarray) -> int:
    """Return a deterministic LunarLander control action for warm-start data.

    The rule is the classic Box2D LunarLander heuristic translated into a small
    function for behavior-cloning demonstrations before PPO fine-tuning.
    """
    obs = np.asarray(obs, dtype=np.float32)
    angle_target = obs[0] * 0.5 + obs[2] * 1.0
    angle_target = float(np.clip(angle_target, -0.4, 0.4))
    hover_target = 0.55 * abs(float(obs[0]))
    angle_todo = (angle_target - float(obs[4])) * 0.5 - float(obs[5]) * 1.0
    hover_todo = (hover_target - float(obs[1])) * 0.5 - float(obs[3]) * 0.5

    if obs[6] or obs[7]:
        angle_todo = 0.0
        hover_todo = -(float(obs[3]) * 0.5)

    if hover_todo > abs(angle_todo) and hover_todo > 0.05:
        return 2
    if angle_todo < -0.05:
        return 3
    if angle_todo > 0.05:
        return 1
    return 0


def collect_heuristic_dataset(
    *,
    seed: int,
    episodes: int,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect observation/action pairs from the deterministic heuristic."""
    env = SubRepEnv(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    successes = 0
    lengths: list[int] = []

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            episode_success = False

            for step_index in range(max_steps):
                action = heuristic_action(obs)
                observations.append(np.asarray(obs, dtype=np.float32))
                actions.append(int(action))
                obs, _, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    episode_success = bool(
                        terminated and not truncated and info.get("landing_success", False)
                    )
                    break

            successes += int(episode_success)
            lengths.append(step_index + 1)
    finally:
        env.close()

    metadata = {
        "episodes": int(episodes),
        "samples": int(len(actions)),
        "success_rate": float(successes / episodes),
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
    }
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        metadata,
    )


def behavior_clone_warm_start(
    pilot: RLPilot,
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> dict[str, Any]:
    """Warm-start the actor with supervised heuristic actions."""
    if len(observations) == 0:
        raise ValueError("warm-start dataset is empty")

    rng = np.random.default_rng(seed)
    pilot.to(device)
    pilot.train()
    optimizer = optim.Adam(pilot.parameters(), lr=learning_rate)
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, device=device)
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    losses: list[float] = []

    for _ in range(epochs):
        indices = rng.permutation(len(observations))
        epoch_losses: list[float] = []
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            logits, _ = pilot(obs_tensor[batch_indices])
            loss = F.cross_entropy(logits, action_tensor[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pilot.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        losses.append(float(np.mean(epoch_losses)))

    pilot.eval()
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "final_loss": float(losses[-1]),
        "loss_history": losses,
    }


def evaluate_checkpoint(
    pilot: RLPilot,
    *,
    eval_episodes: int,
    gate_episodes: int,
    eval_seed: int,
    gate_seed: int,
    max_steps: int,
    gamma: float,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Compute strict pilot competence and certification metrics."""
    eval_metrics = _run_pilot_episodes(
        pilot,
        episodes=eval_episodes,
        seed=eval_seed,
        max_steps=max_steps,
        gamma=gamma,
    )
    baseline_env = SubRepEnv(seed=eval_seed)
    try:
        baseline_stats = IdlePolicy(baseline_env, gamma=gamma).run_baseline_episodes(
            num_episodes=eval_episodes,
            seed=eval_seed,
        )
    finally:
        baseline_env.close()

    calculator = ImprovementCalculator(baseline_stats)
    delta_r, delta_n = calculator.compute_improvements(
        skill_payoff=eval_metrics["mean_payoff"],
        skill_motives=eval_metrics["mean_motives"],
    )

    gate_passes = _count_gate_passes(
        checkpoint_path=checkpoint_path,
        episodes=gate_episodes,
        seed=gate_seed,
        max_steps=max_steps,
        gamma=gamma,
    )
    return {
        "success_rate": float(eval_metrics["success_rate"]),
        "successes": int(round(float(eval_metrics["success_rate"]) * eval_episodes)),
        "eval_episodes": int(eval_episodes),
        "mean_payoff": float(eval_metrics["mean_payoff"]),
        "mean_motives": _jsonable(eval_metrics["mean_motives"]),
        "mean_episode_length": float(eval_metrics["mean_episode_length"]),
        "idle_baseline_payoff": float(baseline_stats["baseline_payoff"]),
        "idle_baseline_motives": _jsonable(baseline_stats["baseline_motives"]),
        "mean_delta_r_vs_idle": float(delta_r),
        "mean_delta_n_vs_idle": _jsonable(delta_n),
        "gate_passes": int(gate_passes),
        "gate_episodes": int(gate_episodes),
        "gate_pass_rate": float(gate_passes / gate_episodes),
        "strict_success_definition": (
            "terminated and not truncated and info['landing_success'] is true"
        ),
    }


def build_metadata(
    *,
    args: argparse.Namespace,
    ppo_config: PPOConfig,
    reward_config: RewardShapingConfig,
    warm_start_dataset: dict[str, Any] | None,
    warm_start_training: dict[str, Any] | None,
    ppo_results: dict[str, Any] | None,
    expected_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build auditable checkpoint metadata for regeneration and evaluation."""
    return {
        "training_source": "heuristic_warm_start_plus_ppo_finetune",
        "training_entrypoint": "pilot.train_pilot",
        "regeneration_command": "python -m pilot.train_pilot --seed 7 --output models/pilot_ppo.pt",
        "command_used": _command_string(args),
        "metadata_only": bool(args.metadata_only),
        "seed": int(args.seed),
        "device": str(args.device),
        "ppo_config": asdict(ppo_config),
        "reward_shaping_config": asdict(reward_config),
        "warm_start": {
            "method": "deterministic_lunar_lander_heuristic_behavior_cloning",
            "heuristic_function": "pilot.train_pilot.heuristic_action",
            "config": {
                "episodes": int(args.warm_start_episodes),
                "max_steps": int(args.max_steps),
                "bc_epochs": int(args.bc_epochs),
                "bc_batch_size": int(args.bc_batch_size),
                "bc_learning_rate": float(args.bc_learning_rate),
            },
            "dataset": warm_start_dataset,
            "training": warm_start_training,
        },
        "ppo_results": _jsonable(ppo_results or {}),
        "expected_evaluation_metrics": expected_metrics,
        "environment": {
            "wrapper": "env.lunar_lander_wrapper.SubRepEnv",
            "env_id": "mo-lunar-lander-v3",
            "observation_shape": [8],
            "raw_reward_order": [
                "terminal_result",
                "dense_shaping",
                "main_engine_cost",
                "side_engine_cost",
            ],
            "subrep_reward_order": ["safety", "fuel"],
            "action_order": ["noop", "left_engine", "main_engine", "right_engine"],
        },
        "dependencies": dependency_info(),
    }


def dependency_info() -> dict[str, Any]:
    """Return lightweight environment/dependency metadata."""
    packages = {}
    for package_name in ("torch", "numpy", "gymnasium", "mo-gymnasium"):
        try:
            packages[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def train_and_save(args: argparse.Namespace) -> None:
    """Run warm start, PPO fine-tuning, strict evaluation, and checkpoint save."""
    _seed_everything(args.seed)
    output_path = Path(args.output)
    reward_config = RewardShapingConfig()
    ppo_config = PPOConfig(
        seed=args.seed,
        checkpoint_path=str(output_path),
        total_updates=args.ppo_updates,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.ppo_learning_rate,
        eval_episodes=args.ppo_eval_episodes,
        max_episode_steps=args.max_steps,
    )

    if args.metadata_only:
        pilot = RLPilot.load(output_path, map_location=args.device)
        warm_dataset_metadata = None
        warm_training_metadata = None
        ppo_results = None
    else:
        pilot = RLPilot()
        observations, actions, warm_dataset_metadata = collect_heuristic_dataset(
            seed=args.seed,
            episodes=args.warm_start_episodes,
            max_steps=args.max_steps,
        )
        warm_training_metadata = behavior_clone_warm_start(
            pilot,
            observations,
            actions,
            seed=args.seed,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            learning_rate=args.bc_learning_rate,
            device=args.device,
        )
        ppo_results = pilot.train_ppo(
            lambda: SubRepEnv(seed=args.seed),
            config=ppo_config,
            reward_config=reward_config,
            device=args.device,
        )
        pilot = RLPilot.load(output_path, map_location=args.device)

    temporary_metadata = build_metadata(
        args=args,
        ppo_config=ppo_config,
        reward_config=reward_config,
        warm_start_dataset=warm_dataset_metadata,
        warm_start_training=warm_training_metadata,
        ppo_results=ppo_results,
        expected_metrics={},
    )
    pilot.save(output_path, metadata=temporary_metadata)
    expected_metrics = evaluate_checkpoint(
        pilot,
        eval_episodes=args.eval_episodes,
        gate_episodes=args.gate_episodes,
        eval_seed=args.eval_seed,
        gate_seed=args.gate_seed,
        max_steps=args.max_steps,
        gamma=args.gamma,
        checkpoint_path=output_path,
    )
    final_metadata = build_metadata(
        args=args,
        ppo_config=ppo_config,
        reward_config=reward_config,
        warm_start_dataset=warm_dataset_metadata,
        warm_start_training=warm_training_metadata,
        ppo_results=ppo_results,
        expected_metrics=expected_metrics,
    )
    pilot.save(output_path, metadata=final_metadata)
    print(f"Saved PPO pilot checkpoint to {output_path}")
    print(f"Strict success rate: {expected_metrics['success_rate']:.2%}")
    print(f"Mean delta_r vs idle: {expected_metrics['mean_delta_r_vs_idle']:.4f}")
    print(f"Gate passes: {expected_metrics['gate_passes']}/{expected_metrics['gate_episodes']}")


def _run_pilot_episodes(
    pilot: RLPilot,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    gamma: float,
) -> dict[str, Any]:
    env = SubRepEnv(seed=seed)
    payoffs: list[float] = []
    motives: list[np.ndarray] = []
    successes = 0
    lengths: list[int] = []

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total_payoff = 0.0
            motive_deltas = np.zeros(2, dtype=np.float32)
            discount = 1.0
            episode_success = False

            for step_index in range(max_steps):
                action = pilot.predict(obs, deterministic=True, return_probability=False)
                obs, reward_vec, terminated, truncated, info = env.step(action)
                reward_vec = np.asarray(reward_vec, dtype=np.float32)
                total_payoff += discount * float(np.sum(reward_vec))
                motive_deltas += discount * reward_vec

                if terminated or truncated:
                    episode_success = bool(
                        terminated and not truncated and info.get("landing_success", False)
                    )
                    break
                discount *= gamma

            payoffs.append(float(total_payoff))
            motives.append(motive_deltas)
            successes += int(episode_success)
            lengths.append(step_index + 1)
    finally:
        env.close()

    return {
        "success_rate": float(successes / episodes),
        "mean_payoff": float(np.mean(payoffs)),
        "mean_motives": np.mean(np.asarray(motives, dtype=np.float32), axis=0).astype(np.float32),
        "mean_episode_length": float(np.mean(lengths)),
    }


def _count_gate_passes(
    *,
    checkpoint_path: Path,
    episodes: int,
    seed: int,
    max_steps: int,
    gamma: float,
) -> int:
    env = SubRepEnv(seed=seed)
    baseline_env = SubRepEnv(seed=seed)
    try:
        executor = SkillExecutor.from_pilot_checkpoint(
            env=env,
            checkpoint_path=str(checkpoint_path),
            gamma=gamma,
            max_steps=max_steps,
            deterministic=True,
        )
        baseline_stats = IdlePolicy(baseline_env, gamma=gamma).run_baseline_episodes(
            num_episodes=episodes,
            seed=seed,
        )
        calculator = ImprovementCalculator(baseline_stats)
        cds_gate = CDSGate()
        pds_gate = PDSGate(epsilon=0.1)
        gate_passes = 0

        for episode in range(episodes):
            env.reset(seed=seed + episode)
            payoff, motives, _ = executor.run_episode()
            delta_r, delta_n = calculator.compute_improvements(payoff, motives)
            if cds_gate.admit(delta_r, delta_n) or pds_gate.admit(delta_r, delta_n):
                gate_passes += 1
        return gate_passes
    finally:
        env.close()
        baseline_env.close()


def _jsonable(value: Any) -> Any:
    """Convert numpy/torch values into checkpoint-friendly Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _command_string(args: argparse.Namespace) -> str:
    parts = ["python", "-m", "pilot.train_pilot", "--seed", str(args.seed), "--output", str(args.output)]
    if args.metadata_only:
        parts.append("--metadata-only")
    return " ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="training seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="checkpoint path")
    parser.add_argument("--device", default="cpu", help="torch device")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="refresh metadata/evaluation for an existing checkpoint without retraining",
    )
    parser.add_argument("--warm-start-episodes", type=int, default=300)
    parser.add_argument("--bc-epochs", type=int, default=8)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ppo-updates", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--ppo-learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-eval-episodes", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--gate-episodes", type=int, default=DEFAULT_GATE_EPISODES)
    parser.add_argument("--eval-seed", type=int, default=2_000)
    parser.add_argument("--gate-seed", type=int, default=4_000)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    return parser.parse_args()

def main() -> None:
    train_and_save(parse_args())

if __name__ == "__main__":
    main()

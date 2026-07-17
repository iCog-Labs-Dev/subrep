"""Train a small PPO baseline for Safety-Gymnasium benchmark pilots.

Example:
    python -m pilot.train_safety_gymnasium_ppo \
      --env-id SafetyPointGoal1-v0 \
      --total-updates 25 \
      --output models/safety_ppo_point_goal.pt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from pilot.safety_gymnasium_ppo import SafetyPPOConfig, SafetyPPOPilot, config_to_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Safety-Gymnasium PPO baseline.")
    parser.add_argument("--env-id", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument("--output", type=str, default="models/safety_ppo_point_goal.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--total-updates", type=int, default=25)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--max-episode-steps", type=int, default=200)
    parser.add_argument("--cost-penalty", type=float, default=1.0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SafetyPPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        learning_rate=args.learning_rate,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        total_updates=args.total_updates,
        max_episode_steps=args.max_episode_steps,
        cost_penalty=args.cost_penalty,
        seed=args.seed,
    )
    result = train_safety_ppo(
        env_id=args.env_id,
        output_path=args.output,
        config=config,
        device=args.device,
        eval_episodes=args.eval_episodes,
    )
    print("Safety-Gymnasium PPO Training Complete", flush=True)
    print("======================================", flush=True)
    print(f"checkpoint: {args.output}", flush=True)
    for key, value in result["evaluation"].items():
        print(f"{key}: {value}", flush=True)


def train_safety_ppo(
    *,
    env_id: str,
    output_path: str | Path,
    config: SafetyPPOConfig,
    device: str = "cpu",
    eval_episodes: int = 10,
) -> dict[str, Any]:
    import safety_gymnasium

    _seed_everything(config.seed)
    env = safety_gymnasium.make(env_id)
    env.reset(seed=config.seed)
    env.action_space.seed(config.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    model = SafetyPPOPilot(
        observation_dim=obs_dim,
        action_dim=action_dim,
        action_low=np.asarray(env.action_space.low, dtype=np.float32).reshape(-1),
        action_high=np.asarray(env.action_space.high, dtype=np.float32).reshape(-1),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    obs, _ = env.reset(seed=config.seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    episode_return = 0.0
    episode_cost = 0.0
    episode_length = 0
    completed_returns: list[float] = []
    completed_costs: list[float] = []

    for update in range(1, config.total_updates + 1):
        rollout = _collect_rollout(
            env=env,
            model=model,
            obs=obs,
            config=config,
            device=device,
            episode_return=episode_return,
            episode_cost=episode_cost,
            episode_length=episode_length,
            completed_returns=completed_returns,
            completed_costs=completed_costs,
        )
        obs = rollout.pop("next_obs")
        episode_return = rollout.pop("episode_return")
        episode_cost = rollout.pop("episode_cost")
        episode_length = rollout.pop("episode_length")

        _ppo_update(model=model, optimizer=optimizer, rollout=rollout, config=config, device=device)

        if update == 1 or update % max(1, config.total_updates // 5) == 0:
            recent_return = float(np.mean(completed_returns[-10:])) if completed_returns else 0.0
            recent_cost = float(np.mean(completed_costs[-10:])) if completed_costs else 0.0
            print(
                f"[{update:04d}/{config.total_updates:04d}] "
                f"recent_return={recent_return:.4f} recent_cost={recent_cost:.4f}",
                flush=True,
            )

    evaluation = evaluate_safety_ppo(
        model,
        env_id=env_id,
        seed=config.seed + 10_000,
        episodes=eval_episodes,
        max_episode_steps=config.max_episode_steps,
    )
    env.close()

    metadata = {
        "training_entrypoint": "pilot.train_safety_gymnasium_ppo",
        "environment": env_id,
        "config": config_to_dict(config),
        "evaluation": evaluation,
    }
    model.save(output_path, metadata=metadata)
    return {"evaluation": evaluation, "metadata": metadata}


def _collect_rollout(
    *,
    env,
    model: SafetyPPOPilot,
    obs: np.ndarray,
    config: SafetyPPOConfig,
    device: str,
    episode_return: float,
    episode_cost: float,
    episode_length: int,
    completed_returns: list[float],
    completed_costs: list[float],
) -> dict[str, Any]:
    observations: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []

    current_obs = obs
    for _ in range(config.rollout_steps):
        env_action, raw_action, log_prob, value = model.act(
            current_obs,
            deterministic=False,
            return_raw=True,
        )
        next_obs, reward, cost, terminated, truncated, _ = env.step(env_action)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        reward_value = float(reward) - config.cost_penalty * float(cost)
        done = bool(terminated or truncated)

        observations.append(current_obs)
        raw_actions.append(raw_action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward_value)
        dones.append(float(done))

        episode_return += float(reward)
        episode_cost += float(cost)
        episode_length += 1

        current_obs = next_obs
        if done or episode_length >= config.max_episode_steps:
            completed_returns.append(float(episode_return))
            completed_costs.append(float(episode_cost))
            current_obs, _ = env.reset()
            current_obs = np.asarray(current_obs, dtype=np.float32).reshape(-1)
            episode_return = 0.0
            episode_cost = 0.0
            episode_length = 0

    with torch.no_grad():
        _, next_value = model.distribution(
            torch.as_tensor(current_obs, dtype=torch.float32, device=device)
        )
    advantages, returns = _compute_gae(
        rewards=np.asarray(rewards, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        next_value=float(next_value.detach().cpu().item()),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(raw_actions, dtype=np.float32),
        "old_log_probs": np.asarray(log_probs, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "next_obs": current_obs,
        "episode_return": episode_return,
        "episode_cost": episode_cost,
        "episode_length": episode_length,
    }


def _compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    for index in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[index]
        next_val = next_value if index == len(rewards) - 1 else values[index + 1]
        delta = rewards[index] + gamma * next_val * next_non_terminal - values[index]
        last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
        advantages[index] = last_advantage
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages.astype(np.float32), returns.astype(np.float32)


def _ppo_update(
    *,
    model: SafetyPPOPilot,
    optimizer,
    rollout: dict[str, np.ndarray],
    config: SafetyPPOConfig,
    device: str,
) -> None:
    obs = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(rollout["old_log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)

    rng = np.random.default_rng(config.seed)
    indices = np.arange(len(obs))
    model.train()
    for _ in range(config.update_epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), config.minibatch_size):
            batch = indices[start : start + config.minibatch_size]
            batch_tensor = torch.as_tensor(batch, dtype=torch.long, device=device)
            new_log_probs, entropy, values = model.evaluate_actions(
                obs[batch_tensor],
                actions[batch_tensor],
            )
            ratio = torch.exp(new_log_probs - old_log_probs[batch_tensor])
            unclipped = ratio * advantages[batch_tensor]
            clipped = torch.clamp(
                ratio,
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * advantages[batch_tensor]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[batch_tensor])
            entropy_loss = -entropy.mean()
            loss = (
                policy_loss
                + config.value_coef * value_loss
                + config.entropy_coef * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
    model.eval()


def evaluate_safety_ppo(
    model: SafetyPPOPilot,
    *,
    env_id: str,
    seed: int,
    episodes: int,
    max_episode_steps: int,
) -> dict[str, float]:
    import safety_gymnasium

    env = safety_gymnasium.make(env_id)
    returns: list[float] = []
    costs: list[float] = []
    lengths: list[int] = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            total_return = 0.0
            total_cost = 0.0
            for step in range(1, max_episode_steps + 1):
                action = model.predict(obs, deterministic=True)
                obs, reward, cost, terminated, truncated, _ = env.step(action)
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                total_return += float(reward)
                total_cost += float(cost)
                if terminated or truncated:
                    break
            returns.append(total_return)
            costs.append(total_cost)
            lengths.append(step)
    finally:
        env.close()

    return {
        "episodes": float(episodes),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_cost": float(np.mean(costs)) if costs else 0.0,
        "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()

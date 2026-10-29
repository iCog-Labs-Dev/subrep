"""Train a small PPO baseline for Safety-Gymnasium benchmark pilots.

Example:
    python -m pilot.train_safety_gymnasium_ppo \
      --env-id SafetyPointGoal1-v0 \
      --total-updates 25 \
      --output models/safety_ppo_point_goal.pt
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--use-lagrangian", action="store_true")
    parser.add_argument("--cost-limit", type=float, default=1.0)
    parser.add_argument("--lambda-lr", type=float, default=0.05)
    parser.add_argument("--initial-lagrange-multiplier", type=float, default=1.0)
    parser.add_argument("--max-lagrange-multiplier", type=float, default=50.0)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-seed", type=int, default=180000)
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
        use_lagrangian=args.use_lagrangian,
        cost_limit=args.cost_limit,
        lambda_lr=args.lambda_lr,
        initial_lagrange_multiplier=args.initial_lagrange_multiplier,
        max_lagrange_multiplier=args.max_lagrange_multiplier,
        seed=args.seed,
    )
    result = train_safety_ppo(
        env_id=args.env_id,
        output_path=args.output,
        config=config,
        device=args.device,
        eval_episodes=args.eval_episodes,
        eval_every=args.eval_every, eval_seed=args.eval_seed,
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
    eval_every: int = 25,
    eval_seed: int = 180000,
) -> dict[str, Any]:
    import safety_gymnasium

    if min(config.total_updates, config.rollout_steps, config.update_epochs,
           config.minibatch_size, config.max_episode_steps, eval_episodes, eval_every) < 1:
        raise ValueError("Training and evaluation counts must be positive")
    output_path = Path(output_path)
    run_dir = output_path.parent / (output_path.stem + "_run")
    if output_path.exists() or run_dir.exists():
        raise FileExistsError("Use a new checkpoint path and run directory")
    run_dir.mkdir(parents=True)
    history = []
    best_key = None
    best_metadata = None
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
        separate_critic=config.use_lagrangian,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Training-only cost value function; inference still uses the same actor API.
    cost_critic = None
    cost_optimizer = None
    if config.use_lagrangian:
        cost_critic = torch.nn.Sequential(torch.nn.Linear(obs_dim + 1, 128), torch.nn.Tanh(),
            torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1)).to(device)
        cost_optimizer = optim.Adam(cost_critic.parameters(), lr=config.learning_rate)

    obs, _ = env.reset(seed=config.seed)
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    episode_return = 0.0
    episode_cost = 0.0
    episode_length = 0
    completed_returns: list[float] = []
    completed_costs: list[float] = []
    lagrange_multiplier = (
        float(config.initial_lagrange_multiplier)
        if config.use_lagrangian
        else float(config.cost_penalty)
    )

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
            current_cost_penalty=lagrange_multiplier,
            cost_critic=cost_critic,
        )
        obs = rollout.pop("next_obs")
        episode_return = rollout.pop("episode_return")
        episode_cost = rollout.pop("episode_cost")
        episode_length = rollout.pop("episode_length")

        optimization = _ppo_update(model=model, optimizer=optimizer, rollout=rollout,
            config=config, device=device, cost_critic=cost_critic, cost_optimizer=cost_optimizer)

        if config.use_lagrangian:
            lagrange_multiplier = _update_lagrange_multiplier(
                lagrange_multiplier, rollout["rollout_mean_episode_cost"], config)

        row = {"update": update, "environment_steps": update * config.rollout_steps,
               "recent_return": float(np.mean(completed_returns[-10:])) if completed_returns else None,
               "recent_cost": float(np.mean(completed_costs[-10:])) if completed_costs else None,
               "lagrange_multiplier": lagrange_multiplier,
               "rollout_mean_episode_cost": rollout["rollout_mean_episode_cost"]}
        row.update(optimization)
        with torch.no_grad():
            observations = torch.as_tensor(rollout["observations"], device=device)
            means, stds, _ = model(observations)
            low, high = model.action_low, model.action_high
            near = ((means-low)/(high-low) < .01) | ((high-means)/(high-low) < .01)
            row["mean_near_bound_rate"] = float(near.float().mean().cpu())
            row["exploration_std"] = stds.mean(0).cpu().tolist()
            actions = rollout["actions"]
            row["sampled_action_clip_rate"] = float(np.mean(
                (actions < low.cpu().numpy()) | (actions > high.cpu().numpy())))
        if update == 1 or update % eval_every == 0 or update == config.total_updates:
            # Evaluation must not perturb the training random streams.
            numpy_state, python_state = np.random.get_state(), random.getstate()
            try:
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                    evaluation = evaluate_safety_ppo(model, env_id=env_id, seed=eval_seed,
                        episodes=eval_episodes, max_episode_steps=config.max_episode_steps)
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)
            metadata = {"training_entrypoint": "pilot.train_safety_gymnasium_ppo",
                "environment": env_id, "config": config_to_dict(config),
                "trainer_version": "finite_episode_cost_v3" if config.use_lagrangian else "ppo_v1",
                "evaluation": evaluation, "evaluation_role": "development checkpoint selection, not final test",
                "evaluation_seeds": list(range(eval_seed, eval_seed+eval_episodes)),
                "update": update, "final_lagrange_multiplier": lagrange_multiplier if config.use_lagrangian else None,
                "selection_rule": "PPO: goal rate then return; Lagrangian: budget feasibility first, then goal rate/return; if infeasible, lower cost first."}
            model.save(run_dir / f"update_{update:05d}.pt", metadata=metadata)
            key = _checkpoint_rank(evaluation, config)
            if best_key is None or key > best_key:
                best_key, best_metadata = key, metadata
                model.save(output_path, metadata=metadata)
            row["evaluation"] = evaluation
            row["selected_best_update"] = best_metadata["update"]
            print(f"Development evaluation update {update}: {evaluation}; best={best_metadata['update']}", flush=True)
        history.append(row)
        with (run_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        if update == 1 or update % max(1, config.total_updates // 5) == 0:
            recent_return = float(np.mean(completed_returns[-10:])) if completed_returns else 0.0
            recent_cost = float(np.mean(completed_costs[-10:])) if completed_costs else 0.0
            print(
                f"[{update:04d}/{config.total_updates:04d}] "
                f"recent_return={recent_return:.4f} recent_cost={recent_cost:.4f} "
                f"cost_penalty={lagrange_multiplier:.4f}",
                flush=True,
            )

    env.close()
    (run_dir / "summary.json").write_text(json.dumps({"best_checkpoint": str(output_path),
        "best_metadata": best_metadata, "updates": len(history),
        "budget_met": best_metadata["evaluation"]["mean_cost"] <= config.cost_limit,
        "note": "Development selection only. Check independent contexts before accepting this policy."}, indent=2))
    return {"evaluation": best_metadata["evaluation"], "metadata": best_metadata}


def _checkpoint_rank(evaluation, config):
    goal, reward, cost = (evaluation[k] for k in ("goal_reached_rate", "mean_return", "mean_cost"))
    if config.use_lagrangian:
        if cost <= config.cost_limit:
            return (1., goal, reward, -cost)
        return (0., -cost, goal, reward)
    return (goal, reward, -cost)


def _update_lagrange_multiplier(multiplier, mean_completed_episode_cost, config):
    """Episode budget must be compared with completed episodes, not fragments."""
    if mean_completed_episode_cost is None:
        return float(multiplier)
    if not np.isfinite(mean_completed_episode_cost):
        raise ValueError("Episode cost must be finite")
    return float(np.clip(multiplier + config.lambda_lr *
                         (mean_completed_episode_cost - config.cost_limit),
                         0., config.max_lagrange_multiplier))


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
    current_cost_penalty: float,
    cost_critic=None,
) -> dict[str, Any]:
    observations: list[np.ndarray] = []
    raw_actions: list[np.ndarray] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    rollout_completed_costs: list[float] = []

    if config.use_lagrangian and cost_critic is None:
        raise ValueError("Lagrangian training requires a separate cost critic")
    cost_values, cost_rewards, cost_observations = [], [], []
    current_obs = obs
    for _ in range(config.rollout_steps):
        env_action, raw_action, log_prob, value = model.act(
            current_obs,
            deterministic=False,
            return_raw=True,
        )
        next_obs, reward, cost, terminated, truncated, _ = env.step(env_action)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
        reward_value = float(reward) if config.use_lagrangian else float(reward) - float(current_cost_penalty) * float(cost)
        cost_value = 0.
        cost_reward = float(cost)
        if cost_critic is not None:
            with torch.no_grad():
                cost_obs = _cost_observation(current_obs, episode_length, config.max_episode_steps)
                cost_value = float(cost_critic(torch.as_tensor(cost_obs, device=device)).item())
            cost_observations.append(cost_obs)
        cutoff = episode_length + 1 >= config.max_episode_steps
        done = bool(terminated or truncated or cutoff)
        # Time limits end the sampled episode, not the underlying task. Bootstrap
        # from its final observation before resetting, while stopping GAE at the
        # boundary. True terminal states have no continuation value.
        if done and not terminated:
            with torch.no_grad():
                _, final_value = model.distribution(
                    torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                )
            reward_value += config.gamma * float(final_value.detach().cpu().item())
            # Cost budget ends at every episode boundary, including time limits.
            # Reward retains its existing continuing-task bootstrap.

        observations.append(current_obs)
        raw_actions.append(raw_action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward_value)
        cost_values.append(cost_value)
        cost_rewards.append(cost_reward)
        dones.append(float(done))

        episode_return += float(reward)
        episode_cost += float(cost)
        episode_length += 1

        current_obs = next_obs
        if done:
            completed_returns.append(float(episode_return))
            completed_costs.append(float(episode_cost))
            rollout_completed_costs.append(float(episode_cost))
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
        normalize=not config.use_lagrangian,
    )
    cost_returns = None
    if cost_critic is not None:
        with torch.no_grad():
            next_cost = float(cost_critic(torch.as_tensor(
                _cost_observation(current_obs, episode_length, config.max_episode_steps),
                device=device)).item())
        cost_advantages, cost_returns = _compute_gae(
            rewards=np.asarray(cost_rewards, dtype=np.float32),
            values=np.asarray(cost_values, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32), next_value=next_cost,
            gamma=1.0, gae_lambda=config.gae_lambda, normalize=False)
        advantages = _lagrangian_advantages(advantages, cost_advantages, current_cost_penalty)

    return {
        "cost_returns": cost_returns,
        "cost_observations": np.asarray(cost_observations, dtype=np.float32),
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(raw_actions, dtype=np.float32),
        "old_log_probs": np.asarray(log_probs, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "rollout_mean_episode_cost": (
            float(np.mean(rollout_completed_costs))
            if rollout_completed_costs
            else None
        ),
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
    normalize: bool = True,
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
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages.astype(np.float32), returns.astype(np.float32)


def _cost_observation(observation, elapsed_steps, horizon):
    """Time-aware value of remaining raw cost within this evaluation episode."""
    if horizon < 1 or not 0 <= elapsed_steps <= horizon:
        raise ValueError("Invalid episode time")
    return np.concatenate((np.asarray(observation, dtype=np.float32).reshape(-1),
                           np.array([1. - elapsed_steps / horizon], dtype=np.float32)))


def _lagrangian_advantages(reward_advantages, cost_advantages, multiplier):
    # Preserve the relative reward/cost scale; normalize only after combining.
    combined = (reward_advantages - multiplier * cost_advantages) / (1. + multiplier)
    return ((combined - combined.mean()) / (combined.std() + 1e-8)).astype(np.float32)


def _ppo_update(
    *,
    model: SafetyPPOPilot,
    optimizer,
    rollout: dict[str, np.ndarray],
    config: SafetyPPOConfig,
    device: str,
    cost_critic=None,
    cost_optimizer=None,
) -> dict[str, float]:
    obs = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(rollout["old_log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)

    losses = []
    cost_returns = (torch.as_tensor(rollout["cost_returns"], device=device)
                    if cost_critic is not None else None)
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
            if model.separate_critic:
                actor_parameters = list(model.backbone.parameters()) + list(model.actor_mean.parameters()) + [model.log_std]
                critic_parameters = list(model.critic_backbone.parameters()) + list(model.critic.parameters())
                torch.nn.utils.clip_grad_norm_(actor_parameters, config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(critic_parameters, config.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            cost_loss = torch.zeros((), device=device)
            if cost_critic is not None:
                cost_loss = F.mse_loss(cost_critic(torch.as_tensor(rollout["cost_observations"], device=device)[batch_tensor]).squeeze(-1), cost_returns[batch_tensor])
                cost_optimizer.zero_grad()
                cost_loss.backward()
                torch.nn.utils.clip_grad_norm_(cost_critic.parameters(), config.max_grad_norm)
                cost_optimizer.step()
            with torch.no_grad():
                log_ratio = new_log_probs - old_log_probs[batch_tensor]
                approx_kl = ((ratio - 1) - log_ratio).mean()
            losses.append([policy_loss.item(), value_loss.item(), cost_loss.item(), approx_kl.item()])
    model.eval()
    return dict(zip(("policy_loss", "reward_value_loss", "cost_value_loss", "approx_kl"),
                    np.mean(losses, axis=0).tolist()))


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
    goals: list[int] = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            total_return = 0.0
            total_cost = 0.0
            episode_goals = 0
            for step in range(1, max_episode_steps + 1):
                action = model.predict(obs, deterministic=True)
                obs, reward, cost, terminated, truncated, info = env.step(action)
                episode_goals += int(bool(info.get("goal_met", False)))
                obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                total_return += float(reward)
                total_cost += float(cost)
                if terminated or truncated:
                    break
            returns.append(total_return)
            costs.append(total_cost)
            lengths.append(step)
            goals.append(episode_goals)
    finally:
        env.close()

    return {
        "episodes": float(episodes),
        "goal_reached_rate": float(np.mean(np.asarray(goals)>0)) if goals else 0.,
        "mean_goals": float(np.mean(goals)) if goals else 0.,
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

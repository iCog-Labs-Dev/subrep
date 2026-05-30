"""PPO-trained neural pilot for MO-LunarLander skill execution.

The pilot is intentionally separate from certification. PPO optimizes a scalar
training reward shaped from raw MO-LunarLander signals, while SubRep
certification continues to consume the environment's 2D motive vector.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.distributions import Categorical


@dataclass
class RewardShapingConfig:
    """Scalar PPO reward weights aligned with SubRep safety/fuel motives."""

    # Dense shaping guides the lander during flight; the terminal bonus/penalty
    # is what distinguishes a true successful landing from merely stable motion.
    shaping_weight: float = 1.0
    landing_success_bonus: float = 100.0
    crash_penalty_weight: float = 1.0
    # Engine use is penalized during PPO training, while SubRep certification
    # later sees the mapped fuel motive where larger means less fuel consumed.
    fuel_weight: float = 0.1
    main_engine_cost: float = -1.0
    side_engine_cost: float = -1.0


@dataclass
class PPOConfig:
    """Training parameters for clipped PPO."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    rollout_steps: int = 2048
    update_epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    total_updates: int = 100
    eval_episodes: int = 20
    max_episode_steps: int = 1000
    target_kl: Optional[float] = 0.03
    checkpoint_path: str = "models/pilot_ppo.pt"
    seed: int = 0


class RLPilot(nn.Module):
    """Actor-critic PPO pilot for discrete MO-LunarLander actions."""

    def __init__(
        self,
        observation_dim: int = 8,
        action_dim: int = 4,
        hidden_sizes: Iterable[int] = (128, 128, 64),
    ) -> None:
        super().__init__()

        hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if action_dim <= 1:
            raise ValueError("action_dim must be greater than 1")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("all hidden layer sizes must be positive")

        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = hidden_sizes

        layers: list[nn.Module] = []
        input_dim = self.observation_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Shared trunk follows the existing generator style, but the heads are
        # RL-specific: actor logits choose actions, critic values stabilize PPO.
        self.backbone = nn.Sequential(*layers)
        self.actor = nn.Linear(input_dim, self.action_dim)
        self.critic = nn.Linear(input_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Use stable Xavier initialization for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _prepare_observation(self, obs: Tensor) -> tuple[Tensor, bool]:
        if obs.ndim not in (1, 2):
            raise ValueError(
                f"Expected obs shape ({self.observation_dim},) or "
                f"(N, {self.observation_dim}), got {tuple(obs.shape)}"
            )

        is_single_input = obs.ndim == 1
        if is_single_input:
            if obs.shape[0] != self.observation_dim:
                raise ValueError(
                    f"Expected single obs shape ({self.observation_dim},), got {tuple(obs.shape)}"
                )
            obs = obs.unsqueeze(0)
        elif obs.shape[1] != self.observation_dim:
            raise ValueError(
                f"Expected batched obs shape (N, {self.observation_dim}), got {tuple(obs.shape)}"
            )

        return obs.float(), is_single_input

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Return action logits and value estimate for one or many observations."""
        obs, is_single_input = self._prepare_observation(obs)
        features = self.backbone(obs)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)

        if is_single_input:
            logits = logits.squeeze(0)
            values = values.squeeze(0)

        return logits, values

    def action_distribution(self, obs: Tensor) -> tuple[Categorical, Tensor]:
        """Build a categorical action distribution and critic value."""
        logits, values = self(obs)
        return Categorical(logits=logits), values

    def act(
        self,
        obs: np.ndarray | Tensor,
        *,
        deterministic: bool = False,
        return_probability: bool = True,
    ) -> int | tuple[int, float]:
        """Choose an action for a single observation."""
        device = next(self.parameters()).device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)

        self.eval()
        with torch.no_grad():
            logits, _ = self(obs_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            if deterministic:
                action_tensor = torch.argmax(probabilities, dim=-1)
            else:
                action_tensor = Categorical(probs=probabilities).sample()
            action = int(action_tensor.item())
            probability = float(probabilities[action].clamp_min(1e-12).item())

        if return_probability:
            return action, probability
        return action

    def predict(
        self,
        obs: np.ndarray | Tensor,
        *,
        deterministic: bool = True,
        return_probability: bool = True,
    ) -> int | tuple[int, float]:
        """Policy callable compatible with SkillExecutor."""
        # Deterministic mode is used for evaluation/certification so the same
        # checkpoint produces stable pass/fail metrics across test runs.
        return self.act(
            obs,
            deterministic=deterministic,
            return_probability=return_probability,
        )

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return log-probabilities, entropy, and values for PPO batches."""
        distribution, values = self.action_distribution(obs)
        log_probs = distribution.log_prob(actions.long())
        entropy = distribution.entropy()
        return log_probs, entropy, values

    def save(self, path: str | Path, metadata: Optional[dict[str, Any]] = None) -> None:
        """Save model weights and architecture metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": self.hidden_sizes,
            "metadata": _metadata_to_python(metadata or {}),
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
    ) -> "RLPilot":
        """Load a checkpoint saved by RLPilot.save."""
        payload = torch.load(path, map_location=map_location)
        if isinstance(payload, dict) and "state_dict" in payload:
            model = cls(
                observation_dim=int(payload.get("observation_dim", 8)),
                action_dim=int(payload.get("action_dim", 4)),
                hidden_sizes=tuple(payload.get("hidden_sizes", (128, 128, 64))),
            )
            model.load_state_dict(payload["state_dict"])
        elif isinstance(payload, dict):
            model = cls()
            model.load_state_dict(payload)
        else:
            raise ValueError(f"Unsupported pilot checkpoint format: {type(payload).__name__}")

        model.to(map_location if isinstance(map_location, torch.device) else torch.device(map_location))
        model.eval()
        return model

    @staticmethod
    def shaped_reward(
        raw_rewards: np.ndarray | None,
        action: int,
        info: Optional[dict[str, Any]] = None,
        config: Optional[RewardShapingConfig] = None,
    ) -> float:
        """Compute scalar PPO reward from safety and fuel signals."""
        config = config or RewardShapingConfig()
        info = info or {}

        if raw_rewards is None:
            raw_rewards = info.get("raw_rewards")
        if raw_rewards is None:
            raw_rewards = np.zeros(4, dtype=np.float32)
        raw_rewards = np.asarray(raw_rewards, dtype=np.float32).reshape(-1)
        if raw_rewards.shape[0] < 4:
            raise ValueError(f"raw_rewards must contain 4 objectives, got shape {raw_rewards.shape}")

        # Raw reward order is [terminal_result, dense_shaping, main_cost,
        # side_cost]. Only terminal_result +100 should count as landing success.
        landing_success = bool(info.get("landing_success", raw_rewards[0] >= 100.0))
        safety_reward = (
            config.shaping_weight * float(raw_rewards[1])
            + (config.landing_success_bonus if landing_success else 0.0)
            + config.crash_penalty_weight * float(raw_rewards[0] if raw_rewards[0] < 0.0 else 0.0)
        )
        # MO-LunarLander actions: 0=noop, 1=left engine, 2=main engine,
        # 3=right engine. Fuel costs make the PPO pilot prefer safe landings
        # that also remain useful for SubRep's fuel motive.
        if int(action) == 2:
            fuel_cost = config.main_engine_cost
        elif int(action) in (1, 3):
            fuel_cost = config.side_engine_cost
        else:
            fuel_cost = 0.0
        return float(safety_reward + config.fuel_weight * fuel_cost)

    def train_ppo(
        self,
        env_factory: Callable[[], Any],
        *,
        config: Optional[PPOConfig] = None,
        reward_config: Optional[RewardShapingConfig] = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Train the pilot with clipped PPO and save the best checkpoint."""
        config = config or PPOConfig()
        reward_config = reward_config or RewardShapingConfig()
        _seed_everything(config.seed)

        device = torch.device(device)
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        checkpoint_path = Path(config.checkpoint_path)
        best_metrics = {"success_rate": -1.0, "mean_return": -float("inf")}
        training_history: list[dict[str, float]] = []

        env = env_factory()
        obs, _ = _reset_env(env, config.seed)
        episode_index = 0

        try:
            for update_index in range(config.total_updates):
                # PPO alternates between collecting on-policy rollouts and
                # applying clipped policy/value updates to avoid divergence.
                rollout = _collect_rollout(
                    model=self,
                    env=env,
                    initial_obs=obs,
                    config=config,
                    reward_config=reward_config,
                    device=device,
                    episode_seed_start=config.seed + episode_index + 1,
                )
                obs = rollout.pop("last_obs")
                episode_index += int(rollout.pop("episodes_finished"))

                metrics = _ppo_update(
                    model=self,
                    optimizer=optimizer,
                    rollout=rollout,
                    config=config,
                    device=device,
                )
                metrics["update"] = float(update_index + 1)
                training_history.append(metrics)

                eval_metrics = self.evaluate_policy(
                    env_factory,
                    episodes=config.eval_episodes,
                    max_steps=config.max_episode_steps,
                    seed=config.seed + 10_000 + update_index * config.eval_episodes,
                    reward_config=reward_config,
                    device=device,
                )
                # Prefer checkpoints that land more often; use shaped return as
                # the tie-breaker so the saved model is both competent and fuel-aware.
                improved = (
                    eval_metrics["success_rate"] > best_metrics["success_rate"]
                    or (
                        np.isclose(eval_metrics["success_rate"], best_metrics["success_rate"])
                        and eval_metrics["mean_return"] > best_metrics["mean_return"]
                    )
                )
                if improved:
                    best_metrics = dict(eval_metrics)
                    metadata = {
                        "training_source": "ppo",
                        "ppo_updates_completed": update_index + 1,
                        "seed": config.seed,
                        "ppo_config": asdict(config),
                        "reward_shaping": asdict(reward_config),
                        "best_metrics": best_metrics,
                    }
                    self.save(checkpoint_path, metadata=metadata)
        finally:
            if hasattr(env, "close"):
                env.close()

        return {
            "checkpoint_path": str(checkpoint_path),
            "best_metrics": best_metrics,
            "training_history": training_history,
        }

    def evaluate_policy(
        self,
        env_factory: Callable[[], Any],
        *,
        episodes: int = 100,
        max_steps: int = 1000,
        seed: int = 0,
        reward_config: Optional[RewardShapingConfig] = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Evaluate deterministic pilot performance in an environment factory."""
        if episodes <= 0:
            raise ValueError("episodes must be positive")

        device = torch.device(device)
        self.to(device)
        self.eval()

        env = env_factory()
        returns: list[float] = []
        successes = 0
        episode_lengths: list[int] = []
        motive_returns: list[np.ndarray] = []
        try:
            for episode in range(episodes):
                obs, _ = _reset_env(env, seed + episode)
                total_return = 0.0
                motives = np.zeros(2, dtype=np.float32)
                discount = 1.0
                success = False

                for step_index in range(max_steps):
                    action = self.predict(obs, deterministic=True, return_probability=False)
                    obs, reward_vec, terminated, truncated, info = env.step(action)
                    raw_rewards = info.get("raw_rewards") if isinstance(info, dict) else None
                    total_return += self.shaped_reward(raw_rewards, int(action), info, reward_config)
                    motives += discount * np.asarray(reward_vec, dtype=np.float32)
                    success = success or bool(isinstance(info, dict) and info.get("landing_success", False))
                    if terminated or truncated:
                        break
                    discount *= 0.99

                returns.append(float(total_return))
                successes += int(success)
                episode_lengths.append(step_index + 1)
                motive_returns.append(motives)
        finally:
            if hasattr(env, "close"):
                env.close()

        motive_array = np.asarray(motive_returns, dtype=np.float32)
        return {
            "success_rate": float(successes / episodes),
            "mean_return": float(np.mean(returns)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "mean_motives": np.mean(motive_array, axis=0).astype(np.float32),
            "episode_returns": np.asarray(returns, dtype=np.float32),
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _metadata_to_python(value: Any) -> Any:
    """Convert checkpoint metadata to safe Python primitives before saving."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _metadata_to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_to_python(item) for item in value]
    return value


def _reset_env(env: Any, seed: int):
    try:
        return env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            try:
                seed_attr = getattr(env, "seed")
                if callable(seed_attr):
                    seed_attr(seed)
                else:
                    env.seed = seed
            except Exception:
                pass
        return env.reset()


def _collect_rollout(
    *,
    model: RLPilot,
    env: Any,
    initial_obs: np.ndarray,
    config: PPOConfig,
    reward_config: RewardShapingConfig,
    device: torch.device,
    episode_seed_start: int,
) -> dict[str, Any]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    log_probs: list[float] = []
    values: list[float] = []

    obs = initial_obs
    episodes_finished = 0

    for _ in range(config.rollout_steps):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            distribution, value = model.action_distribution(obs_tensor)
            action_tensor = distribution.sample()
            log_prob = distribution.log_prob(action_tensor)

        action = int(action_tensor.item())
        next_obs, _, terminated, truncated, info = env.step(action)
        raw_rewards = info.get("raw_rewards") if isinstance(info, dict) else None
        reward = model.shaped_reward(raw_rewards, action, info, reward_config)
        done = bool(terminated or truncated)

        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)
        log_probs.append(float(log_prob.item()))
        values.append(float(value.item()))

        obs = next_obs
        if done:
            obs, _ = _reset_env(env, episode_seed_start + episodes_finished)
            episodes_finished += 1

    with torch.no_grad():
        _, next_value = model.action_distribution(torch.as_tensor(obs, dtype=torch.float32, device=device))
    advantages, returns = _compute_gae(
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        values=np.asarray(values, dtype=np.float32),
        next_value=float(next_value.item()),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "old_log_probs": np.asarray(log_probs, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "last_obs": obs,
        "episodes_finished": episodes_finished,
    }


def _compute_gae(
    *,
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for step_index in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - float(dones[step_index])
        next_value_step = next_value if step_index == len(rewards) - 1 else values[step_index + 1]
        delta = rewards[step_index] + gamma * next_value_step * next_non_terminal - values[step_index]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step_index] = gae
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def _ppo_update(
    *,
    model: RLPilot,
    optimizer: optim.Optimizer,
    rollout: dict[str, Any],
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    observations = torch.as_tensor(rollout["observations"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout["actions"], dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(rollout["old_log_probs"], dtype=torch.float32, device=device)
    advantages = torch.as_tensor(rollout["advantages"], dtype=torch.float32, device=device)
    returns = torch.as_tensor(rollout["returns"], dtype=torch.float32, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = observations.shape[0]
    minibatch_size = min(config.minibatch_size, batch_size)
    indices = np.arange(batch_size)
    last_policy_loss = 0.0
    last_value_loss = 0.0
    last_entropy = 0.0
    last_kl = 0.0

    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, batch_size, minibatch_size):
            batch_indices = torch.as_tensor(indices[start : start + minibatch_size], dtype=torch.long, device=device)
            new_log_probs, entropy, values = model.evaluate_actions(
                observations[batch_indices], actions[batch_indices]
            )
            log_ratio = new_log_probs - old_log_probs[batch_indices]
            ratio = torch.exp(log_ratio)
            unclipped = ratio * advantages[batch_indices]
            clipped = torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * advantages[batch_indices]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[batch_indices])
            entropy_mean = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_mean

            if not torch.isfinite(loss):
                raise RuntimeError("PPO loss diverged to a non-finite value")

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()
            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy_mean.item())
            last_kl = float(approximate_kl)

        if config.target_kl is not None and last_kl > config.target_kl:
            break

    return {
        "policy_loss": last_policy_loss,
        "value_loss": last_value_loss,
        "entropy": last_entropy,
        "approx_kl": last_kl,
    }

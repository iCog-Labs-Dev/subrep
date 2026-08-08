"""Continuous-action PPO pilot for Safety-Gymnasium benchmark pilots."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal


@dataclass
class SafetyPPOConfig:
    """Small PPO configuration for first Safety-Gymnasium baselines."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    rollout_steps: int = 1024
    update_epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    total_updates: int = 25
    max_episode_steps: int = 200
    cost_penalty: float = 1.0
    use_lagrangian: bool = False
    cost_limit: float = 1.0
    lambda_lr: float = 0.05
    initial_lagrange_multiplier: float = 1.0
    max_lagrange_multiplier: float = 50.0
    seed: int = 42


class SafetyPPOPilot(nn.Module):
    """Gaussian actor-critic policy for continuous Safety-Gymnasium actions."""

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        action_low: Iterable[float],
        action_high: Iterable[float],
        hidden_sizes: Iterable[int] = (128, 128),
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)

        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer")

        low = torch.as_tensor(tuple(float(v) for v in action_low), dtype=torch.float32)
        high = torch.as_tensor(tuple(float(v) for v in action_high), dtype=torch.float32)
        if low.shape != (self.action_dim,) or high.shape != (self.action_dim,):
            raise ValueError("action bounds must match action_dim")
        if torch.any(high <= low):
            raise ValueError("action_high must be greater than action_low")
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

        layers: list[nn.Module] = []
        input_dim = self.observation_dim
        for hidden_dim in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(input_dim, self.action_dim)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.5))
        self.critic = nn.Linear(input_dim, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
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
        is_single = obs.ndim == 1
        if is_single:
            if obs.shape[0] != self.observation_dim:
                raise ValueError(
                    f"Expected obs shape ({self.observation_dim},), got {tuple(obs.shape)}"
                )
            obs = obs.unsqueeze(0)
        elif obs.shape[1] != self.observation_dim:
            raise ValueError(
                f"Expected batched obs shape (N, {self.observation_dim}), got {tuple(obs.shape)}"
            )
        return obs.float(), is_single

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        obs, is_single = self._prepare_observation(obs)
        features = self.backbone(obs)
        unit_mean = torch.tanh(self.actor_mean(features))
        center = (self.action_high + self.action_low) / 2.0
        scale = (self.action_high - self.action_low) / 2.0
        mean = center + unit_mean * scale
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.critic(features).squeeze(-1)
        if is_single:
            mean = mean.squeeze(0)
            std = std.squeeze(0)
            value = value.squeeze(0)
        return mean, std, value

    def distribution(self, obs: Tensor) -> tuple[Normal, Tensor]:
        mean, std, value = self(obs)
        return Normal(mean, std), value

    def evaluate_actions(self, obs: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution, values = self.distribution(obs)
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_probs, entropy, values

    def act(
        self,
        obs: np.ndarray | Tensor,
        *,
        deterministic: bool = False,
        return_raw: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, float, float]:
        device = next(self.parameters()).device
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        self.eval()
        with torch.no_grad():
            distribution, value = self.distribution(obs_tensor)
            if deterministic:
                raw_action = distribution.mean
            else:
                raw_action = distribution.sample()
            log_prob = distribution.log_prob(raw_action).sum()
            env_action = torch.clamp(raw_action, self.action_low, self.action_high)

        env_np = env_action.detach().cpu().numpy().astype(np.float32)
        if return_raw:
            return (
                env_np,
                raw_action.detach().cpu().numpy().astype(np.float32),
                float(log_prob.detach().cpu().item()),
                float(value.detach().cpu().item()),
            )
        return env_np

    def predict(self, obs: np.ndarray | Tensor, *, deterministic: bool = True) -> np.ndarray:
        return self.act(obs, deterministic=deterministic)  # type: ignore[return-value]

    def save(self, path: str | Path, metadata: Optional[dict[str, Any]] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.state_dict(),
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.detach().cpu().tolist(),
            "action_high": self.action_high.detach().cpu().tolist(),
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
    ) -> "SafetyPPOPilot":
        payload = torch.load(path, map_location=map_location)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise ValueError("Unsupported SafetyPPOPilot checkpoint format")
        model = cls(
            observation_dim=int(payload["observation_dim"]),
            action_dim=int(payload["action_dim"]),
            action_low=payload["action_low"],
            action_high=payload["action_high"],
            hidden_sizes=tuple(payload.get("hidden_sizes", (128, 128))),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(map_location if isinstance(map_location, torch.device) else torch.device(map_location))
        model.eval()
        return model


def _metadata_to_python(value):
    if isinstance(value, dict):
        return {str(key): _metadata_to_python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_to_python(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def config_to_dict(config: SafetyPPOConfig) -> dict[str, Any]:
    return asdict(config)

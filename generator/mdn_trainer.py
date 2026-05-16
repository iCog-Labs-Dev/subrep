"""Contextual-bandit style trainer for MDN alpha policy learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_contracts import MDNDecisionRecord, validate_decision_record
from utils.mdn_reward import compute_advantage, compute_mdn_policy_loss, compute_mdn_utility
from utils.mdn_selection import sample_dirichlet_weights, select_best_candidate


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class MDNTrainerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    random_seed: int = 0
    payoff_weight: float = 0.0
    checkpoint_path: str = "models/mdn_policy_best.pth"
    strict_validation: bool = False


class MDNTrainer:
    """Train MDN alpha outputs through downstream selection utility."""

    def __init__(self, model: MotiveDecompositionNetwork, config: Optional[MDNTrainerConfig] = None, device: Optional[str] = None) -> None:
        self.model = model
        self.config = config or MDNTrainerConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.running_baseline: float | None = None

    def training_step(self, record: MDNDecisionRecord) -> dict[str, float]:
        """Run one offline policy-learning step from a validated decision record."""
        validate_decision_record(record)
        if record.actual_motives is None:
            raise ValueError("training_step requires actual_motives in the decision record")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        context = torch.tensor(record.context, dtype=torch.float32, device=self.device)
        alpha, support_values = self.model(context)
        distribution = torch.distributions.Dirichlet(alpha)
        recorded_weights = torch.tensor(record.weights_used, dtype=torch.float32, device=self.device)
        log_prob = distribution.log_prob(recorded_weights)

        weights_np = recorded_weights.detach().cpu().numpy()
        selected_skill_id, selected_score = select_best_candidate(record.candidate_skills, weights_np)
        if self.config.strict_validation and selected_skill_id != record.selected_skill_id:
            raise ValueError(
                "record selected_skill_id does not match selector output under recorded weights"
            )

        utility = compute_mdn_utility(
            actual_motives=np.asarray(record.actual_motives, dtype=np.float32),
            weights_used=weights_np,
            actual_payoff=record.actual_payoff,
            payoff_weight=self.config.payoff_weight,
        ) if record.utility is None else float(record.utility)
        advantage = compute_advantage(
            utility=utility,
            baseline_utility=None,
            running_baseline=self.running_baseline,
        )
        loss = compute_mdn_policy_loss(log_prob, advantage)
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.running_baseline = utility if self.running_baseline is None else 0.9 * self.running_baseline + 0.1 * utility

        return {
            "loss": float(loss.item()),
            "utility": float(utility),
            "advantage": float(advantage),
            "selected_score": float(selected_score),
            "log_prob": float(log_prob.item()),
            "alpha_mean": float(alpha.detach().mean().item()),
            "support_mean": float(support_values.detach().mean().item()),
        }

    def train_records(self, records: Iterable[MDNDecisionRecord]) -> dict[str, float]:
        """Run one pass over validated offline records."""
        metrics = [self.training_step(record) for record in records]
        if not metrics:
            raise ValueError("train_records requires at least one decision record")
        return {
            key: float(np.mean([item[key] for item in metrics]))
            for key in metrics[0]
        }

    def save_checkpoint(self, path: str | Path | None = None) -> str:
        """Save trainer state for future continuation."""
        checkpoint_path = Path(path or self.config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "running_baseline": self.running_baseline,
            },
            checkpoint_path,
        )
        return str(checkpoint_path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        model: MotiveDecompositionNetwork,
        device: Optional[str] = None,
    ) -> "MDNTrainer":
        """Restore trainer state from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=device or "cpu")
        trainer = cls(
            model=model,
            config=MDNTrainerConfig(**checkpoint["config"]),
            device=device,
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.running_baseline = checkpoint.get("running_baseline")
        return trainer


def create_trainer_for_model(model: MotiveDecompositionNetwork, seed: int = 0, device: Optional[str] = None) -> MDNTrainer:
    """Convenience constructor with deterministic seeding."""
    _seed_everything(seed)
    return MDNTrainer(model=model, config=MDNTrainerConfig(random_seed=seed), device=device)

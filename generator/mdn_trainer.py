"""Contextual-bandit style trainer for MDN alpha policy learning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import random

from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_contracts import MDNDecisionRecord, validate_decision_record
from utils.mdn_reward import compute_advantage, compute_mdn_policy_loss, compute_mdn_utility
from utils.mdn_selection import sample_dirichlet_weights, select_best_candidate


def _seed_everything(seed: int) -> None:
    random.seed(seed)
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
    entropy_beta: float = 0.01
    gradient_clip_norm: float = 1.0
    batch_size: int = 16


class TrainingCallback(Protocol):
    def on_step(self, step: int, metrics: dict[str, float]) -> None: ...


class MDNTrainer:
    """Train MDN alpha outputs through downstream selection utility."""

    def __init__(
        self,
        model: MotiveDecompositionNetwork,
        config: Optional[MDNTrainerConfig] = None,
        device: Optional[str] = None,
        callback: Optional[TrainingCallback] = None,
    ) -> None:
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
        self._context_baselines: dict[tuple[float, ...], float] = {}
        self.callback = callback
        self._step_count = 0

    @staticmethod
    def _get_context_key(context: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(round(value, 3) for value in context)

    def _update_baselines(self, context_key: tuple[float, ...], utility: float) -> None:
        baseline_momentum = 0.9
        if context_key in self._context_baselines:
            self._context_baselines[context_key] = (
                baseline_momentum * self._context_baselines[context_key]
                + (1.0 - baseline_momentum) * utility
            )
        else:
            self._context_baselines[context_key] = utility
        self.running_baseline = (
            utility
            if self.running_baseline is None
            else baseline_momentum * self.running_baseline + (1.0 - baseline_momentum) * utility
        )

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
        entropy = distribution.entropy()

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
        context_key = self._get_context_key(record.context)
        context_baseline = self._context_baselines.get(context_key, self.running_baseline)
        advantage = compute_advantage(
            utility=utility,
            baseline_utility=None,
            running_baseline=context_baseline,
        )
        policy_loss = compute_mdn_policy_loss(log_prob, advantage)
        loss = policy_loss - self.config.entropy_beta * entropy
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clip_norm)
        self.optimizer.step()

        self._update_baselines(context_key, utility)

        metrics = {
            "loss": float(loss.item()),
            "utility": float(utility),
            "advantage": float(advantage),
            "entropy": float(entropy.item()),
            "selected_score": float(selected_score),
            "log_prob": float(log_prob.item()),
            "alpha_mean": float(alpha.detach().mean().item()),
            "alpha_max": float(alpha.detach().max().item()),
            "support_mean": float(support_values.detach().mean().item()),
        }
        self._step_count += 1
        if self.callback is not None:
            self.callback.on_step(self._step_count, metrics)
        return metrics

    def train_records(self, records: Iterable[MDNDecisionRecord]) -> dict[str, float]:
        """Run one pass over offline records with mini-batch gradient updates."""
        records_list = list(records)
        if not records_list:
            raise ValueError("train_records requires at least one decision record")
        random.shuffle(records_list)

        all_metrics: list[dict[str, float]] = []
        batch_size = self.config.batch_size

        for index in range(0, len(records_list), batch_size):
            batch = records_list[index : index + batch_size]
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            batch_loss = torch.tensor(0.0, device=self.device)
            batch_utilities: list[float] = []
            batch_entropies: list[float] = []
            batch_log_probs: list[float] = []
            batch_selected_scores: list[float] = []
            batch_alpha_means: list[float] = []
            batch_alpha_maxes: list[float] = []
            batch_support_means: list[float] = []
            batch_advantages: list[float] = []

            for record in batch:
                validate_decision_record(record)
                if record.actual_motives is None:
                    raise ValueError("train_records requires actual_motives in each record")

                context = torch.tensor(record.context, dtype=torch.float32, device=self.device)
                alpha, support_values = self.model(context)
                distribution = torch.distributions.Dirichlet(alpha)
                recorded_weights = torch.tensor(record.weights_used, dtype=torch.float32, device=self.device)
                log_prob = distribution.log_prob(recorded_weights)
                entropy = distribution.entropy()

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

                context_key = self._get_context_key(record.context)
                context_baseline = self._context_baselines.get(context_key, self.running_baseline)
                advantage = compute_advantage(
                    utility=utility,
                    baseline_utility=None,
                    running_baseline=context_baseline,
                )

                policy_loss = compute_mdn_policy_loss(log_prob, advantage)
                step_loss = policy_loss - self.config.entropy_beta * entropy
                batch_loss = batch_loss + step_loss / len(batch)

                batch_utilities.append(float(utility))
                batch_entropies.append(float(entropy.item()))
                batch_log_probs.append(float(log_prob.item()))
                batch_selected_scores.append(float(selected_score))
                batch_alpha_means.append(float(alpha.detach().mean().item()))
                batch_alpha_maxes.append(float(alpha.detach().max().item()))
                batch_support_means.append(float(support_values.detach().mean().item()))
                batch_advantages.append(float(advantage))

            batch_loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clip_norm)
            self.optimizer.step()

            for record, utility in zip(batch, batch_utilities):
                context_key = self._get_context_key(record.context)
                self._update_baselines(context_key, utility)

            metrics = {
                "loss": float(batch_loss.item()),
                "utility": float(np.mean(batch_utilities)),
                "advantage": float(np.mean(batch_advantages)),
                "entropy": float(np.mean(batch_entropies)),
                "selected_score": float(np.mean(batch_selected_scores)),
                "log_prob": float(np.mean(batch_log_probs)),
                "alpha_mean": float(np.mean(batch_alpha_means)),
                "alpha_max": float(np.mean(batch_alpha_maxes)),
                "support_mean": float(np.mean(batch_support_means)),
            }
            all_metrics.append(metrics)
            self._step_count += 1
            if self.callback is not None:
                self.callback.on_step(self._step_count, metrics)

        return {
            key: float(np.mean([item[key] for item in all_metrics]))
            for key in all_metrics[0]
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
                "context_baselines": self._context_baselines,
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
        trainer._context_baselines = checkpoint.get("context_baselines", {})
        return trainer


def create_trainer_for_model(model: MotiveDecompositionNetwork, seed: int = 0, device: Optional[str] = None) -> MDNTrainer:
    """Convenience constructor with deterministic seeding."""
    _seed_everything(seed)
    return MDNTrainer(model=model, config=MDNTrainerConfig(random_seed=seed), device=device)

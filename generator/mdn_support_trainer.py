"""Trainer for the MDN support head against W_x support-function targets."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_

from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_checkpoint_loader import assert_support_head_compatible
from utils.weight_set_store import WeightSetStore


@dataclass
class SupportTrainerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    min_contexts_to_train: int = 1
    max_epochs: int = 500
    early_stopping_patience: int = 50
    min_delta: float = 1e-6
    checkpoint_path: str = "models/mdn_support_best.pth"


class MDNSupportTrainer:
    """Train the support head to match support-function targets from W_x."""

    def __init__(
        self,
        model: MotiveDecompositionNetwork,
        store: WeightSetStore,
        config: Optional[SupportTrainerConfig] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.store = store
        self.config = config or SupportTrainerConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.loss_fn = MSELoss()
        self.optimizer = torch.optim.AdamW(
            list(self.model.support_head.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        # Updated by every training_step; asserted to be exactly 0.0 in CI.
        self.last_feasibility_violation_rate: float = 0.0

    def training_step(self) -> Optional[float]:
        targets = self.store.get_all_support_targets()
        if len(targets) < self.config.min_contexts_to_train:
            return None

        self.model.train()
        self.model.zero_grad(set_to_none=True)

        contexts = torch.tensor(np.stack([item[0] for item in targets], axis=0), dtype=torch.float32, device=self.device)
        target_values = torch.tensor(np.stack([item[1] for item in targets], axis=0), dtype=torch.float32, device=self.device)

        _, support_predictions = self.model.forward_inference(contexts)
        loss = self.loss_fn(support_predictions, target_values)

        # Diagnostic only -- never added to the loss. Under SASP feasibility is
        # algebraic, so this must read exactly 0.0; a nonzero value means a code
        # regression (a new head, a bypassed decoder), not a tuning problem.
        self.last_feasibility_violation_rate = self._feasibility_violation_rate(
            support_predictions.detach()
        )

        loss.backward()
        clip_grad_norm_(self.model.support_head.parameters(), max_norm=self.config.gradient_clip_norm)
        self.optimizer.step()

        return float(loss.item())

    def evaluation_metrics(self, store: WeightSetStore) -> dict[str, float]:
        """Evaluate support regression without updating the model."""
        targets = store.get_all_support_targets()
        if not targets:
            raise ValueError("Support evaluation requires at least one target context")

        contexts = torch.tensor(
            np.stack([item[0] for item in targets], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        target_values = torch.tensor(
            np.stack([item[1] for item in targets], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        self.model.eval()
        with torch.no_grad():
            _, predictions = self.model.forward_inference(contexts)
            errors = predictions - target_values
        return {
            "mse": float(torch.mean(errors.square()).item()),
            "mae": float(torch.mean(torch.abs(errors)).item()),
            "feasibility_violation_rate": self._feasibility_violation_rate(predictions),
        }

    def fit(self, validation_store: WeightSetStore) -> dict[str, float]:
        """Fit on the attached training store and select by validation MSE."""
        if self.store.context_count() < self.config.min_contexts_to_train:
            raise ValueError(
                "Training store does not meet min_contexts_to_train: "
                f"{self.store.context_count()} < {self.config.min_contexts_to_train}"
            )
        if self.config.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.config.max_epochs}")
        if self.config.early_stopping_patience < 0:
            raise ValueError(
                "early_stopping_patience must be non-negative, got "
                f"{self.config.early_stopping_patience}"
            )

        initial_validation = self.evaluation_metrics(validation_store)
        best_validation_mse = initial_validation["mse"]
        best_epoch = 0
        best_support_state = copy.deepcopy(self.model.support_head.state_dict())
        best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        epochs_without_improvement = 0
        last_train_loss = float("nan")
        epochs_completed = 0

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.training_step()
            if train_loss is None:
                raise ValueError("Support training unexpectedly produced no loss")
            last_train_loss = train_loss
            epochs_completed = epoch
            validation_mse = self.evaluation_metrics(validation_store)["mse"]
            if validation_mse < best_validation_mse - self.config.min_delta:
                best_validation_mse = validation_mse
                best_epoch = epoch
                best_support_state = copy.deepcopy(self.model.support_head.state_dict())
                best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (
                self.config.early_stopping_patience > 0
                and epochs_without_improvement >= self.config.early_stopping_patience
            ):
                break

        self.model.support_head.load_state_dict(best_support_state)
        self.optimizer.load_state_dict(best_optimizer_state)
        final_train = self.evaluation_metrics(self.store)
        final_validation = self.evaluation_metrics(validation_store)
        return {
            "epochs_completed": float(epochs_completed),
            "best_epoch": float(best_epoch),
            "last_train_step_mse": float(last_train_loss),
            "train_mse": final_train["mse"],
            "train_mae": final_train["mae"],
            "validation_mse": final_validation["mse"],
            "validation_mae": final_validation["mae"],
            "initial_validation_mse": initial_validation["mse"],
            "validation_feasibility_violation_rate": final_validation[
                "feasibility_violation_rate"
            ],
        }

    @staticmethod
    def _feasibility_violation_rate(support_values: torch.Tensor) -> float:
        """Fraction of predictions violating either W_x feasibility constraint.

        A prediction is infeasible if any s_i falls outside [0, 1] or the row
        sums to less than 1 (an empty region). Expected to be 0.0 under SASP.
        """
        values = support_values.reshape(-1, support_values.shape[-1])
        out_of_range = ((values < 0.0) | (values > 1.0)).any(dim=-1)
        empty_region = values.sum(dim=-1) < 1.0 - 1e-6
        violations = (out_of_range | empty_region).float()
        return float(violations.mean().item())

    def save_checkpoint(
        self,
        path: str | Path | None = None,
        *,
        metadata: dict[str, object] | None = None,
    ) -> str:
        checkpoint_path = Path(path or self.config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }
        if metadata is not None:
            payload["support_training"] = metadata
        torch.save(payload, checkpoint_path)
        return str(checkpoint_path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        model: MotiveDecompositionNetwork,
        store: WeightSetStore,
        device: Optional[str] = None,
    ) -> "MDNSupportTrainer":
        checkpoint = torch.load(path, map_location=device or "cpu")

        # SASP widened the support head from M to 2M, so both the model state
        # AND the optimizer state (which wraps support_head.parameters()) change
        # shape. Reject pre-SASP support checkpoints with the shared, actionable
        # error instead of an opaque failure inside Adam's state load.
        assert_support_head_compatible(checkpoint["model_state_dict"])

        trainer = cls(
            model=model,
            store=store,
            config=SupportTrainerConfig(**checkpoint["config"]),
            device=device,
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return trainer

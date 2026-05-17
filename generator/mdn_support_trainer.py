"""Trainer for the MDN support head against W_x support-function targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_

from generator.mdn import MotiveDecompositionNetwork
from utils.weight_set_store import WeightSetStore


@dataclass
class SupportTrainerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    min_contexts_to_train: int = 1
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

    def training_step(self) -> Optional[float]:
        targets = self.store.get_all_support_targets()
        if len(targets) < self.config.min_contexts_to_train:
            return None

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        contexts = torch.tensor(np.stack([item[0] for item in targets], axis=0), dtype=torch.float32, device=self.device)
        target_values = torch.tensor(np.stack([item[1] for item in targets], axis=0), dtype=torch.float32, device=self.device)

        _, support_predictions = self.model.forward_inference(contexts)
        loss = self.loss_fn(support_predictions, target_values)
        loss.backward()
        clip_grad_norm_(self.model.support_head.parameters(), max_norm=self.config.gradient_clip_norm)
        self.optimizer.step()

        return float(loss.item())

    def save_checkpoint(self, path: str | Path | None = None) -> str:
        checkpoint_path = Path(path or self.config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.__dict__,
            },
            checkpoint_path,
        )
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
        trainer = cls(
            model=model,
            store=store,
            config=SupportTrainerConfig(**checkpoint["config"]),
            device=device,
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return trainer

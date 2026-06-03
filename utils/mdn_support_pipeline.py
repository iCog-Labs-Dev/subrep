"""Minimal support-pipeline helpers for wiring certified weights into W_x training."""

from __future__ import annotations

import numpy as np

from generator.mdn_support_trainer import MDNSupportTrainer
from utils.weight_set_store import WeightSetStore


def observe_and_train_support(
    *,
    store: WeightSetStore,
    trainer: MDNSupportTrainer,
    context,
    weight_vector,
) -> float | None:
    """Record a certified weight into W_x and run one support training step.
    """
    context_array = np.asarray(context, dtype=np.float32)
    weight_array = np.asarray(weight_vector, dtype=np.float32)
    store.observe_certified_weight(context_array, weight_array)
    return trainer.training_step()

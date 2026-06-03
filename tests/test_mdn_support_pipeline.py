from __future__ import annotations

import numpy as np

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_support_trainer import MDNSupportTrainer, SupportTrainerConfig
from utils.mdn_support_pipeline import observe_and_train_support
from utils.weight_set_store import WeightSetStore


def test_observe_and_train_support_adds_vertex_to_store():
    model = MotiveDecompositionNetwork()
    store = WeightSetStore(num_objectives=2)
    trainer = MDNSupportTrainer(
        model,
        store,
        config=SupportTrainerConfig(min_contexts_to_train=10_000),
        device="cpu",
    )

    result = observe_and_train_support(
        store=store,
        trainer=trainer,
        context=np.array([0.1] * 8, dtype=np.float32),
        weight_vector=np.array([0.8, 0.2], dtype=np.float32),
    )

    assert store.context_count() == 1
    assert store.total_vertex_count() == 1
    assert result is None


def test_observe_and_train_support_returns_finite_loss_when_training_runs():
    model = MotiveDecompositionNetwork()
    store = WeightSetStore(num_objectives=2)
    trainer = MDNSupportTrainer(
        model,
        store,
        config=SupportTrainerConfig(min_contexts_to_train=1),
        device="cpu",
    )

    loss = observe_and_train_support(
        store=store,
        trainer=trainer,
        context=np.array([0.1] * 8, dtype=np.float32),
        weight_vector=np.array([0.8, 0.2], dtype=np.float32),
    )

    assert loss is not None
    assert np.isfinite(loss)


def test_observe_and_train_support_accumulates_multiple_contexts():
    model = MotiveDecompositionNetwork()
    store = WeightSetStore(num_objectives=2)
    trainer = MDNSupportTrainer(
        model,
        store,
        config=SupportTrainerConfig(min_contexts_to_train=1),
        device="cpu",
    )

    observe_and_train_support(
        store=store,
        trainer=trainer,
        context=np.array([0.1] * 8, dtype=np.float32),
        weight_vector=np.array([0.8, 0.2], dtype=np.float32),
    )
    observe_and_train_support(
        store=store,
        trainer=trainer,
        context=np.array([0.2] * 8, dtype=np.float32),
        weight_vector=np.array([0.3, 0.7], dtype=np.float32),
    )

    assert store.context_count() == 2
    assert store.total_vertex_count() == 2

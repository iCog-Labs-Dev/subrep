from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_support_trainer import MDNSupportTrainer, SupportTrainerConfig
from utils.weight_set_store import WeightSetStore


def _store_with_targets() -> WeightSetStore:
    store = WeightSetStore(num_objectives=2)
    store.observe_certified_weight(np.array([0.1] * 8, dtype=np.float32), np.array([0.8, 0.2], dtype=np.float32))
    store.observe_certified_weight(np.array([0.2] * 8, dtype=np.float32), np.array([0.3, 0.7], dtype=np.float32))
    return store


def test_support_trainer_returns_none_when_not_enough_contexts():
    model = MotiveDecompositionNetwork()
    store = WeightSetStore(num_objectives=2)
    trainer = MDNSupportTrainer(
        model,
        store,
        config=SupportTrainerConfig(min_contexts_to_train=1_000),
        device="cpu",
    )

    assert trainer.training_step() is None


def test_support_trainer_one_step_runs_and_returns_finite_loss():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNSupportTrainer(
        model,
        _store_with_targets(),
        config=SupportTrainerConfig(),
        device="cpu",
    )

    loss = trainer.training_step()

    assert loss is not None
    assert np.isfinite(loss)


def test_support_trainer_updates_support_predictions_toward_targets():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    store = _store_with_targets()
    trainer = MDNSupportTrainer(
        model,
        store,
        config=SupportTrainerConfig(learning_rate=5e-3),
        device="cpu",
    )

    context, target_values = store.get_all_support_targets()[0]
    with torch.no_grad():
        _, before = model.forward_inference(torch.tensor(context, dtype=torch.float32))
    before_distance = torch.norm(before - torch.tensor(target_values, dtype=torch.float32)).item()

    for _ in range(30):
        trainer.training_step()

    with torch.no_grad():
        _, after = model.forward_inference(torch.tensor(context, dtype=torch.float32))
    after_distance = torch.norm(after - torch.tensor(target_values, dtype=torch.float32)).item()

    assert after_distance < before_distance


def test_support_trainer_checkpoint_round_trip(tmp_path: Path):
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    store = _store_with_targets()
    trainer = MDNSupportTrainer(
        model,
        store,
        config=SupportTrainerConfig(checkpoint_path=str(tmp_path / "mdn_support_best.pth")),
        device="cpu",
    )
    trainer.training_step()

    checkpoint_path = trainer.save_checkpoint()
    restored_model = MotiveDecompositionNetwork()
    restored_trainer = MDNSupportTrainer.from_checkpoint(checkpoint_path, restored_model, store, device="cpu")

    with torch.no_grad():
        original_output = trainer.model.forward_inference(torch.tensor((0.1,) * 8, dtype=torch.float32))[1]
        restored_output = restored_trainer.model.forward_inference(torch.tensor((0.1,) * 8, dtype=torch.float32))[1]

    assert torch.allclose(original_output, restored_output)

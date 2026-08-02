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

    targets = store.get_all_support_targets()
    contexts = torch.tensor(np.stack([item[0] for item in targets], axis=0), dtype=torch.float32)
    target_values = torch.tensor(np.stack([item[1] for item in targets], axis=0), dtype=torch.float32)
    with torch.no_grad():
        _, before = model.forward_inference(contexts)
    before_loss = torch.nn.functional.mse_loss(before, target_values).item()

    for _ in range(30):
        trainer.training_step()

    with torch.no_grad():
        _, after = model.forward_inference(contexts)
    after_loss = torch.nn.functional.mse_loss(after, target_values).item()

    assert after_loss < before_loss


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


def test_support_trainer_reports_zero_feasibility_violations():
    """The train-time diagnostic must read exactly 0.0 under SASP.

    This is Approach 2's penalty surviving as a regression tripwire rather than
    a corrective term: feasibility is algebraic under SASP, so any nonzero
    reading means a code regression (a new head, a bypassed decoder), not a
    hyperparameter that needs tuning.
    """
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNSupportTrainer(
        model,
        _store_with_targets(),
        config=SupportTrainerConfig(),
        device="cpu",
    )

    assert trainer.last_feasibility_violation_rate == 0.0

    for _ in range(5):
        trainer.training_step()
        assert trainer.last_feasibility_violation_rate == 0.0


def test_support_trainer_feasibility_diagnostic_detects_violations():
    """The diagnostic must actually be able to fail, or it proves nothing."""
    rate = MDNSupportTrainer._feasibility_violation_rate(
        torch.tensor(
            [
                [0.8, 0.4],   # feasible
                [0.4, 0.4],   # sum < 1 -> empty region
                [1.4, 0.2],   # s_i > 1
                [-0.1, 1.2],  # s_i < 0
            ]
        )
    )

    assert abs(rate - 0.75) < 1e-9


def test_support_trainer_diagnostic_is_zero_for_feasible_batch():
    rate = MDNSupportTrainer._feasibility_violation_rate(
        torch.tensor([[0.8, 0.4], [1.0, 1.0], [0.5, 0.5]])
    )

    assert rate == 0.0


def test_support_trainer_rejects_legacy_support_checkpoint(tmp_path: Path):
    """SASP changed the head width, so optimizer state shapes changed too.

    from_checkpoint restores optimizer_state_dict for an optimizer over
    support_head.parameters(); a pre-SASP checkpoint must be rejected with the
    shared migration error rather than failing opaquely inside Adam.
    """
    import pytest

    from utils.mdn_checkpoint_loader import IncompatibleCheckpointError

    model = MotiveDecompositionNetwork()
    store = _store_with_targets()
    trainer = MDNSupportTrainer(model, store, config=SupportTrainerConfig(), device="cpu")
    trainer.training_step()

    checkpoint_path = tmp_path / "legacy_support.pth"
    trainer.save_checkpoint(checkpoint_path)

    # Rewrite the saved head to its pre-SASP M-wide shape.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    state["support_head.weight"] = state["support_head.weight"][:2]
    state["support_head.bias"] = state["support_head.bias"][:2]
    torch.save(payload, checkpoint_path)

    restored_model = MotiveDecompositionNetwork()
    with pytest.raises(IncompatibleCheckpointError, match="legacy support head"):
        MDNSupportTrainer.from_checkpoint(
            checkpoint_path, restored_model, store, device="cpu"
        )

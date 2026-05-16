from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import torch

from generator.mdn import MotiveDecompositionNetwork


def test_auxiliary_model_single_and_batch_output_shapes():
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)

    single_context = torch.randn(14)
    single_skill_id = torch.tensor(3)
    batch_context = torch.randn(4, 14)
    batch_skill_id = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    single_gate_logits, single_q_hat = model.forward_auxiliary(single_context, single_skill_id)
    batch_gate_logits, batch_q_hat = model.forward_auxiliary(batch_context, batch_skill_id)

    assert single_gate_logits.shape == ()
    assert single_q_hat.shape == (2,)
    assert batch_gate_logits.shape == (4,)
    assert batch_q_hat.shape == (4, 2)


def test_auxiliary_model_outputs_are_finite():
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
    context = torch.randn(5, 14)
    skill_id = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)

    gate_logits, q_hat = model.forward_auxiliary(context, skill_id)

    assert torch.isfinite(gate_logits).all()
    assert torch.isfinite(q_hat).all()


def test_auxiliary_model_skill_input_changes_outputs():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
    context = torch.randn(14)

    gate_logits_a, q_hat_a = model.forward_auxiliary(context, torch.tensor(1))
    gate_logits_b, q_hat_b = model.forward_auxiliary(context, torch.tensor(2))

    assert not torch.allclose(gate_logits_a, gate_logits_b)
    assert not torch.allclose(q_hat_a, q_hat_b)


def test_auxiliary_model_checkpoint_round_trip():
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
    context = torch.randn(4, 14)
    skill_id = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    expected_gate_logits, expected_q_hat = model.forward_auxiliary(context, skill_id)

    save_path = Path.cwd() / f"mdn_auxiliary_test_{uuid4().hex}.pt"
    try:
        torch.save(model.state_dict(), save_path)
        restored = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
        restored.load_state_dict(torch.load(save_path, map_location="cpu"))
        restored_gate_logits, restored_q_hat = restored.forward_auxiliary(context, skill_id)

        assert save_path.exists()
        assert torch.allclose(expected_gate_logits, restored_gate_logits)
        assert torch.allclose(expected_q_hat, restored_q_hat)
    finally:
        if save_path.exists():
            save_path.unlink()

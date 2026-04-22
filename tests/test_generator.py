"""
Skill Generator Validation Test
Verifies output contracts, gradient flow, and save/load behavior
This test must pass before generator training logic is introduced.
"""

from pathlib import Path
from uuid import uuid4
import torch

from generator.skill_generator import SkillGenerator

def test_generator_single_and_batch_output_shapes():
    """Validate output tensor shapes for single and batched observations."""
    print("Testing Generator Output Shapes...")
    model = SkillGenerator()

    single_obs = torch.randn(8)
    batch_obs = torch.randn(4, 8)

    single_payoff, single_motives = model(single_obs)
    batch_payoff, batch_motives = model(batch_obs)

    # Single-sample output checks.
    assert single_payoff.shape == (1,)
    assert single_motives.shape == (2,)
    # Batched output checks.
    assert batch_payoff.shape == (4, 1)
    assert batch_motives.shape == (4, 2)
    
    print("Shape checks passed.\n")

def test_generator_rejects_invalid_input_shape():
    """Model should fail fast on tensors that are neither (8,) nor (N, 8)."""
    print("Testing Invalid Input Shape Handling...")
    model = SkillGenerator()
    bad_obs = torch.randn(8, 1)

    try:
        model(bad_obs)
    except ValueError as exc:
        assert "Expected" in str(exc)
        assert "(8, 1)" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid input shape")

    print("Invalid-shape check passed.\n")

def test_generator_gradients_flow_through_trunk_and_both_heads():
    """Loss built from both heads should backpropagate through the full model."""
    print("Testing Gradient Flow...")
    model = SkillGenerator()
    obs = torch.randn(3, 8)

    payoff, motives = model(obs)
    # Include both heads in loss so both receive gradient signal.
    loss = payoff.sum() + motives.sum()
    loss.backward()

    first_trunk_layer = model.trunk[0]

    # Trunk and both heads must receive gradients.
    assert first_trunk_layer.weight.grad is not None
    assert model.payoff_head.weight.grad is not None
    assert model.motive_head.weight.grad is not None

    print("Gradient checks passed.\n")

def test_generator_save_and_load_restores_outputs():
    """A saved and reloaded model should reproduce the same outputs."""
    print("Testing Save/Load Roundtrip...")
    model = SkillGenerator()
    obs = torch.randn(8)
    expected_payoff, expected_motives = model(obs)

    # Example output for quick manual inspection in `-s` test mode.
    print("Example generator output:")
    print(f"  payoff: {expected_payoff.detach()}")
    print(f"  motives: {expected_motives.detach()}")

    save_path = Path.cwd() / f"skill_generator_test_{uuid4().hex}.pt"

    try:
        model.save(save_path)

        restored_model = SkillGenerator()
        restored_model.load(save_path, map_location="cpu")

        restored_payoff, restored_motives = restored_model(obs)

        # File must exist and outputs should match after roundtrip.
        assert save_path.exists()
        assert torch.allclose(restored_payoff, expected_payoff)
        assert torch.allclose(restored_motives, expected_motives)

    finally:
        if save_path.exists():
            save_path.unlink()

    print("Save/load checks passed.\n")

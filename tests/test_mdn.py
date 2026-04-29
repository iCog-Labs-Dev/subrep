"""Validation tests for the SubRep Motive Decomposition Network."""

import pytest
import torch

from generator.mdn import MotiveDecompositionNetwork


def test_mdn_single_input_shape():
    """Single context inputs should preserve unbatched output shapes."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(14)

    weight_params, support_values = model(context)

    assert weight_params.shape == (2,)
    assert support_values.shape == (2,)


def test_mdn_batched_input_shape():
    """Batched context inputs should preserve the batch dimension."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 14)

    weight_params, support_values = model(context)

    assert weight_params.shape == (5, 2)
    assert support_values.shape == (5, 2)


def test_mdn_dirichlet_alpha_parameters_are_strictly_positive():
    """Dirichlet alpha parameters must always be strictly positive."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 14)

    weight_params, _ = model(context)

    assert torch.all(weight_params > 0)


def test_mdn_support_values_are_bounded():
    """Support geometry outputs should stay in the closed unit interval."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 14)

    _, support_values = model(context)

    assert torch.all(support_values >= 0)
    assert torch.all(support_values <= 1)


def test_mdn_outputs_are_finite():
    """Both heads should produce finite tensors without NaN or Inf values."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 14)

    weight_params, support_values = model(context)

    assert torch.isfinite(weight_params).all()
    assert torch.isfinite(support_values).all()


def test_mdn_gradient_flow_reaches_parameters_and_input():
    """A simple scalar loss should backpropagate through both heads and input."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 14, requires_grad=True)

    weight_params, support_values = model(context)
    loss = weight_params.sum() + support_values.sum()
    loss.backward()

    assert context.grad is not None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mdn_rejects_invalid_input_dimension():
    """Model should raise a clear error when the feature dimension is wrong."""
    model = MotiveDecompositionNetwork()
    context = torch.randn(13)

    with pytest.raises(ValueError, match=r"Expected single context shape \(14,\)"):
        model(context)


def test_mdn_heads_are_independent_modules():
    """Distribution and support predictions must come from separate heads."""
    model = MotiveDecompositionNetwork()

    assert hasattr(model, "distribution_head")
    assert hasattr(model, "support_head")
    assert model.distribution_head is not model.support_head

"""Validation tests for the SubRep Motive Decomposition Network."""

import pytest
import torch

from generator.mdn import MotiveDecompositionNetwork


def test_mdn_single_input_shape():
    """Single context inputs should preserve unbatched output shapes."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(8)

    weight_params, support_values = model(context)

    assert weight_params.shape == (2,)
    assert support_values.shape == (2,)


def test_mdn_batched_input_shape():
    """Batched context inputs should preserve the batch dimension."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 8)

    weight_params, support_values = model(context)

    assert weight_params.shape == (5, 2)
    assert support_values.shape == (5, 2)


def test_mdn_dirichlet_alpha_parameters_are_strictly_positive():
    """Dirichlet alpha parameters must always be strictly positive."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 8)

    weight_params, _ = model(context)

    assert torch.all(weight_params > 0)


def test_mdn_support_values_are_non_negative():
    """Support values must satisfy the full SASP feasibility contract."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 8)

    _, support_values = model(context)

    assert torch.all(support_values >= 0)
    assert torch.all(support_values <= 1)
    assert torch.all(torch.sum(support_values, dim=-1) >= 1.0)


def test_mdn_two_objective_support_values_are_feasible_for_single_context():
    """Two-objective support values should define a non-empty W_x interval."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=2)
    context = torch.randn(8)

    _, support_values = model.forward_inference(context)

    assert support_values.shape == (2,)
    assert torch.all(support_values >= 0)
    assert torch.all(support_values <= 1)
    assert torch.sum(support_values) >= 1.0


def test_mdn_two_objective_support_values_are_feasible_for_batched_contexts():
    """Batched two-objective support values should all define non-empty W_x intervals."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=2)
    context = torch.randn(5, 8)

    _, support_values = model.forward_inference(context)

    assert support_values.shape == (5, 2)
    assert torch.all(support_values >= 0)
    assert torch.all(support_values <= 1)
    assert torch.all(torch.sum(support_values, dim=-1) >= 1.0)


@pytest.mark.parametrize("num_objectives", [2, 3, 5, 10, 50])
def test_mdn_support_values_feasible_for_any_M(num_objectives):
    """SASP must guarantee feasibility at every M, including extreme inputs.

    Replaces test_mdn_non_two_objective_support_values_keep_softplus_path,
    which pinned the broken raw-Softplus behavior (asserting support > 1.0)
    as if it were the intended contract.
    """
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=num_objectives)
    context = torch.randn(64, 8) * 10.0  # include extreme-magnitude inputs

    _, support_values = model.forward_inference(context)

    assert support_values.shape == (64, num_objectives)
    assert torch.all(support_values >= 0)
    assert torch.all(support_values <= 1)
    assert torch.all(torch.sum(support_values, dim=-1) >= 1.0 - 1e-6)


@pytest.mark.parametrize("num_objectives", [2, 5, 10])
def test_mdn_support_values_feasible_for_extreme_logits(num_objectives):
    """Feasibility must not depend on training state or logit magnitude."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=num_objectives)

    for magnitude in (-50.0, -1.0, 0.0, 1.0, 50.0):
        raw = torch.full((32, 2 * num_objectives), magnitude)
        support_values = model._support_values_from_raw(raw)

        assert torch.all(support_values >= 0)
        assert torch.all(support_values <= 1)
        assert torch.all(torch.sum(support_values, dim=-1) >= 1.0 - 1e-6)

    # Mixed extremes across the two logit groups.
    raw = torch.cat(
        [
            torch.full((32, num_objectives), -50.0),
            torch.full((32, num_objectives), 50.0),
        ],
        dim=-1,
    )
    support_values = model._support_values_from_raw(raw)
    assert torch.all(support_values >= 0)
    assert torch.all(support_values <= 1)
    assert torch.all(torch.sum(support_values, dim=-1) >= 1.0 - 1e-6)


def test_sasp_is_permutation_equivariant():
    """Permuting objective indices must permute the output identically.

    This is the property the sequential (Approach 1) chaining lacks: it
    privileges whichever objective is indexed first.
    """
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(num_objectives=5)
    raw = torch.randn(16, 10)
    perm = torch.randperm(5)
    permuted_raw = torch.cat(
        [raw[..., :5][..., perm], raw[..., 5:][..., perm]], dim=-1
    )

    support_values = model._support_values_from_raw(raw)
    permuted_support = model._support_values_from_raw(permuted_raw)

    assert torch.allclose(support_values[..., perm], permuted_support, atol=1e-6)


def test_sequential_reparameterization_is_not_permutation_equivariant():
    """Evidence that the Approach 1 baseline is order-biased (paper E3).

    The chained construction is what SASP replaces. Showing it fails the same
    equivariance check makes the ordering-bias claim concrete rather than
    assumed.
    """

    def sequential_support(raw: torch.Tensor) -> torch.Tensor:
        """Approach 1: s_i claims a share of the remaining budget in order."""
        values = []
        remaining = torch.ones(raw.shape[:-1])
        for index in range(raw.shape[-1]):
            share = torch.sigmoid(raw[..., index]) * remaining
            values.append(1.0 - remaining + share)
            remaining = remaining - share
        return torch.stack(values, dim=-1)

    torch.manual_seed(0)
    raw = torch.randn(16, 5)
    perm = torch.randperm(5)
    while torch.equal(perm, torch.arange(5)):
        perm = torch.randperm(5)

    support_values = sequential_support(raw)
    permuted_support = sequential_support(raw[..., perm])

    assert not torch.allclose(support_values[..., perm], permuted_support, atol=1e-6)


@pytest.mark.parametrize("num_objectives", [2, 3, 5])
def test_sasp_surjectivity_round_trip(num_objectives):
    """Unfloored SASP reaches every feasible support vector (paper section 8.3).

    Uses the analytic witness from the proof: p_i = s_i / sum(s) and
    g_i = (s_i - p_i) / (1 - p_i). The witness is valid for any target with
    s_i in (0, 1) and sum(s) > 1, because sum(s) >= 1 is exactly what makes
    p_i <= s_i -- surjectivity and feasibility are the same inequality.

    Tested at slack_floor = 0.0, where the reachable set is the full feasible
    set. See test_sasp_floored_round_trip for the floored variant.
    """
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(
        num_objectives=num_objectives, slack_floor=0.0
    )

    for _ in range(64):
        # Sample an arbitrary feasible target: s_i in (0, 1), sum(s) > 1.
        base = torch.rand(num_objectives)
        base = base / base.sum()
        gate = torch.rand(num_objectives).clamp(0.01, 0.99)
        target = base + (1.0 - base) * gate

        assert torch.all(target > 0.0) and torch.all(target < 1.0)
        assert float(target.sum()) > 1.0

        # Recover the witness from the target alone, as the proof prescribes.
        witness_p = target / target.sum()
        witness_g = (target - witness_p) / (1.0 - witness_p)

        assert torch.all(witness_g >= 0.0) and torch.all(witness_g < 1.0)

        # Invert softmax (up to an additive constant) and sigmoid.
        base_logits = torch.log(witness_p)
        gate_logits = torch.log(witness_g / (1.0 - witness_g))
        raw = torch.cat([base_logits, gate_logits], dim=-1)

        reconstructed = model._support_values_from_raw(raw)

        assert torch.allclose(reconstructed, target, atol=1e-5)


@pytest.mark.parametrize("num_objectives", [2, 3, 5])
def test_sasp_floored_round_trip(num_objectives):
    """A floored gate still reaches every target inside its reachable set.

    The slack floor trades a thin boundary shell of the feasible set for the
    guarantee that W_x never collapses to a point. Within the floored set the
    parameterization remains exactly invertible, so no expressiveness is lost
    beyond that deliberately excluded shell.
    """
    torch.manual_seed(0)
    slack_floor = 0.02
    model = MotiveDecompositionNetwork(
        num_objectives=num_objectives, slack_floor=slack_floor
    )

    for _ in range(64):
        base = torch.rand(num_objectives)
        base = base / base.sum()
        gate = (
            slack_floor + (1.0 - slack_floor) * torch.rand(num_objectives)
        ).clamp(slack_floor + 1e-3, 1.0 - 1e-3)
        target = base + (1.0 - base) * gate

        base_logits = torch.log(base)
        gate_fraction = (gate - slack_floor) / (1.0 - slack_floor)
        gate_logits = torch.log(gate_fraction / (1.0 - gate_fraction))
        raw = torch.cat([base_logits, gate_logits], dim=-1)

        reconstructed = model._support_values_from_raw(raw)

        assert torch.allclose(reconstructed, target, atol=1e-5)


def test_sasp_respects_slack_floor():
    """The region must never collapse to a point, even at extreme logits."""
    torch.manual_seed(0)
    slack_floor = 0.05
    model = MotiveDecompositionNetwork(num_objectives=4, slack_floor=slack_floor)

    raw = torch.full((8, 8), 0.0)
    raw[..., 4:] = -50.0  # drive every slack gate toward its floor
    support_values = model._support_values_from_raw(raw)

    base_allocation = torch.softmax(raw[..., :4], dim=-1)
    floor_values = base_allocation + (1.0 - base_allocation) * slack_floor

    assert torch.all(support_values >= floor_values - 1e-6)
    # Strictly wider than the degenerate s = p region.
    assert torch.all(support_values > base_allocation)


def test_support_head_width_is_twice_num_objectives():
    """SASP consumes 2M logits; guards against a partial revert."""
    for num_objectives in (2, 3, 7):
        model = MotiveDecompositionNetwork(num_objectives=num_objectives)
        assert model.support_head.out_features == 2 * num_objectives


def test_support_head_rejects_wrong_logit_width():
    """A stale M-wide head must fail loudly, not be silently sliced."""
    model = MotiveDecompositionNetwork(num_objectives=3)
    raw = torch.randn(4, 3)  # legacy width

    with pytest.raises(ValueError, match=r"2\*M=6"):
        model._support_values_from_raw(raw)


def test_mdn_rejects_invalid_slack_floor():
    """slack_floor must stay in [0, 1) for the guarantees to hold."""
    for invalid in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match="slack_floor"):
            MotiveDecompositionNetwork(num_objectives=2, slack_floor=invalid)


def test_mdn_support_activation_is_removed():
    """The raw-Softplus support path must be gone, not merely bypassed."""
    model = MotiveDecompositionNetwork(num_objectives=3)
    assert not hasattr(model, "support_activation")


def test_mdn_outputs_are_finite():
    """Both heads should produce finite tensors without NaN or Inf values."""
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 8)

    weight_params, support_values = model(context)

    assert torch.isfinite(weight_params).all()
    assert torch.isfinite(support_values).all()


def test_mdn_synthetic_gradient_flow_reaches_parameters_and_input():
    """A synthetic combined loss should backpropagate through both heads and input.
    """
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    context = torch.randn(5, 8, requires_grad=True)

    weight_params, support_values = model(context)
    loss = weight_params.sum() + support_values.sum()
    loss.backward()

    assert context.grad is not None
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_mdn_rejects_invalid_input_dimension():
    """Model should raise a clear error when the feature dimension is wrong."""
    model = MotiveDecompositionNetwork()
    context = torch.randn(7)

    with pytest.raises(ValueError, match=r"Expected single context shape \(8,\)"):
        model(context)


def test_mdn_heads_are_independent_modules():
    """Distribution and support predictions must come from separate heads."""
    model = MotiveDecompositionNetwork()

    assert hasattr(model, "distribution_head")
    assert hasattr(model, "support_head")
    assert model.distribution_head is not model.support_head

"""
Motive Decomposition Network Validation Tests.

Verifies MDN output shapes and positivity for Dirichlet concentration params.
"""

from pathlib import Path
from uuid import uuid4

import torch

from generator.mdn import MotiveDecompositionNetwork


def test_mdn_single_and_batch_output_shapes():
	"""Validate output tensor shapes for single and batched context features."""
	model = MotiveDecompositionNetwork()

	single_context = torch.randn(14)
	batch_context = torch.randn(4, 14)

	single_weight_params = model(single_context)
	batch_weight_params = model(batch_context)

	assert single_weight_params.shape == (2,)
	assert batch_weight_params.shape == (4, 2)


def test_mdn_weight_params_are_positive():
	"""Softplus should keep Dirichlet concentration parameters positive."""
	model = MotiveDecompositionNetwork()

	single_context = torch.randn(14)
	batch_context = torch.randn(3, 14)

	single_weight_params = model(single_context)
	batch_weight_params = model(batch_context)

	assert torch.all(single_weight_params > 0)
	assert torch.all(batch_weight_params > 0)


def test_mdn_rejects_invalid_input_shape():
	"""Model should fail fast on tensors that are neither (14,) nor (N, 14)."""
	model = MotiveDecompositionNetwork()
	bad_context = torch.randn(14, 1)

	try:
		model(bad_context)
	except ValueError as exc:
		assert "Expected" in str(exc)
		assert "(14, 1)" in str(exc)
	else:
		raise AssertionError("Expected ValueError for invalid input shape")


def test_mdn_save_and_load_restores_outputs():
	"""A saved and reloaded model should reproduce the same outputs."""
	model = MotiveDecompositionNetwork()
	context = torch.randn(14)
	expected_weight_params = model(context)

	save_path = Path.cwd() / f"mdn_test_{uuid4().hex}.pt"

	try:
		model.save(save_path)

		restored_model = MotiveDecompositionNetwork()
		restored_model.load(save_path, map_location="cpu")

		restored_weight_params = restored_model(context)

		assert save_path.exists()
		assert torch.allclose(restored_weight_params, expected_weight_params)
	finally:
		if save_path.exists():
			save_path.unlink()

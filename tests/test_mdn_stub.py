"""
test_mdn_stub.py — Unit tests for the deterministic MDN testing stub.

Run with:
    python -m pytest tests/test_mdn_stub.py -v
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_runtime_selector import MDNRuntimeSelector
from utils.mdn_stub import StubMDN, load_mdn_or_stub


class TestStubMDN:
    def test_single_obs_forward(self):
        stub = StubMDN(
            input_dim=8,
            num_objectives=2,
            fixed_alpha=[2.5, 3.5],
            fixed_support_values=[0.1, 0.9],
        )
        obs = torch.zeros(8, dtype=torch.float32)

        alpha, support = stub.forward_inference(obs)

        assert alpha.shape == (2,)
        assert support.shape == (2,)
        assert alpha.tolist() == [2.5, 3.5]
        # Use approximate comparison for float32 precision
        support_list = support.tolist()
        assert abs(support_list[0] - 0.1) < 1e-6, f"Expected ~0.1, got {support_list[0]}"
        assert abs(support_list[1] - 0.9) < 1e-6, f"Expected ~0.9, got {support_list[1]}"

    def test_batched_obs_forward(self):
        stub = StubMDN(
            input_dim=8,
            num_objectives=2,
            fixed_alpha=[5.0, 5.0],
            fixed_support_values=[1.0, 1.0],
        )
        batch_size = 5
        obs = torch.zeros((batch_size, 8), dtype=torch.float32)

        alpha, support = stub.forward_inference(obs)

        assert alpha.shape == (batch_size, 2)
        assert support.shape == (batch_size, 2)
        
        # Verify all batches get identical deterministic values
        for i in range(batch_size):
            assert alpha[i].tolist() == [5.0, 5.0]
            assert support[i].tolist() == [1.0, 1.0]

    def test_validation_errors(self):
        stub = StubMDN(input_dim=8)

        with pytest.raises(ValueError, match="shape"):
            stub.forward_inference(torch.zeros(7))  # Wrong single dim

        with pytest.raises(ValueError, match="shape"):
            stub.forward_inference(torch.zeros((4, 7)))  # Wrong batch dim

        with pytest.raises(ValueError, match="shape"):
            stub.forward_inference(torch.zeros((1, 2, 8)))  # Wrong ndim

    def test_device_mock(self):
        stub = StubMDN()
        stub = stub.to("cpu")  # Should not raise
        assert stub.device == "cpu"
        # Test that .to() returns self for method chaining
        returned_stub = stub.to("cpu")
        assert returned_stub is stub  # Verify it returns self


class TestLoadMDNOrStub:
    def test_missing_file_returns_stub(self):
        model = load_mdn_or_stub("does_not_exist_xyz.pt")
        assert isinstance(model, StubMDN)
        assert model.input_dim == 8

    def test_corrupted_file_returns_stub_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "corrupt.pt"
            bad_path.write_text("this is not a valid pytorch checkpoint", encoding="utf-8")

            # Must catch the torch.load exception internally and return a stub
            model = load_mdn_or_stub(bad_path)
            assert isinstance(model, StubMDN)

    def test_valid_checkpoint_returns_real_mdn(self):
        # Create a tiny real model and save it
        real_model = MotiveDecompositionNetwork(input_dim=4, num_objectives=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "valid.pt"
            torch.save(real_model.state_dict(), ckpt_path)

            loaded_model = load_mdn_or_stub(ckpt_path, input_dim=4)
            # Should be the real PyTorch module, not the stub
            assert isinstance(loaded_model, MotiveDecompositionNetwork)
            assert not isinstance(loaded_model, StubMDN)
            assert loaded_model.input_dim == 4

    def test_m5_checkpoint_returns_real_mdn(self):
        """SASP is feasible at any M, so loading must not be M=2-specific."""
        real_model = MotiveDecompositionNetwork(input_dim=8, num_objectives=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "valid_m5.pt"
            torch.save(real_model.state_dict(), ckpt_path)

            loaded = load_mdn_or_stub(ckpt_path, input_dim=8, num_objectives=5)

            assert isinstance(loaded, MotiveDecompositionNetwork)
            assert loaded.num_objectives == 5
            assert loaded.support_head.out_features == 10


class TestLegacyCheckpointMigration:
    """Pre-SASP checkpoints must fail loudly, never be reinterpreted.

    SASP widened the support head from M to 2M outputs. A legacy checkpoint's
    weights are not a subset of the new head's meaning, so loading them would
    silently produce wrong support geometry -- the exact class of invisible
    failure this task exists to remove.
    """

    @staticmethod
    def _write_legacy_checkpoint(path: Path, num_objectives: int = 2) -> None:
        """Build a state dict whose support head has the pre-SASP width M."""
        model = MotiveDecompositionNetwork(
            input_dim=8, num_objectives=num_objectives, hidden_dim=16
        )
        state = dict(model.state_dict())
        # Shrink the support head back to its legacy M-wide shape.
        state["support_head.weight"] = state["support_head.weight"][:num_objectives]
        state["support_head.bias"] = state["support_head.bias"][:num_objectives]
        torch.save({"model_state_dict": state}, path)

    def test_load_mdn_or_stub_falls_back_on_legacy_head(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.pth"
            self._write_legacy_checkpoint(path)

            model = load_mdn_or_stub(path, input_dim=8, num_objectives=2)

            # Falls back rather than reinterpreting weights.
            assert isinstance(model, StubMDN)

    def test_load_mdn_checkpoint_raises_actionable_error(self):
        from utils.mdn_checkpoint_loader import (
            IncompatibleCheckpointError,
            load_mdn_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.pth"
            self._write_legacy_checkpoint(path)

            # This loader has no stub contract, so it must propagate.
            with pytest.raises(IncompatibleCheckpointError) as exc_info:
                load_mdn_checkpoint(path)

            message = str(exc_info.value)
            assert "legacy support head" in message
            assert "2*M=4" in message
            # The message must tell the operator what to actually do.
            assert "train_mdn_candidate_sets" in message

    def test_sasp_checkpoint_passes_compatibility_check(self):
        from utils.mdn_checkpoint_loader import (
            assert_support_head_compatible,
            load_mdn_checkpoint,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sasp.pth"
            model = MotiveDecompositionNetwork(
                input_dim=8, num_objectives=3, hidden_dim=16
            )
            torch.save({"model_state_dict": model.state_dict()}, path)

            assert_support_head_compatible(model.state_dict())  # must not raise
            loaded = load_mdn_checkpoint(path)

            assert loaded.num_objectives == 3
            assert loaded.support_head.out_features == 6

    def test_compatibility_check_tolerates_partial_state_dicts(self):
        """A state dict without the relevant keys is not our failure to report."""
        from utils.mdn_checkpoint_loader import assert_support_head_compatible

        assert_support_head_compatible({})  # must not raise
        assert_support_head_compatible(
            {"support_head.weight": torch.zeros(4, 8)}  # no distribution head
        )

    def test_stub_functions_in_mdn_runtime_selector(self):
        """Prove that the stub is interface-compatible with the main runtime selector."""
        stub = StubMDN(fixed_alpha=[2.0, 2.0], fixed_support_values=[1.0, 1.0])
        # The selector calls model.eval() and model.to(device) internally
        selector = MDNRuntimeSelector(stub)
        
        # Build fake certified candidate
        from utils.mdn_contracts import CandidateSkillRecord
        candidates = [
            CandidateSkillRecord(
                skill_id="test_candidate",
                delta_r=5.0,
                delta_n=(2.0, 3.0),
                is_certified=True,
                gate_type="CDS",
                admission_margin=7.0,
                epsilon=0.0,
                baseline_id=None,
            )
        ]
        
        obs = np.zeros(8, dtype=np.float32)
        result = selector.select(obs, candidates)

        assert result.selected_skill_id == "test_candidate"
        # Ensure our fixed stub values propagated fully through the selector
        assert np.array_equal(result.alpha, np.array([2.0, 2.0], dtype=np.float32))
        assert np.array_equal(result.support_values, np.array([1.0, 1.0], dtype=np.float32))

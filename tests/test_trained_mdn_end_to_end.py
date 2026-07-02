"""Tests for trained MDN checkpoint integration.

This module validates that the MDN loader can:
1. Infer model dimensions from checkpoint state_dict
2. Load real trained MDN checkpoints
3. Fall back to StubMDN when checkpoint is missing
4. Include MDN metadata in admission reports
5. Run MDN selection with trained checkpoints
"""
from __future__ import annotations

import os
import tempfile
import pytest
import torch
from pathlib import Path

from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_stub import load_mdn_or_stub, StubMDN
from utils.admission_report import AdmissionReport


class TestTrainedMDNIntegration:
    """Test trained MDN checkpoint loading and integration."""
    
    def test_loader_infers_dimensions_from_checkpoint(self):
        """Verify loader can infer model dimensions from checkpoint without explicit parameters."""
        # Create a tiny MDN with known dimensions
        model = MotiveDecompositionNetwork(
            input_dim=8,
            num_objectives=2,
            hidden_dim=32,
            num_hidden_layers=2,
            num_skills=64,
            skill_embedding_dim=4,
        )
        
        # Save to temporary checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        
        try:
            # Load using our loader (dimensions should be inferred)
            loaded_model = load_mdn_or_stub(
                checkpoint_path=checkpoint_path,
                input_dim=8,  # These should be ignored
                num_objectives=2,
            )
            
            # Verify it's not a stub
            assert not isinstance(loaded_model, StubMDN), "Should load real model, not stub"
            assert isinstance(loaded_model, MotiveDecompositionNetwork), "Should be MDN instance"
            
            # Verify dimensions match
            assert loaded_model.input_dim == 8, f"Expected input_dim=8, got {loaded_model.input_dim}"
            assert loaded_model.num_objectives == 2, f"Expected num_objectives=2, got {loaded_model.num_objectives}"
        finally:
            os.unlink(checkpoint_path)
    
    def test_fallback_to_stub_when_checkpoint_missing(self):
        """Verify loader falls back to StubMDN when checkpoint doesn't exist."""
        loaded_model = load_mdn_or_stub(
            checkpoint_path="models/nonexistent_checkpoint.pth",
            input_dim=8,
            num_objectives=2,
        )
        
        assert isinstance(loaded_model, StubMDN), "Should fall back to StubMDN"
        assert loaded_model.input_dim == 8, "Stub should have correct input_dim"
        assert loaded_model.num_objectives == 2, "Stub should have correct num_objectives"
    
    def test_admission_report_includes_mdn_metadata(self):
        """Verify admission report includes MDN metadata when set."""
        report = AdmissionReport()
        
        # Set MDN metadata
        report.set_mdn_metadata(
            source="trained_checkpoint",
            checkpoint_path="models/mdn_policy_best.pth",
            alpha_values=[0.7, 0.3],
            derived_weights=[0.7, 0.3],
            support_values=[0.2, 0.8],
            support_geometry_feasible=True,
        )
        
        # Compile report
        stats = report.compile()
        
        # Verify metadata is included
        assert "mdn_source" in stats, "Report should include mdn_source"
        assert stats["mdn_source"] == "trained_checkpoint", "Should match source"
        assert stats["checkpoint_path"] == "models/mdn_policy_best.pth", "Should match path"
        assert stats["alpha_values"] == [0.7, 0.3], "Should match alpha"
        assert stats["derived_weights"] == [0.7, 0.3], "Should match weights"
        assert stats["support_values"] == [0.2, 0.8], "Should match support"
        assert stats["support_geometry_feasible"] is True, "Should match feasibility"
    
    def test_mdn_selection_with_trained_checkpoint(self):
        """Verify MDN can run forward_inference with trained checkpoint."""
        # Create tiny trained MDN
        model = MotiveDecompositionNetwork(
            input_dim=8,
            num_objectives=2,
            hidden_dim=32,
            num_hidden_layers=2,
        )
        
        # Save to temporary checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        
        try:
            # Load using our loader
            loaded_model = load_mdn_or_stub(
                checkpoint_path=checkpoint_path,
                input_dim=8,
                num_objectives=2,
            )
            
            # Verify it can run forward_inference
            obs = torch.tensor([0.1] * 8, dtype=torch.float32)
            alpha, support = loaded_model.forward_inference(obs)
            
            # Verify outputs are valid
            assert alpha.shape == (2,), f"Expected alpha shape (2,), got {alpha.shape}"
            assert support.shape == (2,), f"Expected support shape (2,), got {support.shape}"
            assert torch.all(alpha > 0), "Alpha should be positive"
            assert torch.all(support >= 0), "Support should be non-negative"
        finally:
            os.unlink(checkpoint_path)
    
    def test_loader_handles_both_checkpoint_formats(self):
        """Verify loader handles both {'model_state_dict': ...} and raw state_dict formats."""
        model = MotiveDecompositionNetwork(
            input_dim=8,
            num_objectives=2,
            hidden_dim=32,
            num_hidden_layers=2,
        )
        
        # Test format 1: wrapped in dict
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        
        try:
            loaded_model = load_mdn_or_stub(
                checkpoint_path=checkpoint_path,
                input_dim=8,
                num_objectives=2,
            )
            assert isinstance(loaded_model, MotiveDecompositionNetwork)
        finally:
            os.unlink(checkpoint_path)
        
        # Test format 2: raw state_dict
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            checkpoint_path = f.name
            torch.save(model.state_dict(), checkpoint_path)
        
        try:
            loaded_model = load_mdn_or_stub(
                checkpoint_path=checkpoint_path,
                input_dim=8,
                num_objectives=2,
            )
            assert isinstance(loaded_model, MotiveDecompositionNetwork)
        finally:
            os.unlink(checkpoint_path)
    
    def test_admission_report_without_mdn_metadata(self):
        """Verify admission report works without MDN metadata (backward compatibility)."""
        report = AdmissionReport()
        
        # Don't set MDN metadata
        stats = report.compile()
        
        # Verify report still works
        assert "total_attempted" in stats
        assert "admitted" in stats
        assert "rejected" in stats
        
        # Verify MDN metadata is NOT included
        assert "mdn_source" not in stats, "Should not include mdn_source when not set"
    
    def test_stub_alpha_and_support_values(self):
        """Verify StubMDN returns expected fixed alpha and support values."""
        stub = StubMDN(
            input_dim=8,
            num_objectives=2,
            fixed_alpha=[2.0, 2.0],
            fixed_support_values=[1.0, 1.0],
        )
        
        obs = torch.tensor([0.1] * 8, dtype=torch.float32)
        alpha, support = stub.forward_inference(obs)
        
        assert alpha.tolist() == [2.0, 2.0], "Stub should return fixed alpha"
        assert support.tolist() == [1.0, 1.0], "Stub should return fixed support"

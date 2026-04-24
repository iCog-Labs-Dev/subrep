"""
Certification Gates Validation Tests.

Verifies CDS and PDS admission gates work correctly for:
- Universally beneficial skills (pass CDS)
- Harmful skills (fail CDS)
- Trade-off skills (pass PDS within ε bounds)

Reference: SubRep Paper Section 3.2 (Definitions 1 & 2)
"""

import numpy as np
import pytest
from certification.gate import AdmissionGate
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from utils.cone_utils import (  
    validate_simplex_weights,
    compute_support_function,
    compute_worst_case_motive,
    normalize_weights,
)

# ============================================================================
# CDS Gate Tests
# ============================================================================

def test_cds_universally_beneficial_skill_passes():
    """Skill that improves all motives should pass CDS."""
    gate = CDSGate()
    delta_r = 0.5
    delta_n = np.array([0.3, 0.2])  # Both motives improve
    
    assert gate.admit(delta_r, delta_n) is True
    assert gate.get_admission_margin(delta_r, delta_n) > 0

def test_cds_harmful_skill_fails():
    """Skill that worsens payoff should fail CDS."""
    gate = CDSGate()
    delta_r = -0.5
    delta_n = np.array([0.1, 0.1])
    
    assert gate.admit(delta_r, delta_n) is False
    assert gate.get_admission_margin(delta_r, delta_n) < 0

def test_cds_trade_off_within_budget_fails():
    """CDS does not allow trade-offs (ε=0)."""
    gate = CDSGate()
    delta_r = 0.5
    delta_n = np.array([0.8, -0.6])  # Fuel worsens by 0.6
    
    # Δr + min(Δn) = 0.5 + (-0.6) = -0.1 < 0 → Fail
    assert gate.admit(delta_r, delta_n) is False

def test_cds_boundary_case():
    """Skill exactly at boundary (margin=0) should pass."""
    gate = CDSGate()
    delta_r = 0.5
    delta_n = np.array([0.5, -0.5])  # min = -0.5
    
    # Δr + min(Δn) = 0.5 + (-0.5) = 0.0 → Pass (≥ 0)
    assert gate.admit(delta_r, delta_n) is True
    assert np.isclose(gate.get_admission_margin(delta_r, delta_n), 0.0)

# ============================================================================
# PDS Gate Tests
# ============================================================================

def test_pds_trade_off_within_epsilon_passes():
    """Skill with trade-off within ε budget should pass PDS."""
    gate = PDSGate(epsilon=0.1)
    delta_r = 0.5
    delta_n = np.array([0.8, -0.6])  # Fuel worsens by 0.6
    
    # Δr + min(Δn) = 0.5 + (-0.6) = -0.1 ≥ -0.1 → Pass
    assert gate.admit(delta_r, delta_n) is True
    assert gate.get_admission_margin(delta_r, delta_n) >= 0

def test_pds_trade_off_exceeds_epsilon_fails():
    """Skill with trade-off exceeding ε budget should fail PDS."""
    gate = PDSGate(epsilon=0.1)
    delta_r = 0.5
    delta_n = np.array([0.8, -0.7])  # Fuel worsens by 0.7
    
    # Δr + min(Δn) = 0.5 + (-0.7) = -0.2 < -0.1 → Fail
    assert gate.admit(delta_r, delta_n) is False
    assert gate.get_admission_margin(delta_r, delta_n) < 0

def test_pds_zero_epsilon_equals_cds():
    """PDS with ε=0 should behave identically to CDS."""
    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=0.0)
    
    test_cases = [
        (0.5, np.array([0.3, 0.2])),
        (-0.5, np.array([0.1, 0.1])),
        (0.5, np.array([0.8, -0.6])),
    ]
    
    for delta_r, delta_n in test_cases:
        assert cds_gate.admit(delta_r, delta_n) == pds_gate.admit(delta_r, delta_n)

def test_pds_custom_epsilon():
    """PDS should respect custom epsilon values."""
    gate_small = PDSGate(epsilon=0.05)
    gate_large = PDSGate(epsilon=0.2)
    
    delta_r = 0.5
    delta_n = np.array([0.8, -0.6])  # Δr + min(Δn) = -0.1
    
    # -0.1 < -0.05 → Fail for small epsilon
    assert gate_small.admit(delta_r, delta_n) is False
    # -0.1 ≥ -0.2 → Pass for large epsilon
    assert gate_large.admit(delta_r, delta_n) is True

# ============================================================================
# Input Validation Tests
# ============================================================================

def test_gate_rejects_invalid_delta_r_type():
    """Gates should reject non-scalar delta_r."""
    gate = CDSGate()
    
    with pytest.raises(ValueError):
        gate.admit([0.5], np.array([0.1, 0.2]))

def test_gate_rejects_invalid_delta_n_shape():
    """Gates should reject non-1D delta_n."""
    gate = CDSGate()
    
    with pytest.raises(ValueError):
        gate.admit(0.5, np.array([[0.1, 0.2]]))

def test_gate_rejects_infinite_values():
    """Gates should reject infinite inputs."""
    gate = CDSGate()
    
    with pytest.raises(ValueError):
        gate.admit(np.inf, np.array([0.1, 0.2]))
    
    with pytest.raises(ValueError):
        gate.admit(0.5, np.array([np.inf, 0.2]))

# ============================================================================
# Cone Utilities Tests
# ============================================================================

def test_validate_simplex_weights_valid():
    """Valid simplex weights should pass validation."""
    weights = np.array([0.5, 0.5])
    assert validate_simplex_weights(weights) is True
    
    weights = np.array([1.0, 0.0, 0.0])
    assert validate_simplex_weights(weights) is True

def test_validate_simplex_weights_invalid_sum():
    """Weights not summing to 1 should fail validation."""
    weights = np.array([0.3, 0.3])  # Sum = 0.6
    assert validate_simplex_weights(weights) is False

def test_validate_simplex_weights_invalid_negative():
    """Negative weights should fail validation."""
    weights = np.array([1.5, -0.5])
    assert validate_simplex_weights(weights) is False

def test_compute_support_function():
    """Support function for simplex should return max component."""
    u = np.array([0.3, 0.7, 0.5])
    assert np.isclose(compute_support_function(u), 0.7)

def test_compute_worst_case_motive():
    """Worst-case motive should return min component."""
    delta_n = np.array([0.3, -0.5, 0.2])
    assert np.isclose(compute_worst_case_motive(delta_n), -0.5)

def test_normalize_weights():
    """Normalization should produce valid simplex weights."""
    weights = np.array([2.0, 1.0, 1.0])
    normalized = normalize_weights(weights)
    
    assert validate_simplex_weights(normalized) is True
    assert np.all(normalized >= 0)
    assert np.isclose(np.sum(normalized), 1.0)

def test_normalize_zero_weights():
    """Zero weights should normalize to uniform distribution."""
    weights = np.array([0.0, 0.0, 0.0])
    normalized = normalize_weights(weights)
    
    expected = np.array([1/3, 1/3, 1/3])
    assert np.allclose(normalized, expected)

# ============================================================================
# Gate Type Identification Tests
# ============================================================================

def test_gate_type_identifiers():
    """Gates should return correct type identifiers."""
    cds_gate = CDSGate()
    pds_gate = PDSGate()
    
    assert cds_gate.get_gate_type() == "CDS"
    assert pds_gate.get_gate_type() == "PDS"

# ============================================================================
# Integration Test: Full Certification Flow
# ============================================================================

def test_full_certification_flow():
    """Test complete certification workflow with multiple skills."""
    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=0.1)
    
    # Skill 1: Universally beneficial (should pass both)
    skill1_r, skill1_n = 0.8, np.array([0.5, 0.3])
    assert cds_gate.admit(skill1_r, skill1_n) is True
    assert pds_gate.admit(skill1_r, skill1_n) is True
    
    # Skill 2: Harmful (should fail both)
    skill2_r, skill2_n = -0.3, np.array([-0.1, -0.2])
    assert cds_gate.admit(skill2_r, skill2_n) is False
    assert pds_gate.admit(skill2_r, skill2_n) is False
    
    # Skill 3: Trade-off within PDS budget (fail CDS, pass PDS)
    skill3_r, skill3_n = 0.5, np.array([0.8, -0.6])
    assert cds_gate.admit(skill3_r, skill3_n) is False
    assert pds_gate.admit(skill3_r, skill3_n) is True
    
    # Skill 4: Trade-off exceeds PDS budget (fail both)
    skill4_r, skill4_n = 0.5, np.array([0.8, -0.7])
    assert cds_gate.admit(skill4_r, skill4_n) is False
    assert pds_gate.admit(skill4_r, skill4_n) is False
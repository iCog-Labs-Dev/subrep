from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from certification.certificate_schema import Certificate
from generator.mdn import MotiveDecompositionNetwork
from library.skill_library import SkillLibrary, _compute_wx_worst_case
from library.skill_metadata import FULL_SIMPLEX, MDN_WX, SkillEntry
from library.skill_selector import SkillSelector
from utils.mdn_selection import alpha_to_mean_weights
from utils.support_geometry import make_basis_query_directions

def _make_cert(
    skill_id: str = "s1",
    gate_type: str = "CDS",
    delta_r: float = 0.5,
    delta_n: tuple = (0.3, 0.2),
    epsilon: float = 0.0,
    margin: float = 0.5,
) -> Certificate:
    return Certificate(
        skill_id=skill_id,
        gate_type=gate_type,
        delta_r=delta_r,
        delta_n=delta_n,
        admission_margin=margin,
        epsilon=epsilon,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="idle_v1",
        environment="MO-LunarLander-v3",
        episode_length=200,
        version="0.1.0",
    )


def _make_entry(
    skill_id: str = "s1",
    gate_type: str = "CDS",
    delta_r: float = 0.5,
    delta_n: tuple = (0.3, 0.2),
    epsilon: float = 0.0,
    weight_region_type: str = FULL_SIMPLEX,
    certification_context: tuple = None,
    mdn_alpha: tuple = None,
    wx_support_directions: tuple = None,
    wx_support_values: tuple = None,
) -> SkillEntry:
    cert = _make_cert(
        skill_id=skill_id,
        gate_type=gate_type,
        delta_r=delta_r,
        delta_n=delta_n,
        epsilon=epsilon,
    )
    return SkillEntry(
        skill_id=skill_id,
        gate_type=gate_type,
        certificate=cert,
        policy=lambda obs: 0,
        weight_region_type=weight_region_type,
        certification_context=certification_context,
        mdn_alpha=mdn_alpha,
        wx_support_directions=wx_support_directions,
        wx_support_values=wx_support_values,
    )


def _build_library_with_both_types() -> SkillLibrary:
    """Build a library with one FULL_SIMPLEX and one MDN_WX skill."""
    lib = SkillLibrary()

    # FULL_SIMPLEX CDS skill: safety-dominant
    entry_fs = _make_entry(
        skill_id="fs-cds",
        delta_r=1.0,
        delta_n=(0.5, 0.1),
    )
    lib._skills["fs-cds"] = entry_fs

    # MDN_WX CDS skill: fuel-dominant
    entry_wx = _make_entry(
        skill_id="wx-cds",
        delta_r=0.8,
        delta_n=(0.1, 0.6),
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 14,
        mdn_alpha=(3.0, 2.0),
        wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
        wx_support_values=(0.8, 0.4),
    )
    lib._skills["wx-cds"] = entry_wx

    return lib

def test_save_load_preserves_mdn_audit_metadata(tmp_path):
    """library.json should round-trip all MDN audit fields."""
    lib = SkillLibrary()
    entry = _make_entry(
        skill_id="mdn-skill",
        weight_region_type=MDN_WX,
        certification_context=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                0.5, 0.3, 0.2, 1.0, 0.0, 0.1),
        mdn_alpha=(3.0, 2.0),
        wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
        wx_support_values=(0.8, 0.4),
    )
    lib._skills["mdn-skill"] = entry

    save_path = str(tmp_path / "library.json")
    lib.save(save_path)

    lib2 = SkillLibrary()
    lib2.load(save_path)

    loaded = lib2.get_skill("mdn-skill")
    assert loaded is not None
    assert loaded.weight_region_type == MDN_WX
    assert loaded.certification_context is not None
    assert len(loaded.certification_context) == 14
    assert loaded.mdn_alpha == (3.0, 2.0)
    assert loaded.wx_support_directions == ((1.0, 0.0), (0.0, 1.0))
    assert loaded.wx_support_values == (0.8, 0.4)

def test_old_entries_load_as_full_simplex(tmp_path):
    """library.json without weight_region_type should default to FULL_SIMPLEX."""
    cert = _make_cert("old-skill")
    old_data = {
        "version": 1,
        "skill_count": 1,
        "skills": {
            "old-skill": {
                "skill_id": "old-skill",
                "gate_type": "CDS",
                "certificate": cert.to_dict(),
                "executions": 0,
                "success_rate": 0.0,
                "avg_payoff": 0.0,
            }
        },
    }
    save_path = tmp_path / "library.json"
    save_path.write_text(json.dumps(old_data))

    lib = SkillLibrary()
    lib.load(str(save_path))

    loaded = lib.get_skill("old-skill")
    assert loaded is not None
    assert loaded.weight_region_type == FULL_SIMPLEX
    assert loaded.certification_context is None
    assert loaded.mdn_alpha is None
    assert loaded.wx_support_directions is None
    assert loaded.wx_support_values is None

def test_select_by_mdn_empty_library_returns_none():
    """select_by_mdn on empty library returns None."""
    mdn = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)
    lib = SkillLibrary()
    selector = SkillSelector(library=lib, mdn=mdn)
    assert selector.select_by_mdn(np.zeros(8)) is None


def test_select_by_mdn_without_mdn_raises():
    """select_by_mdn should raise ValueError without an MDN."""
    lib = SkillLibrary()
    selector = SkillSelector(library=lib)
    with pytest.raises(ValueError, match="MotiveDecompositionNetwork"):
        selector.select_by_mdn(np.zeros(8))


def test_full_simplex_skills_always_admissible():
    """FULL_SIMPLEX skills should pass admissibility for any valid weight."""
    lib = SkillLibrary()
    entry = _make_entry(skill_id="fs", weight_region_type=FULL_SIMPLEX)
    lib._skills["fs"] = entry

    for w in [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.3, 0.7]]:
        result = lib.query_admissible(
            current_weight=np.array(w),
            support_directions=np.eye(2),
            support_values=np.array([0.8, 0.4]),
        )
        assert len(result) == 1
        assert result[0].skill_id == "fs"

def test_invalid_current_weights_rejected():
    """query_admissible should reject invalid simplex weights."""
    lib = SkillLibrary()
    entry = _make_entry(skill_id="fs")
    lib._skills["fs"] = entry

    with pytest.raises(ValueError, match="simplex"):
        lib.query_admissible(current_weight=np.array([0.3, 0.3]))  # sums to 0.6

    with pytest.raises(ValueError, match="simplex"):
        lib.query_admissible(current_weight=np.array([-0.5, 1.5]))  # negative

def test_mdn_wx_admitted_when_support_passes():
    """MDN_WX CDS skill with large delta_r should pass admissibility."""
    lib = SkillLibrary()
    entry = _make_entry(
        skill_id="wx-pass",
        delta_r=1.0,
        delta_n=(-0.2, 0.1),
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 14,
        mdn_alpha=(3.0, 2.0),
        wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
        wx_support_values=(0.8, 0.4),
    )
    lib._skills["wx-pass"] = entry

    # support_values=[0.8, 0.4] → vertices [0.8, 0.2] and [0.6, 0.4]
    # -delta_n = [0.2, -0.1]
    # [0.8,0.2]·[0.2,-0.1] = 0.14
    # [0.6,0.4]·[0.2,-0.1] = 0.08
    # h_Wx = 0.14.  delta_r=1.0 >= 0.14 → PASS
    result = lib.query_admissible(
        current_weight=np.array([0.6, 0.4]),
        support_directions=np.eye(2),
        support_values=np.array([0.8, 0.4]),
    )
    assert len(result) == 1
    assert result[0].skill_id == "wx-pass"

def test_mdn_wx_filtered_when_support_fails():
    """MDN_WX CDS skill with small delta_r should fail admissibility."""
    lib = SkillLibrary()
    entry = _make_entry(
        skill_id="wx-fail",
        delta_r=0.05,       # too small
        delta_n=(-0.2, 0.1),
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 14,
        mdn_alpha=(3.0, 2.0),
        wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
        wx_support_values=(0.8, 0.4),
    )
    lib._skills["wx-fail"] = entry

    # h_Wx(-delta_n) = 0.14.  delta_r=0.05 < 0.14 → FAIL
    result = lib.query_admissible(
        current_weight=np.array([0.6, 0.4]),
        support_directions=np.eye(2),
        support_values=np.array([0.8, 0.4]),
    )
    assert len(result) == 0

def test_safety_heavy_alpha_selects_safety_skill():
    """With safety-biased weights, the safety-dominant skill should win."""
    lib = _build_library_with_both_types()
    mdn = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)

    # Mock forward_inference to return exact, controlled values.
    # α = [10, 1] → w = [10/11, 1/11] ≈ [0.909, 0.091]
    # support = [0.8, 0.4] → admits the MDN_WX skill
    mock_alpha = torch.tensor([10.0, 1.0])
    mock_support = torch.tensor([0.8, 0.4])

    with patch.object(mdn, "forward_inference", return_value=(mock_alpha, mock_support)):
        selector = SkillSelector(library=lib, mdn=mdn, seed=42)
        chosen = selector.select_by_mdn(np.zeros(8))

    # w = [10/11, 1/11]
    # fs-cds: score = 1.0 + (10/11)*0.5 + (1/11)*0.1 ≈ 1.464
    # wx-cds: score = 0.8 + (10/11)*0.1 + (1/11)*0.6 ≈ 0.945
    # Safety-heavy weights favor fs-cds (higher delta_n[0])
    assert chosen == "fs-cds"

def test_fuel_heavy_alpha_selects_fuel_skill():
    """With fuel-biased weights, the fuel-dominant skill should win."""
    lib = _build_library_with_both_types()
    mdn = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)

    # Mock forward_inference to return exact, controlled values.
    # α = [1, 10] → w = [1/11, 10/11] ≈ [0.091, 0.909]
    # support = [0.8, 0.4] → admits the MDN_WX skill
    mock_alpha = torch.tensor([1.0, 10.0])
    mock_support = torch.tensor([0.8, 0.4])

    with patch.object(mdn, "forward_inference", return_value=(mock_alpha, mock_support)):
        selector = SkillSelector(library=lib, mdn=mdn, seed=42)
        chosen = selector.select_by_mdn(np.zeros(8))

    # w = [1/11, 10/11]
    # fs-cds: score = 1.0 + (1/11)*0.5 + (10/11)*0.1 ≈ 1.136
    # wx-cds: score = 0.8 + (1/11)*0.1 + (10/11)*0.6 ≈ 1.355
    # Fuel-heavy weights favor wx-cds (higher delta_n[1])
    assert chosen == "wx-cds"

def test_selection_score_matches_manual_formula():
    """Verify the selection score equals delta_r + w^T · delta_n."""
    lib = SkillLibrary()
    entry = _make_entry(
        skill_id="verify",
        delta_r=0.5,
        delta_n=(0.3, 0.2),
    )
    lib._skills["verify"] = entry

    w = np.array([0.6, 0.4])
    expected_score = 0.5 + 0.6 * 0.3 + 0.4 * 0.2  # = 0.76

    # Omitting support_directions/values is safe here because the only
    # skill is FULL_SIMPLEX, which never touches the support path.
    admissible = lib.query_admissible(current_weight=w)
    assert len(admissible) == 1

    delta_n = np.array(admissible[0].delta_n)
    actual_score = admissible[0].delta_r + float(np.dot(w, delta_n))
    assert abs(actual_score - expected_score) < 1e-10

def test_wx_worst_case_matches_spec_example():
    """Verify the h_Wx(-delta_n) example from the spec."""
    delta_n = np.array([-0.2, 0.1])
    support_directions = np.eye(2)
    support_values = np.array([0.8, 0.4])

    h_wx = _compute_wx_worst_case(delta_n, support_directions, support_values)

    # From spec:
    # vertices: [0.8, 0.2] and [0.6, 0.4]
    # -delta_n = [0.2, -0.1]
    # [0.8,0.2]·[0.2,-0.1] = 0.16 - 0.02 = 0.14
    # [0.6,0.4]·[0.2,-0.1] = 0.12 - 0.04 = 0.08
    # h_Wx = max(0.14, 0.08) = 0.14
    assert abs(h_wx - 0.14) < 1e-10

def test_mdn_wx_without_support_data_raises():
    """query_admissible should raise when MDN_WX skill has no current support data."""
    lib = SkillLibrary()
    entry = _make_entry(
        skill_id="wx-no-support",
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 14,
        mdn_alpha=(3.0, 2.0),
        wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
        wx_support_values=(0.8, 0.4),
    )
    lib._skills["wx-no-support"] = entry

    with pytest.raises(ValueError, match="support_directions"):
        lib.query_admissible(current_weight=np.array([0.5, 0.5]))

def test_pds_mdn_wx_uses_epsilon():
    """PDS MDN_WX: delta_r >= h_Wx(-delta_n) - epsilon."""
    lib = SkillLibrary()
    entry = _make_entry(
        skill_id="wx-pds",
        gate_type="PDS",
        delta_r=0.1,
        delta_n=(-0.2, 0.1),
        epsilon=0.1,
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 14,
        mdn_alpha=(3.0, 2.0),
        wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
        wx_support_values=(0.8, 0.4),
    )
    lib._skills["wx-pds"] = entry

    # h_Wx = 0.14. For PDS: delta_r >= h_Wx - epsilon → 0.1 >= 0.14 - 0.1 = 0.04 ✓
    result = lib.query_admissible(
        current_weight=np.array([0.6, 0.4]),
        support_directions=np.eye(2),
        support_values=np.array([0.8, 0.4]),
    )
    assert len(result) == 1


def test_pds_mdn_wx_fails_without_epsilon():
    """PDS MDN_WX without enough epsilon should fail."""
    lib = SkillLibrary()
    entry = _make_entry(
        skill_id="wx-pds-fail",
        gate_type="PDS",
        delta_r=0.01,
        delta_n=(-0.2, 0.1),
        epsilon=0.1,
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 14,
        mdn_alpha=(3.0, 2.0),
        wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
        wx_support_values=(0.8, 0.4),
    )
    lib._skills["wx-pds-fail"] = entry

    # h_Wx = 0.14. 0.01 >= 0.14 - 0.1 = 0.04? NO → FAIL
    result = lib.query_admissible(
        current_weight=np.array([0.6, 0.4]),
        support_directions=np.eye(2),
        support_values=np.array([0.8, 0.4]),
    )
    assert len(result) == 0

def test_mdn_wx_without_audit_fields_raises_at_construction():
    """MDN_WX SkillEntry must require all four audit fields."""
    # All four missing — error should list every one.
    with pytest.raises(ValueError) as exc_info:
        _make_entry(
            skill_id="bad-wx",
            weight_region_type=MDN_WX,
        )
    msg = str(exc_info.value)
    for field in ["certification_context", "mdn_alpha",
                  "wx_support_directions", "wx_support_values"]:
        assert field in msg, f"Expected '{field}' in error message, got: {msg}"

    with pytest.raises(ValueError) as exc_info:
        _make_entry(
            skill_id="bad-wx-2",
            weight_region_type=MDN_WX,
            mdn_alpha=(3.0, 2.0),
        )
    msg = str(exc_info.value)
    assert "mdn_alpha" not in msg, "mdn_alpha was provided, should not be listed"
    for field in ["certification_context", "wx_support_directions", "wx_support_values"]:
        assert field in msg, f"Expected '{field}' in error message, got: {msg}"

def test_wx_worst_case_rejects_non_2d():
    """Vertex reconstruction should reject M != 2."""
    with pytest.raises(ValueError, match="M=2"):
        _compute_wx_worst_case(
            delta_n=np.array([0.1, 0.2, 0.3]),
            support_directions=np.eye(3),
            support_values=np.array([0.5, 0.5, 0.5]),
        )

def test_wx_worst_case_rejects_non_basis_directions():
    """Vertex reconstruction should reject non-standard-basis directions."""
    with pytest.raises(ValueError, match="standard basis"):
        _compute_wx_worst_case(
            delta_n=np.array([-0.2, 0.1]),
            support_directions=np.array([[0.7, 0.3], [0.3, 0.7]]),
            support_values=np.array([0.8, 0.4]),
        )

def test_wx_worst_case_rejects_wrong_shape_directions():
    """1D or mismatched-shape directions should get a shape error, not broadcast."""
    with pytest.raises(ValueError, match="shape"):
        _compute_wx_worst_case(
            delta_n=np.array([-0.2, 0.1]),
            support_directions=np.array([1.0, 0.0]),
            support_values=np.array([0.8, 0.4]),
        )

    with pytest.raises(ValueError, match="shape"):
        _compute_wx_worst_case(
            delta_n=np.array([-0.2, 0.1]),
            support_directions=np.array([[1.0, 0.0]]),
            support_values=np.array([0.8, 0.4]),
        )

def test_tie_break_selects_lexicographically_smaller_id():
    """When scores are equal, the selector should pick the lexicographically smaller skill_id."""
    lib = SkillLibrary()

    for sid in ["skill_b", "skill_a"]:
        entry = _make_entry(
            skill_id=sid,
            delta_r=0.5,
            delta_n=(0.3, 0.2),
        )
        lib._skills[sid] = entry

    mdn = MotiveDecompositionNetwork(input_dim=8, num_objectives=2)

    mock_alpha = torch.tensor([5.0, 5.0])
    mock_support = torch.tensor([0.9, 0.9])

    with patch.object(mdn, "forward_inference", return_value=(mock_alpha, mock_support)):
        selector = SkillSelector(library=lib, mdn=mdn, seed=42)
        chosen = selector.select_by_mdn(np.zeros(8))

    assert chosen == "skill_a"
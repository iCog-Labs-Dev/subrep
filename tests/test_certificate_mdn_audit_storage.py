from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from certification.certificate_schema import (
    Certificate,
    is_mdn_certificate,
    validate_mdn_certificate,
)


def _bridge():
    pytest.importorskip("hyperon")
    from certification.metta_bridge import (
        atom_to_cert,
        cert_to_atom,
        metta_to_python_value,
        parse_atom,
        python_to_metta_value,
        serialize_atom,
    )

    return {
        "atom_to_cert": atom_to_cert,
        "cert_to_atom": cert_to_atom,
        "metta_to_python_value": metta_to_python_value,
        "parse_atom": parse_atom,
        "python_to_metta_value": python_to_metta_value,
        "serialize_atom": serialize_atom,
    }


def _store_class():
    pytest.importorskip("hyperon")
    from certification.metta_storage import CertificateStore

    return CertificateStore


def _base_kwargs(**overrides):
    data = {
        "skill_id": "audit_skill",
        "gate_type": "CDS",
        "delta_r": 1.0,
        "delta_n": (0.4, 0.2),
        "admission_margin": 1.2,
        "epsilon": 0.0,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": 7,
        "gamma": 0.99,
        "baseline_id": "idle_policy",
        "environment": "MO-LunarLander-v3",
        "episode_length": 100,
        "version": "test",
    }
    data.update(overrides)
    return data


def _full_simplex_cert(**overrides) -> Certificate:
    return Certificate(**_base_kwargs(**overrides))


def _mdn_cert(**overrides) -> Certificate:
    data = _base_kwargs(
        weight_region_type="MDN_WX",
        certification_context=(-0.1, 0.2, 0.3),
        mdn_alpha=(1.5, 2.0),
        wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
        wx_support_values=(1.0, 1.0),
    )
    data.update(overrides)
    return Certificate(**data)


def test_full_simplex_certificate_defaults_to_none_mdn_fields():
    cert = _full_simplex_cert()

    assert cert.weight_region_type == "FULL_SIMPLEX"
    assert cert.certification_context is None
    assert cert.mdn_alpha is None
    assert cert.wx_support_directions is None
    assert cert.wx_support_values is None
    assert is_mdn_certificate(cert) is False
    validate_mdn_certificate(cert)


def test_full_simplex_rejects_non_none_mdn_fields():
    with pytest.raises(ValueError):
        _full_simplex_cert(mdn_alpha=(1.0, 1.0))


def test_mdn_certificate_preserves_audit_fields():
    cert = _mdn_cert()

    assert is_mdn_certificate(cert) is True
    assert cert.certification_context == (-0.1, 0.2, 0.3)
    assert cert.mdn_alpha == (1.5, 2.0)
    assert cert.wx_support_directions == ((1.0, 0.0), (0.0, 1.0))
    assert cert.wx_support_values == (1.0, 1.0)
    validate_mdn_certificate(cert)


def test_mdn_certificate_rejects_missing_audit_fields():
    with pytest.raises(ValueError):
        _mdn_cert(certification_context=None)
    with pytest.raises(ValueError):
        _mdn_cert(mdn_alpha=None)
    with pytest.raises(ValueError):
        _mdn_cert(wx_support_directions=None)
    with pytest.raises(ValueError):
        _mdn_cert(wx_support_values=None)


def test_validate_mdn_certificate_catches_missing_mdn_fields():
    incomplete = SimpleNamespace(
        weight_region_type="MDN_WX",
        certification_context=None,
        mdn_alpha=(1.0, 2.0),
        wx_support_directions=((1.0, 0.0),),
        wx_support_values=(1.0,),
    )

    with pytest.raises(ValueError, match="certification_context"):
        validate_mdn_certificate(incomplete)  # type: ignore[arg-type]


def test_invalid_alpha_and_support_shapes_fail():
    with pytest.raises(ValueError):
        _mdn_cert(mdn_alpha=(0.0, 1.0))
    with pytest.raises(ValueError):
        _mdn_cert(mdn_alpha=(float("inf"), 1.0))
    with pytest.raises(ValueError):
        _mdn_cert(wx_support_directions=((1.0, 0.0), (0.0,)))
    with pytest.raises(ValueError):
        _mdn_cert(wx_support_values=(1.0,))
    with pytest.raises(ValueError):
        _mdn_cert(wx_support_values=(-0.1, 1.0))


def test_metta_roundtrip_preserves_mdn_audit_metadata():
    bridge = _bridge()
    cert = _mdn_cert(skill_id="roundtrip_skill")
    text = bridge["serialize_atom"](bridge["cert_to_atom"](cert))
    restored = bridge["atom_to_cert"](bridge["parse_atom"](text))

    assert restored.to_dict() == cert.to_dict()
    assert restored.certification_context == cert.certification_context
    assert restored.mdn_alpha == cert.mdn_alpha
    assert restored.wx_support_directions == cert.wx_support_directions
    assert restored.wx_support_values == cert.wx_support_values


def test_old_certificate_loads_with_full_simplex_defaults():
    bridge = _bridge()
    old_text = (
        '(Certificate '
        '(skill_id "old_skill") '
        '(gate_type "CDS") '
        '(delta_r 1.0) '
        '(delta_n (vec 0.4 0.2)) '
        '(admission_margin 1.2) '
        '(epsilon 0.0) '
        '(timestamp "2026-06-07T12:00:00") '
        '(seed 7) '
        '(gamma 0.99) '
        '(baseline_id "idle_policy") '
        '(environment "MO-LunarLander-v3") '
        '(episode_length 100) '
        '(version "test"))'
    )

    cert = bridge["atom_to_cert"](bridge["parse_atom"](old_text))

    assert cert.skill_id == "old_skill"
    assert cert.weight_region_type == "FULL_SIMPLEX"
    assert cert.certification_context is None
    assert cert.mdn_alpha is None
    assert cert.wx_support_directions is None
    assert cert.wx_support_values is None


def test_nil_none_and_vector_conversion_helpers():
    bridge = _bridge()
    nil_atom = bridge["python_to_metta_value"](None)
    assert str(nil_atom) == "Nil"
    assert bridge["metta_to_python_value"](nil_atom) is None

    vector_atom = bridge["python_to_metta_value"]([0.8, 0.2])
    assert bridge["metta_to_python_value"](vector_atom) == [0.8, 0.2]

    matrix_atom = bridge["python_to_metta_value"]([[1, 0], [0, 1]])
    assert bridge["metta_to_python_value"](matrix_atom) == [[1.0, 0.0], [0.0, 1.0]]


def test_runtime_certificates_contain_python_none_not_raw_nil(tmp_path: Path):
    CertificateStore = _store_class()
    cert = _full_simplex_cert(skill_id="nil_runtime")
    path = tmp_path / "certs.metta"
    store = CertificateStore()
    store.add(cert)
    store.save_to_file(path)

    loaded = CertificateStore()
    loaded.load_from_file(path)
    restored = loaded.get_certificate("nil_runtime")

    assert restored is not None
    assert restored.certification_context is None
    assert restored.mdn_alpha is None
    assert restored.wx_support_directions is None
    assert restored.wx_support_values is None


def test_full_simplex_runtime_safety_branch_does_not_crash():
    cert = _full_simplex_cert(skill_id="safe_full_simplex")

    if cert.weight_region_type == "MDN_WX":
        _ = cert.mdn_alpha[0]  # pragma: no cove
    else:
        assert cert.mdn_alpha is None
        assert cert.certification_context is None


def test_is_mdn_certificate_identifies_certificate_types():
    assert is_mdn_certificate(_full_simplex_cert()) is False
    assert is_mdn_certificate(_mdn_cert()) is True


# Arbitrary-M certificate storage (SASP downstream generalization).
#
# The MDN support head is now feasible at any objective count, so the
# certificate schema and the MeTTa bridge/store must carry M > 2 vectors
# end to end. Quarter Plan Objective 3, KR 1 and KR 2.

def _m5_mdn_cert(**overrides) -> Certificate:
    """A five-objective MDN_WX certificate with a feasible W_x region."""
    identity = tuple(
        tuple(1.0 if row == col else 0.0 for col in range(5)) for row in range(5)
    )
    data = _base_kwargs(
        skill_id="audit_skill_m5",
        delta_n=(0.4, 0.2, 0.5, 0.1, 0.3),
        weight_region_type="MDN_WX",
        certification_context=(-0.1, 0.2, 0.3),
        mdn_alpha=(1.5, 2.0, 1.0, 2.5, 1.25),
        wx_support_directions=identity,
        wx_support_values=(0.5, 0.5, 0.5, 0.5, 0.5),
    )
    data.update(overrides)
    return Certificate(**data)


def test_m5_mdn_certificate_construction_and_validation():
    """The schema must accept a five-objective MDN_WX certificate."""
    cert = _m5_mdn_cert()

    assert is_mdn_certificate(cert) is True
    assert len(cert.delta_n) == 5
    assert len(cert.mdn_alpha) == 5
    assert len(cert.wx_support_values) == 5
    assert len(cert.wx_support_directions) == 5
    validate_mdn_certificate(cert)


def test_m5_certificate_rejects_inconsistent_vector_lengths():
    """Every MDN_WX vector must agree with M inferred from delta_n."""
    with pytest.raises(ValueError, match="mdn_alpha must have length"):
        _m5_mdn_cert(mdn_alpha=(1.5, 2.0))

    with pytest.raises(ValueError, match="wx_support_values must have length"):
        _m5_mdn_cert(
            wx_support_values=(0.5, 0.5, 0.5),
            wx_support_directions=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        )


def test_m5_certificate_rejects_empty_wx_region():
    """sum(s) < 1 means no weighting exists; it must not be certifiable."""
    with pytest.raises(ValueError, match="sum"):
        _m5_mdn_cert(wx_support_values=(0.1, 0.1, 0.1, 0.1, 0.1))


def test_metta_roundtrip_preserves_m5_audit_metadata():
    """The MeTTa bridge must serialize length-5 vectors without loss."""
    bridge = _bridge()
    cert = _m5_mdn_cert(skill_id="roundtrip_skill_m5")

    text = bridge["serialize_atom"](bridge["cert_to_atom"](cert))
    restored = bridge["atom_to_cert"](bridge["parse_atom"](text))

    assert restored.to_dict() == cert.to_dict()
    assert restored.delta_n == cert.delta_n
    assert restored.mdn_alpha == cert.mdn_alpha
    assert restored.wx_support_directions == cert.wx_support_directions
    assert restored.wx_support_values == cert.wx_support_values


def test_metta_store_roundtrip_m5_certificate(tmp_path: Path):
    """Store -> save -> load -> retrieve must be exact at M = 5."""
    CertificateStore = _store_class()
    cert = _m5_mdn_cert(skill_id="stored_m5")
    path = tmp_path / "certs_m5.metta"

    store = CertificateStore()
    assert store.add(cert) is True
    store.save_to_file(path)

    loaded = CertificateStore()
    loaded.load_from_file(path)
    restored = loaded.get_certificate("stored_m5")

    assert restored is not None
    assert restored.to_dict() == cert.to_dict()
    # Exact float equality, not approximate: audit records must not drift.
    assert restored.wx_support_values == (0.5, 0.5, 0.5, 0.5, 0.5)
    assert restored.delta_n == (0.4, 0.2, 0.5, 0.1, 0.3)


def test_metta_store_query_by_weights_accepts_m5(tmp_path: Path):
    """query_by_weights was pinned to length-2; it must accept any M >= 2."""
    CertificateStore = _store_class()
    store = CertificateStore()
    store.add(_m5_mdn_cert(skill_id="queried_m5"))

    results = store.query_by_weights([0.2, 0.2, 0.2, 0.2, 0.2])

    assert [c.skill_id for c in results] == ["queried_m5"]

    # Still rejects genuinely invalid simplex vectors at any length.
    with pytest.raises(ValueError):
        store.query_by_weights([0.5, 0.5, 0.5, 0.5, 0.5])  # sums to 2.5


def test_metta_store_backward_compatible_with_m2_certificates(tmp_path: Path):
    """Existing two-objective artifacts must still round-trip unchanged.

    Guards the generalization: widening every length check from exactly-2 to
    M >= 2 must not alter M = 2 behavior in any way.
    """
    CertificateStore = _store_class()
    full_simplex = _full_simplex_cert(skill_id="legacy_m2_fs")
    mdn_wx = _mdn_cert(skill_id="legacy_m2_wx")
    path = tmp_path / "certs_legacy_m2.metta"

    store = CertificateStore()
    store.add(full_simplex)
    store.add(mdn_wx)
    store.save_to_file(path)

    loaded = CertificateStore()
    loaded.load_from_file(path)

    assert loaded.count() == 2
    assert loaded.get_certificate("legacy_m2_fs").to_dict() == full_simplex.to_dict()
    assert loaded.get_certificate("legacy_m2_wx").to_dict() == mdn_wx.to_dict()
    assert [c.skill_id for c in loaded.query_by_weights([0.5, 0.5])] == [
        "legacy_m2_fs",
        "legacy_m2_wx",
    ]


def test_legacy_m2_metta_text_still_parses():
    """A hand-written pre-change M=2 expression must still load.

    Hyperon-free equivalents live in tests/test_skill_library.py; this one
    exercises the real bridge in the WSL environment where hyperon is present.
    """
    bridge = _bridge()
    legacy_text = (
        '(Certificate '
        '(skill_id "legacy_wx") '
        '(gate_type "PDS") '
        '(delta_r 0.5) '
        '(delta_n (vec 0.8 -0.6)) '
        '(admission_margin 0.0) '
        '(epsilon 0.1) '
        '(timestamp "2026-06-07T12:00:00") '
        '(seed 7) '
        '(gamma 0.99) '
        '(baseline_id "idle_policy") '
        '(environment "MO-LunarLander-v3") '
        '(episode_length 100) '
        '(version "test") '
        '(weight_region_type "MDN_WX") '
        '(certification_context (vec 0.0 0.0)) '
        '(mdn_alpha (vec 3.0 2.0)) '
        '(wx_support_directions (list (vec 1.0 0.0) (vec 0.0 1.0))) '
        '(wx_support_values (vec 0.8 0.4)))'
    )

    cert = bridge["atom_to_cert"](bridge["parse_atom"](legacy_text))

    assert cert.skill_id == "legacy_wx"
    assert cert.weight_region_type == "MDN_WX"
    assert cert.delta_n == (0.8, -0.6)
    assert cert.wx_support_values == (0.8, 0.4)
    assert cert.wx_support_directions == ((1.0, 0.0), (0.0, 1.0))

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

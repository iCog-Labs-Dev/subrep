"""
Certificate Storage Validation Tests
Verifies schema validation, MeTTA roundtrip conversion, and storage/query APIs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

pytest.importorskip("hyperon")

from certification.certificate_schema import Certificate
from certification.metta_bridge import atom_to_cert, cert_to_atom, parse_atom, serialize_atom
from certification.metta_storage import CertificateStore


def _sample_certificate(
    *,
    skill_id: str = "landing_skill_01",
    gate_type: str = "CDS",
    delta_r: float = 0.5,
    delta_n: tuple[float, float] = (0.2, -0.1),
    admission_margin: float = 0.4,
    epsilon: float = 0.0,
) -> Certificate:
    """Build a valid certificate fixture with overridable key fields."""
    return Certificate(
        skill_id=skill_id,
        gate_type=gate_type,
        delta_r=delta_r,
        delta_n=delta_n,
        admission_margin=admission_margin,
        epsilon=epsilon,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        seed=42,
        gamma=0.99,
        baseline_id="idle_policy",
        environment="MO-LunarLander-v0",
        episode_length=120,
        version="subrep-q1-v0.1",
    )


def test_certificate_gate_type_normalization():
    """Gate labels should normalize to canonical uppercase."""
    cert = _sample_certificate(gate_type="cds")
    assert cert.gate_type == "CDS"


def test_certificate_invalid_gate_type_fails():
    with pytest.raises(ValueError):
        _sample_certificate(gate_type="INVALID")


def test_certificate_invalid_timestamp_fails():
    with pytest.raises(ValueError):
        Certificate(
            skill_id="x",
            gate_type="CDS",
            delta_r=0.1,
            delta_n=(0.1, 0.1),
            admission_margin=0.1,
            epsilon=0.0,
            timestamp="not-a-timestamp",
            seed=1,
            gamma=0.9,
            baseline_id="idle_policy",
            environment="MO-LunarLander-v0",
            episode_length=10,
            version="v1",
        )


def test_certificate_cds_nonzero_epsilon_fails():
    with pytest.raises(ValueError):
        _sample_certificate(gate_type="CDS", epsilon=0.1)


def test_certificate_wrong_delta_n_length_fails():
    with pytest.raises(ValueError):
        _sample_certificate(delta_n=(0.1, 0.2, 0.3))  # type: ignore[arg-type]


def test_certificate_negative_admission_margin_fails():
    with pytest.raises(ValueError):
        _sample_certificate(admission_margin=-0.01)


def test_bridge_roundtrip_preserves_data_exactly():
    """Direct atom conversion must preserve all schema fields."""
    cert = _sample_certificate(gate_type="PDS", epsilon=0.1, skill_id="bridge_case")
    atom = cert_to_atom(cert)
    restored = atom_to_cert(atom)
    assert restored.to_dict() == cert.to_dict()


def test_bridge_string_roundtrip_preserves_data_exactly():
    """Text serialization/parsing path must preserve all schema fields."""
    cert = _sample_certificate(gate_type="PDS", epsilon=0.1, skill_id="string_case")
    atom = cert_to_atom(cert)
    text = serialize_atom(atom)
    parsed = parse_atom(text)
    restored = atom_to_cert(parsed)
    assert restored.to_dict() == cert.to_dict()


def test_bridge_parse_malformed_expression_fails():
    malformed = '(Certificate (skill_id "x") (gate_type "CDS"'
    with pytest.raises(ValueError):
        parse_atom(malformed)


def test_store_add_get_count_and_contains():
    """Basic store lifecycle: empty -> add -> lookup -> count."""
    store = CertificateStore()
    cert = _sample_certificate(skill_id="skill_a")

    assert store.count() == 0
    assert store.contains("skill_a") is False
    assert store.add(cert) is True
    assert store.count() == 1
    assert store.contains("skill_a") is True
    assert store.get_certificate("skill_a") == cert


def test_store_duplicate_skill_rejected():
    store = CertificateStore()
    cert = _sample_certificate(skill_id="duplicate_case")
    assert store.add(cert) is True
    assert store.add(cert) is False
    assert store.count() == 1


def test_store_get_and_remove_missing_skill():
    store = CertificateStore()
    assert store.get_certificate("missing") is None
    assert store.remove_skill("missing") is False


def test_query_by_gate_type_filters_and_normalizes():
    store = CertificateStore()
    cds = _sample_certificate(skill_id="cds_1", gate_type="CDS", epsilon=0.0)
    pds = _sample_certificate(
        skill_id="pds_1",
        gate_type="PDS",
        epsilon=0.1,
        delta_r=0.2,
        delta_n=(0.1, -0.4),
        admission_margin=0.05,
    )
    store.add(cds)
    store.add(pds)

    cds_results = store.query_by_gate_type("cds")
    pds_results = store.query_by_gate_type("PDS")

    assert [c.skill_id for c in cds_results] == ["cds_1"]
    assert [c.skill_id for c in pds_results] == ["pds_1"]


def test_query_by_gate_type_invalid_value_raises():
    store = CertificateStore()
    with pytest.raises(ValueError):
        store.query_by_gate_type("bad")


def test_query_by_weights_admissibility_logic():
    """Weight query should apply CDS global pass and PDS inequality check."""
    store = CertificateStore()
    cds = _sample_certificate(skill_id="cds_global", gate_type="CDS", epsilon=0.0)
    pds_pass = _sample_certificate(
        skill_id="pds_pass",
        gate_type="PDS",
        epsilon=0.1,
        delta_r=0.2,
        delta_n=(0.0, -0.25),
        admission_margin=0.02,
    )
    pds_fail = _sample_certificate(
        skill_id="pds_fail",
        gate_type="PDS",
        epsilon=0.1,
        delta_r=0.0,
        delta_n=(0.0, -0.3),
        admission_margin=0.01,
    )
    store.add(cds)
    store.add(pds_pass)
    store.add(pds_fail)

    results = store.query_by_weights([0.5, 0.5])
    ids = {c.skill_id for c in results}
    assert "cds_global" in ids
    assert "pds_pass" in ids
    assert "pds_fail" not in ids

    results_alt = store.query_by_weights([1.0, 0.0])
    ids_alt = {c.skill_id for c in results_alt}
    assert "cds_global" in ids_alt
    assert "pds_pass" in ids_alt
    assert "pds_fail" in ids_alt


def test_query_by_weights_invalid_values_raise():
    """Non-simplex/invalid vectors must be rejected with clear errors."""
    store = CertificateStore()
    store.add(_sample_certificate(skill_id="x"))

    with pytest.raises(ValueError):
        store.query_by_weights([0.2, 0.2])  # sum != 1
    with pytest.raises(ValueError):
        store.query_by_weights([-0.1, 1.1])  # negative component
    with pytest.raises(ValueError):
        store.query_by_weights([0.5, np.inf])  # non-finite
    with pytest.raises(ValueError):
        store.query_by_weights([0.5, 0.3, 0.2])  # wrong shape


def test_query_by_weights_raises_even_when_store_empty():
    store = CertificateStore()
    with pytest.raises(ValueError):
        store.query_by_weights([0.4, 0.4])


def test_save_and_load_file_roundtrip_replaces_store():
    """Persisted certificates should reload exactly and replace existing state."""
    store = CertificateStore()
    cert1 = _sample_certificate(skill_id="persist_1", gate_type="CDS", epsilon=0.0)
    cert2 = _sample_certificate(skill_id="persist_2", gate_type="PDS", epsilon=0.1)
    store.add(cert1)
    store.add(cert2)

    file_path = Path.cwd() / f"certificates_{uuid4().hex}.metta"
    try:
        store.save_to_file(file_path)
        assert file_path.exists()

        loaded = CertificateStore()
        loaded.load_from_file(file_path)
        assert loaded.count() == 2
        assert {c.skill_id for c in loaded.load_all()} == {"persist_1", "persist_2"}

        # Replace semantics on load: current store content gets replaced.
        loaded.remove_skill("persist_1")
        assert loaded.count() == 1
        loaded.load_from_file(file_path)
        assert loaded.count() == 2
    finally:
        if file_path.exists():
            file_path.unlink()


def test_load_from_file_duplicate_skill_id_fails():
    """Duplicate IDs in file input should fail before mutating store."""
    cert = _sample_certificate(skill_id="dup_in_file")
    line = serialize_atom(cert_to_atom(cert))
    file_path = Path.cwd() / f"dup_{uuid4().hex}.metta"
    try:
        file_path.write_text(f"{line}\n{line}\n", encoding="utf-8")
        store = CertificateStore()
        with pytest.raises(ValueError):
            store.load_from_file(file_path)
    finally:
        if file_path.exists():
            file_path.unlink()

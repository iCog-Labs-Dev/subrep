"""
Skill Library Validation Tests.

Verifies the full lifecycle of the SubRep Skill Library:
- SkillEntry creation and serialization
- Adding/removing certified skills
- Query by gate type and weight vectors
- JSON save/load roundtrips
- Random selection with reproducibility
- Integration with certification gates

"""

import numpy as np
import pytest

from library.skill_metadata import Certificate, SkillEntry
from library.skill_library import SkillLibrary
from library.skill_selector import SkillSelector


def make_dummy_policy(action: int = 0):
    """Create a simple deterministic policy that always returns `action`."""
    return lambda obs: action

def make_cds_certificate(cert_id: str = "cert-cds-001"):
    """
    Create a CDS certificate for a universally beneficial skill.

    Δr=0.5, Δn=[0.3, 0.2] → margin = 0.5 + min(0.3, 0.2) = 0.7
    Passes CDS because Δr + min(Δn) ≥ 0.
    """
    return Certificate(
        certificate_id=cert_id,
        gate_type="CDS",
        delta_r=0.5,
        delta_n=[0.3, 0.2],
        admission_margin=0.7,
        epsilon=0.0,
    )

def make_pds_certificate(cert_id: str = "cert-pds-001"):
    """
    Create a PDS certificate for a trade-off skill.

    Δr=0.5, Δn=[0.8, -0.6], ε=0.1
    margin = 0.5 + (-0.6) + 0.1 = 0.0  (exactly at boundary)
    Passes PDS because Δr + min(Δn) ≥ -ε.
    """
    return Certificate(
        certificate_id=cert_id,
        gate_type="PDS",
        delta_r=0.5,
        delta_n=[0.8, -0.6],
        admission_margin=0.0,
        epsilon=0.1,
    )

def build_populated_library():
    """
    Build a library with 3 skills for query tests:
      - skill-1: CDS, universally beneficial
      - skill-2: PDS, trade-off within ε
      - skill-3: CDS, another universally beneficial
    """
    lib = SkillLibrary()
    lib.add_skill("skill-1", make_cds_certificate("cert-cds-001"), make_dummy_policy(0))
    lib.add_skill("skill-2", make_pds_certificate("cert-pds-001"), make_dummy_policy(1))
    lib.add_skill("skill-3", make_cds_certificate("cert-cds-002"), make_dummy_policy(2))
    return lib

def test_skill_entry_creation():
    """SkillEntry should store all fields correctly."""
    cert = make_cds_certificate()
    entry = SkillEntry(
        skill_id="skill-1",
        gate_type="CDS",
        certificate=cert,
        policy=make_dummy_policy(),
    )

    assert entry.skill_id == "skill-1"
    assert entry.gate_type == "CDS"
    assert entry.delta_r == 0.5           
    assert entry.delta_n == [0.3, 0.2]    
    assert entry.admission_margin == 0.7  
    assert entry.executions == 0          
    assert entry.policy is not None


def test_skill_entry_rejects_invalid_gate_type():
    """SkillEntry should reject gate types other than CDS/PDS."""
    cert = Certificate(
        certificate_id="bad",
        gate_type="CDS",   
        delta_r=0.5,
        delta_n=[0.1],
    )

    with pytest.raises(ValueError, match="gate_type"):
        # entry gate_type is wrong
        SkillEntry(skill_id="x", gate_type="INVALID", certificate=cert)


def test_skill_entry_rejects_mismatched_gate_type():
    """SkillEntry gate_type must match its certificate's gate_type."""
    cert = make_cds_certificate()  # gate_type = "CDS"

    with pytest.raises(ValueError, match="does not match"):
        SkillEntry(skill_id="x", gate_type="PDS", certificate=cert)


def test_skill_entry_to_dict_roundtrip():
    """to_dict() → from_dict() should preserve all serializable fields."""
    cert = make_pds_certificate()
    entry = SkillEntry(
        skill_id="skill-rt",
        gate_type="PDS",
        certificate=cert,
        policy=make_dummy_policy(),
        executions=5,
        success_rate=0.8,
        avg_payoff=1.23,
    )

    d = entry.to_dict()
    restored = SkillEntry.from_dict(d)

    assert restored.skill_id == entry.skill_id
    assert restored.gate_type == entry.gate_type
    assert restored.delta_r == entry.delta_r
    assert restored.delta_n == entry.delta_n
    assert restored.admission_margin == entry.admission_margin
    assert restored.epsilon == entry.epsilon
    assert restored.executions == entry.executions
    assert np.isclose(restored.success_rate, entry.success_rate)
    assert np.isclose(restored.avg_payoff, entry.avg_payoff)
    # Policy is NOT preserved across serialization — this is by design
    assert restored.policy is None

# Certificate Tests
def test_certificate_rejects_invalid_gate_type():
    """Certificate should reject gate types other than CDS/PDS."""
    with pytest.raises(ValueError, match="gate_type"):
        Certificate(certificate_id="x", gate_type="XYZ", delta_r=0.0)


def test_certificate_to_dict_roundtrip():
    """Certificate serialization should preserve all fields."""
    cert = make_pds_certificate()
    d = cert.to_dict()
    restored = Certificate.from_dict(d)

    assert restored.certificate_id == cert.certificate_id
    assert restored.gate_type == cert.gate_type
    assert np.isclose(restored.delta_r, cert.delta_r)
    assert restored.delta_n == cert.delta_n
    assert np.isclose(restored.epsilon, cert.epsilon)


# Add / Get / Remove Tests
def test_add_certified_skill_succeeds():
    """Adding a skill with a valid certificate should succeed."""
    lib = SkillLibrary()
    cert = make_cds_certificate()
    result = lib.add_skill("skill-1", cert, make_dummy_policy())

    assert result is True
    assert lib.count() == 1


def test_add_multiple_skills():
    """Library should store multiple distinct skills."""
    lib = build_populated_library()
    assert lib.count() == 3


def test_add_overwrites_existing_skill():
    """Adding a skill with an existing ID should overwrite it."""
    lib = SkillLibrary()
    cert1 = make_cds_certificate("cert-001")
    cert2 = make_pds_certificate("cert-002")

    lib.add_skill("skill-1", cert1, make_dummy_policy(0))
    lib.add_skill("skill-1", cert2, make_dummy_policy(1))

    assert lib.count() == 1
    # Should have the second certificate's data
    assert lib.get_skill("skill-1").gate_type == "PDS"


def test_add_noncertified_skill_rejected():
    """With cert_store, skills with unknown certificates should be rejected."""

    # Minimal mock: only needs has_certificate()
    class MockCertStore:
        def __init__(self, known_ids):
            self._known = set(known_ids)

        def has_certificate(self, cert_id):
            return cert_id in self._known

    store = MockCertStore(known_ids={"cert-known"})
    lib = SkillLibrary(cert_store=store)

    # Known certificate → accepted
    known_cert = make_cds_certificate(cert_id="cert-known")
    assert lib.add_skill("good", known_cert, make_dummy_policy()) is True

    # Unknown certificate → rejected
    unknown_cert = make_cds_certificate(cert_id="cert-unknown")
    assert lib.add_skill("bad", unknown_cert, make_dummy_policy()) is False
    assert lib.count() == 1  # only the good one


def test_get_skill_returns_correct_entry():
    """get_skill should return the matching SkillEntry."""
    lib = build_populated_library()
    entry = lib.get_skill("skill-2")

    assert entry is not None
    assert entry.skill_id == "skill-2"
    assert entry.gate_type == "PDS"


def test_get_nonexistent_skill_returns_none():
    """get_skill should return None for unknown IDs."""
    lib = SkillLibrary()
    assert lib.get_skill("nonexistent") is None


def test_remove_skill_succeeds():
    """Removing an existing skill should return True and reduce count."""
    lib = build_populated_library()
    assert lib.count() == 3

    result = lib.remove_skill("skill-2")

    assert result is True
    assert lib.count() == 2
    assert lib.get_skill("skill-2") is None


def test_remove_nonexistent_skill_returns_false():
    """Removing a skill that doesn't exist should return False."""
    lib = SkillLibrary()
    assert lib.remove_skill("ghost") is False


# Query Tests
def test_query_by_gate_type_cds():
    """query_by_gate_type('CDS') should return only CDS skills."""
    lib = build_populated_library()  # 2 CDS, 1 PDS
    cds_skills = lib.query_by_gate_type("CDS")

    assert len(cds_skills) == 2
    assert all(s.gate_type == "CDS" for s in cds_skills)


def test_query_by_gate_type_pds():
    """query_by_gate_type('PDS') should return only PDS skills."""
    lib = build_populated_library()
    pds_skills = lib.query_by_gate_type("PDS")

    assert len(pds_skills) == 1
    assert pds_skills[0].gate_type == "PDS"
    assert pds_skills[0].skill_id == "skill-2"


def test_query_by_gate_type_empty_result():
    """query_by_gate_type should return empty list if no match."""
    lib = SkillLibrary()
    lib.add_skill("s1", make_cds_certificate(), make_dummy_policy())

    assert lib.query_by_gate_type("PDS") == []


def test_query_by_weights_cds_always_admissible():
    """CDS skills should be admissible under ANY valid weight vector."""
    lib = SkillLibrary()
    lib.add_skill("cds-skill", make_cds_certificate(), make_dummy_policy())

    # Try multiple weight vectors — CDS should always pass
    for w in [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.3, 0.7]]:
        result = lib.query_by_weights(w)
        assert len(result) == 1, f"CDS skill should pass for weights {w}"


def test_query_by_weights_pds_depends_on_weights():
    """PDS skills should only pass for weight vectors where Δr + w^T·Δn ≥ -ε."""
    lib = SkillLibrary()
    # PDS cert: Δr=0.5, Δn=[0.8, -0.6], ε=0.1
    lib.add_skill("pds-skill", make_pds_certificate(), make_dummy_policy())

    # w=[0.5, 0.5]: score = 0.5 + 0.5*0.8 + 0.5*(-0.6) = 0.5 + 0.1 = 0.6 ≥ -0.1 → Pass
    assert len(lib.query_by_weights([0.5, 0.5])) == 1

    # w=[1.0, 0.0]: score = 0.5 + 1.0*0.8 + 0.0*(-0.6) = 1.3 ≥ -0.1 → Pass
    assert len(lib.query_by_weights([1.0, 0.0])) == 1

    # w=[0.0, 1.0]: score = 0.5 + 0.0*0.8 + 1.0*(-0.6) = -0.1 ≥ -0.1 → Pass (boundary)
    assert len(lib.query_by_weights([0.0, 1.0])) == 1


def test_query_by_weights_pds_rejected():
    """PDS skill should be rejected when score < -ε for given weights."""
    lib = SkillLibrary()
    # Manually craft a PDS cert that fails for w=[0.0, 1.0]:
    # Δr=0.3, Δn=[0.8, -0.6], ε=0.1
    # score = 0.3 + 0.0*0.8 + 1.0*(-0.6) = -0.3 < -0.1 → Fail
    bad_cert = Certificate(
        certificate_id="cert-hard-pds",
        gate_type="PDS",
        delta_r=0.3,
        delta_n=[0.8, -0.6],
        admission_margin=0.0,
        epsilon=0.1,
    )
    lib.add_skill("hard-pds", bad_cert, make_dummy_policy())

    # w=[0.0, 1.0] → should reject
    assert len(lib.query_by_weights([0.0, 1.0])) == 0

    # w=[1.0, 0.0] → score = 0.3 + 0.8 = 1.1 ≥ -0.1 → should pass
    assert len(lib.query_by_weights([1.0, 0.0])) == 1


def test_query_by_weights_mixed_library():
    """With mixed CDS/PDS, only CDS + qualifying PDS should pass."""
    lib = build_populated_library()  # 2 CDS + 1 PDS (Δr=0.5, Δn=[0.8,-0.6], ε=0.1)

    # w=[0.5, 0.5]: PDS score = 0.5 + 0.1 = 0.6 ≥ -0.1 → all 3 pass
    assert len(lib.query_by_weights([0.5, 0.5])) == 3


def test_query_by_weights_rejects_invalid_weights():
    """Invalid weight vectors should raise ValueError."""
    lib = build_populated_library()

    with pytest.raises(ValueError):
        lib.query_by_weights([0.3, 0.3])   # doesn't sum to 1

    with pytest.raises(ValueError):
        lib.query_by_weights([1.5, -0.5])  # negative component


# Persistence Tests
def test_save_load_roundtrip(tmp_path):
    """Library should survive a JSON save → load cycle."""
    save_file = str(tmp_path / "test_library.json")

    # Save
    lib = build_populated_library()
    lib.save(save_file)

    # Load into a fresh library
    lib2 = SkillLibrary()
    lib2.load(save_file)

    assert lib2.count() == 3
    assert lib2.get_skill("skill-1") is not None
    assert lib2.get_skill("skill-2") is not None
    assert lib2.get_skill("skill-3") is not None


def test_loaded_skills_have_no_policy(tmp_path):
    """After load, all skills should have policy=None."""
    save_file = str(tmp_path / "test_library.json")

    lib = build_populated_library()
    lib.save(save_file)

    lib2 = SkillLibrary()
    lib2.load(save_file)

    for entry in lib2.get_admitted_skills():
        assert entry.policy is None


def test_loaded_skills_preserve_data(tmp_path):
    """Loaded skills should preserve gate_type, delta_r, delta_n, etc."""
    save_file = str(tmp_path / "test_library.json")

    lib = SkillLibrary()
    cert = make_pds_certificate()
    lib.add_skill("s1", cert, make_dummy_policy())
    lib.save(save_file)

    lib2 = SkillLibrary()
    lib2.load(save_file)
    entry = lib2.get_skill("s1")

    assert entry.gate_type == "PDS"
    assert np.isclose(entry.delta_r, 0.5)
    assert entry.delta_n == [0.8, -0.6]
    assert np.isclose(entry.epsilon, 0.1)


def test_register_policy_after_load(tmp_path):
    """register_policy should re-attach a callable after loading."""
    save_file = str(tmp_path / "test_library.json")

    lib = build_populated_library()
    lib.save(save_file)

    lib2 = SkillLibrary()
    lib2.load(save_file)

    # Before registration
    assert lib2.get_skill("skill-1").policy is None

    # Register a policy
    new_policy = make_dummy_policy(99)
    assert lib2.register_policy("skill-1", new_policy) is True

    # After registration
    assert lib2.get_skill("skill-1").policy is not None
    assert lib2.get_skill("skill-1").policy(None) == 99


def test_register_policy_nonexistent_skill():
    """register_policy should return False for unknown skill IDs."""
    lib = SkillLibrary()
    assert lib.register_policy("ghost", make_dummy_policy()) is False


# Selector Tests
def test_select_random_returns_valid_skill():
    """select_random should return a skill_id that exists in the library."""
    lib = build_populated_library()
    selector = SkillSelector(library=lib, seed=42)
    obs = np.zeros(8)

    skill_id = selector.select_random(obs)

    assert skill_id is not None
    assert lib.get_skill(skill_id) is not None


def test_select_random_reproducible_with_seed():
    """Same seed should produce the same selection sequence."""
    lib = build_populated_library()
    obs = np.zeros(8)

    # Two selectors with the same seed
    sel_a = SkillSelector(library=lib, seed=123)
    sel_b = SkillSelector(library=lib, seed=123)

    # Generate a sequence of selections from each
    seq_a = [sel_a.select_random(obs) for _ in range(10)]
    seq_b = [sel_b.select_random(obs) for _ in range(10)]

    assert seq_a == seq_b


def test_select_random_different_seeds_differ():
    """Different seeds should (very likely) produce different sequences."""
    lib = build_populated_library()
    obs = np.zeros(8)

    sel_a = SkillSelector(library=lib, seed=1)
    sel_b = SkillSelector(library=lib, seed=999)

    seq_a = [sel_a.select_random(obs) for _ in range(20)]
    seq_b = [sel_b.select_random(obs) for _ in range(20)]

    # With 3 skills and 20 draws, identical sequences are astronomically unlikely
    assert seq_a != seq_b


def test_select_random_empty_library_returns_none():
    """select_random on an empty library should return None, not crash."""
    lib = SkillLibrary()
    selector = SkillSelector(library=lib, seed=42)
    obs = np.zeros(8)

    result = selector.select_random(obs)
    assert result is None


def test_select_by_payoff_raises_not_implemented():
    """select_by_payoff should raise NotImplementedError (Stage 5 stub)."""
    lib = build_populated_library()
    selector = SkillSelector(library=lib)
    obs = np.zeros(8)

    with pytest.raises(NotImplementedError, match="Stage 5"):
        selector.select_by_payoff(obs)


def test_select_by_mdn_raises_not_implemented():
    """select_by_mdn should raise NotImplementedError (Stage 6 stub)."""
    lib = build_populated_library()
    selector = SkillSelector(library=lib)
    obs = np.zeros(8)

    with pytest.raises(NotImplementedError, match="Stage 6"):
        selector.select_by_mdn(obs)

# Integration: Certification → Library → Selection
def test_certification_to_library_flow():
    """
    End-to-end: create certificates, add to library, query, select.

    This mirrors the SubRep loop: Certify → Store → Select → Execute.
    """
    from certification.cds_test import CDSGate
    from certification.pds_test import PDSGate

    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=0.1)

    # Skill A: universally beneficial → passes CDS
    skill_a_r, skill_a_n = 0.8, np.array([0.5, 0.3])
    assert cds_gate.admit(skill_a_r, skill_a_n) is True
    cert_a = Certificate(
        certificate_id="cert-int-a",
        gate_type="CDS",
        delta_r=skill_a_r,
        delta_n=skill_a_n.tolist(),
        admission_margin=cds_gate.get_admission_margin(skill_a_r, skill_a_n),
        epsilon=0.0,
    )

    # Skill B: trade-off → fails CDS, passes PDS
    skill_b_r, skill_b_n = 0.5, np.array([0.8, -0.6])
    assert cds_gate.admit(skill_b_r, skill_b_n) is False
    assert pds_gate.admit(skill_b_r, skill_b_n) is True
    cert_b = Certificate(
        certificate_id="cert-int-b",
        gate_type="PDS",
        delta_r=skill_b_r,
        delta_n=skill_b_n.tolist(),
        admission_margin=pds_gate.get_admission_margin(skill_b_r, skill_b_n),
        epsilon=pds_gate.get_epsilon(),
    )

    # Build library
    lib = SkillLibrary()
    lib.add_skill("hover", cert_a, lambda obs: 0)
    lib.add_skill("land-fast", cert_b, lambda obs: 2)
    assert lib.count() == 2

    # Query: both should appear for equal weights
    admissible = lib.query_by_weights([0.5, 0.5])
    assert len(admissible) == 2

    # Query: only CDS for gate type
    cds_only = lib.query_by_gate_type("CDS")
    assert len(cds_only) == 1
    assert cds_only[0].skill_id == "hover"

    # Select: should return one of the two skill IDs
    selector = SkillSelector(library=lib, seed=42)
    chosen = selector.select_random(np.zeros(8))
    assert chosen in {"hover", "land-fast"}

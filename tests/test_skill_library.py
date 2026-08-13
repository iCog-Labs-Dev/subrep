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
import warnings

import numpy as np
import pytest
from datetime import datetime

from certification.certificate_schema import Certificate
from library.skill_metadata import FULL_SIMPLEX, MDN_WX, SkillEntry
from library.skill_library import SkillLibrary
from library.skill_selector import SkillSelector

def make_dummy_policy(action: int = 0):
    """Create a simple deterministic policy that always returns `action`."""
    return lambda obs: action

def make_cds_certificate(skill_id: str = "cert-cds-001"):
    """
    Create a CDS certificate for a universally beneficial skill.

    Δr=0.5, Δn=[0.3, 0.2] → margin = 0.5 + min(0.3, 0.2) = 0.7
    Passes CDS because Δr + min(Δn) ≥ 0.
    """
    return Certificate(
        skill_id=skill_id,
        gate_type="CDS",
        delta_r=0.5,
        delta_n=(0.3, 0.2),
        admission_margin=0.7,
        epsilon=0.0,
        timestamp=datetime.now().isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="baseline-noop",
        environment="MO-LunarLander-v2",
        episode_length=200,
        version="0.1.0",
    )

def make_pds_certificate(skill_id: str = "cert-pds-001"):
    """
    Create a PDS certificate for a trade-off skill.

    Δr=0.5, Δn=(0.8, -0.6), ε=0.1
    margin = 0.5 + (-0.6) + 0.1 = 0.0  (exactly at boundary)
    Passes PDS because Δr + min(Δn) ≥ -ε.
    """
    return Certificate(
        skill_id=skill_id,
        gate_type="PDS",
        delta_r=0.5,
        delta_n=(0.8, -0.6),
        admission_margin=0.0,
        epsilon=0.1,
        timestamp=datetime.now().isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="baseline-noop",
        environment="MO-LunarLander-v2",
        episode_length=200,
        version="0.1.0",
    )

def build_populated_library():
    """
    Build a library with 3 skills for query tests:
      - cert-cds-001: CDS, universally beneficial
      - cert-pds-001: PDS, trade-off within ε
      - cert-cds-002: CDS, another universally beneficial
    """
    lib = SkillLibrary()
    
    cert1 = make_cds_certificate("cert-cds-001")
    lib.add_skill(cert1.skill_id, cert1, make_dummy_policy(0))
    
    cert2 = make_pds_certificate("cert-pds-001")
    lib.add_skill(cert2.skill_id, cert2, make_dummy_policy(1))
    
    cert3 = make_cds_certificate("cert-cds-002")
    lib.add_skill(cert3.skill_id, cert3, make_dummy_policy(2))
    
    return lib

def test_skill_entry_creation():
    """SkillEntry should store all fields correctly."""
    cert = make_cds_certificate("skill-1")
    entry = SkillEntry(
        skill_id="skill-1",
        gate_type="CDS",
        certificate=cert,
        policy=make_dummy_policy(),
    )

    assert entry.skill_id == "skill-1"
    assert entry.gate_type == "CDS"
    assert entry.delta_r == 0.5
    assert entry.delta_n == (0.3, 0.2)
    assert entry.admission_margin == 0.7
    assert entry.executions == 0
    assert entry.policy is not None


def test_skill_entry_rejects_invalid_gate_type():
    """SkillEntry should reject gate types other than CDS/PDS."""
    cert = make_cds_certificate()

    with pytest.raises(ValueError, match="gate_type"):
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
        Certificate(
            skill_id="x",
            gate_type="XYZ",
            delta_r=0.0,
            delta_n=(0.0, 0.0),
            admission_margin=0.0,
            epsilon=0.0,
            timestamp=datetime.now().isoformat(),
            seed=42,
            gamma=0.99,
            baseline_id="baseline-noop",
            environment="MO-LunarLander-v2",
            episode_length=200,
            version="0.1.0",
        )


def test_certificate_to_dict_roundtrip():
    """Certificate serialization should preserve all fields."""
    cert = make_pds_certificate()
    d = cert.to_dict()
    restored = Certificate.from_dict(d)

    assert restored.skill_id == cert.skill_id
    assert restored.gate_type == cert.gate_type
    assert np.isclose(restored.delta_r, cert.delta_r)
    assert restored.delta_n == cert.delta_n
    assert np.isclose(restored.epsilon, cert.epsilon)
    # Verify audit fields survive the roundtrip
    assert restored.seed == cert.seed
    assert restored.gamma == cert.gamma
    assert restored.baseline_id == cert.baseline_id
    assert restored.environment == cert.environment


# Add / Get / Remove Tests
def test_add_certified_skill_succeeds():
    """Adding a skill with a valid certificate should succeed."""
    lib = SkillLibrary()
    cert = make_cds_certificate("skill-1")
    result = lib.add_skill(cert.skill_id, cert, make_dummy_policy())

    assert result is True
    assert lib.count() == 1


def test_add_multiple_skills():
    """Library should store multiple distinct skills."""
    lib = build_populated_library()
    assert lib.count() == 3


def test_add_overwrites_existing_skill():
    """Adding a skill with an existing ID should overwrite it."""
    lib = SkillLibrary()
    cert1 = make_cds_certificate("skill-1")
    cert2 = make_pds_certificate("skill-1")

    lib.add_skill(cert1.skill_id, cert1, make_dummy_policy(0))
    lib.add_skill(cert2.skill_id, cert2, make_dummy_policy(1))

    assert lib.count() == 1
    # Should have the second certificate's data
    assert lib.get_skill("skill-1").gate_type == "PDS"


def test_add_noncertified_skill_rejected():
    """With cert_store, skills with unknown certificates should be rejected."""

    # Minimal mock: only needs contains()
    class MockCertStore:
        def __init__(self, known_ids):
            self._known = set(known_ids)

        def contains(self, skill_id):
            return skill_id in self._known

    store = MockCertStore(known_ids={"cert-known"})
    lib = SkillLibrary(cert_store=store)

    # Known certificate → accepted
    known_cert = make_cds_certificate(skill_id="cert-known")
    assert lib.add_skill(known_cert.skill_id, known_cert, make_dummy_policy()) is True

    # Unknown certificate (ID mismatch) → rejected
    unknown_cert = make_cds_certificate(skill_id="cert-unknown")
    assert lib.add_skill("cert-known", unknown_cert, make_dummy_policy()) is False
    assert lib.count() == 1  # only the first one


def test_get_skill_returns_correct_entry():
    """get_skill should return the matching SkillEntry."""
    lib = build_populated_library()
    entry = lib.get_skill("cert-pds-001")

    assert entry is not None
    assert entry.skill_id == "cert-pds-001"
    assert entry.gate_type == "PDS"


def test_get_nonexistent_skill_returns_none():
    """get_skill should return None for unknown IDs."""
    lib = SkillLibrary()
    assert lib.get_skill("nonexistent") is None


def test_remove_skill_succeeds():
    """Removing an existing skill should return True and reduce count."""
    lib = build_populated_library()
    assert lib.count() == 3

    result = lib.remove_skill("cert-pds-001")

    assert result is True
    assert lib.count() == 2
    assert lib.get_skill("cert-pds-001") is None


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
    assert pds_skills[0].skill_id == "cert-pds-001"


def test_query_by_gate_type_empty_result():
    """query_by_gate_type should return empty list if no match."""
    lib = SkillLibrary()
    cert = make_cds_certificate("s1")
    lib.add_skill(cert.skill_id, cert, make_dummy_policy())

    assert lib.query_by_gate_type("PDS") == []


def test_query_by_weights_cds_always_admissible():
    """CDS skills should be admissible under ANY valid weight vector."""
    lib = SkillLibrary()
    cert = make_cds_certificate("cds-skill")
    lib.add_skill(cert.skill_id, cert, make_dummy_policy())

    # Try multiple weight vectors — CDS should always pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for w in [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0], [0.3, 0.7]]:
            result = lib.query_by_weights(w)
            assert len(result) == 1, f"CDS skill should pass for weights {w}"


def test_query_by_weights_pds_depends_on_weights():
    """PDS skills should only pass for weight vectors where Δr + w^T·Δn ≥ -ε."""
    lib = SkillLibrary()
    # PDS cert: Δr=0.5, Δn=(0.8, -0.6), ε=0.1
    cert = make_pds_certificate("pds-skill")
    lib.add_skill(cert.skill_id, cert, make_dummy_policy())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # w=[0.5, 0.5]: score = 0.5 + 0.5*0.8 + 0.5*(-0.6) = 0.6 ≥ -0.1 → Pass
        assert len(lib.query_by_weights([0.5, 0.5])) == 1

        # w=[1.0, 0.0]: score = 0.5 + 1.0*0.8 = 1.3 ≥ -0.1 → Pass
        assert len(lib.query_by_weights([1.0, 0.0])) == 1

        # w=[0.0, 1.0]: score = 0.5 + (-0.6) = -0.1 ≥ -0.1 → Pass (boundary)
        assert len(lib.query_by_weights([0.0, 1.0])) == 1


def test_query_by_weights_pds_rejected():
    """PDS skill should be rejected when score < -ε for given weights."""
    lib = SkillLibrary()
    # Manually craft a PDS cert that fails for w=[0.0, 1.0]:
    # Δr=0.3, Δn=[0.8, -0.6], ε=0.1
    # score = 0.3 + 0.0*0.8 + 1.0*(-0.6) = -0.3 < -0.1 → Fail
    bad_cert = Certificate(
        skill_id="cert-hard-pds",
        gate_type="PDS",
        delta_r=0.3,
        delta_n=(0.8, -0.6),
        admission_margin=0.0,
        epsilon=0.1,
        timestamp=datetime.now().isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="baseline-noop",
        environment="MO-LunarLander-v2",
        episode_length=200,
        version="0.1.0",
    )
    # We must bypass add_skill here because our new Chain of Safety correctly
    # rejects this mathematically failing certificate before it enters the library!
    # We inject it directly into _skills to test query_by_weights in isolation.
    lib._skills["hard-pds"] = SkillEntry(
        skill_id="hard-pds",
        gate_type="PDS",
        certificate=bad_cert,
        policy=make_dummy_policy()
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # w=[0.0, 1.0] → should reject
        assert len(lib.query_by_weights([0.0, 1.0])) == 0

        # w=[1.0, 0.0] → score = 0.3 + 0.8 = 1.1 ≥ -0.1 → should pass
        assert len(lib.query_by_weights([1.0, 0.0])) == 1


def test_query_by_weights_mixed_library():
    """With mixed CDS/PDS, only CDS + qualifying PDS should pass."""
    lib = build_populated_library()  # 2 CDS + 1 PDS (Δr=0.5, Δn=[0.8,-0.6], ε=0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # w=[0.5, 0.5]: PDS score = 0.5 + 0.1 = 0.6 ≥ -0.1 → all 3 pass
        assert len(lib.query_by_weights([0.5, 0.5])) == 3


def test_query_by_weights_rejects_invalid_weights():
    """Invalid weight vectors should raise ValueError."""
    lib = build_populated_library()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
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
    assert lib2.get_skill("cert-cds-001") is not None
    assert lib2.get_skill("cert-pds-001") is not None
    assert lib2.get_skill("cert-cds-002") is not None


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
    cert = make_pds_certificate("s1")
    lib.add_skill(cert.skill_id, cert, make_dummy_policy())
    lib.save(save_file)

    lib2 = SkillLibrary()
    lib2.load(save_file)
    entry = lib2.get_skill("s1")

    assert entry.gate_type == "PDS"
    assert np.isclose(entry.delta_r, 0.5)
    assert entry.delta_n == (0.8, -0.6)
    assert np.isclose(entry.epsilon, 0.1)
    # Verify audit fields survive the roundtrip
    assert entry.certificate.seed == 42
    assert entry.certificate.environment == "MO-LunarLander-v2"


def test_register_policy_after_load(tmp_path):
    """register_policy should re-attach a callable after loading."""
    save_file = str(tmp_path / "test_library.json")

    lib = build_populated_library()
    lib.save(save_file)

    lib2 = SkillLibrary()
    lib2.load(save_file)

    # Before registration
    assert lib2.get_skill("cert-cds-001").policy is None

    # Register a policy
    new_policy = make_dummy_policy(99)
    assert lib2.register_policy("cert-cds-001", new_policy) is True

    # After registration
    assert lib2.get_skill("cert-cds-001").policy is not None
    assert lib2.get_skill("cert-cds-001").policy(None) == 99


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


def test_select_by_mdn_requires_mdn():
    """select_by_mdn should raise ValueError without an MDN."""
    lib = build_populated_library()
    selector = SkillSelector(library=lib)
    obs = np.zeros(8)

    with pytest.raises(ValueError, match="MotiveDecompositionNetwork"):
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
        skill_id="cert-int-a",
        gate_type="CDS",
        delta_r=skill_a_r,
        delta_n=tuple(skill_a_n.tolist()),
        admission_margin=cds_gate.get_admission_margin(skill_a_r, skill_a_n),
        epsilon=0.0,
        timestamp=datetime.now().isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="baseline-noop",
        environment="MO-LunarLander-v2",
        episode_length=200,
        version="0.1.0",
    )

    # Skill B: trade-off → fails CDS, passes PDS
    skill_b_r, skill_b_n = 0.5, np.array([0.8, -0.6])
    assert cds_gate.admit(skill_b_r, skill_b_n) is False
    assert pds_gate.admit(skill_b_r, skill_b_n) is True
    cert_b = Certificate(
        skill_id="cert-int-b",
        gate_type="PDS",
        delta_r=skill_b_r,
        delta_n=tuple(skill_b_n.tolist()),
        admission_margin=pds_gate.get_admission_margin(skill_b_r, skill_b_n),
        epsilon=pds_gate.get_epsilon(),
        timestamp=datetime.now().isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="baseline-noop",
        environment="MO-LunarLander-v2",
        episode_length=200,
        version="0.1.0",
    )

    # Build library
    lib = SkillLibrary()
    lib.add_skill(cert_a.skill_id, cert_a, lambda obs: 0)
    lib.add_skill(cert_b.skill_id, cert_b, lambda obs: 2)
    assert lib.count() == 2

    # Query: both should appear for equal weights
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        admissible = lib.query_by_weights([0.5, 0.5])
        assert len(admissible) == 2

    # Query: only CDS for gate type
    cds_only = lib.query_by_gate_type("CDS")
    assert len(cds_only) == 1
    assert cds_only[0].skill_id == "cert-int-a"

    # Select: should return one of the two skill IDs
    selector = SkillSelector(library=lib, seed=42)
    chosen = selector.select_random(np.zeros(8))
    assert chosen in {"cert-int-a", "cert-int-b"}


# Exact Greedy W_x Solver Tests (SASP downstream generalization)
def test_greedy_solver_matches_legacy_vertices_at_M2():
    """The greedy LP must reproduce the retired two-vertex enumeration exactly.

    This is the backward-compatibility guarantee: at M = 2 the region
    { w in simplex : w_i <= s_i } is the segment whose endpoints are precisely
    the vertices [s0, 1-s0] and [1-s1, s1] the old code built by hand, so the
    two computations must agree to floating-point exactness.
    """
    from library.skill_library import _compute_wx_worst_case

    def legacy_two_vertex(delta_n: np.ndarray, support: np.ndarray) -> float:
        neg_delta_n = -np.asarray(delta_n, dtype=np.float64)
        vertices = np.array(
            [[support[0], 1.0 - support[0]], [1.0 - support[1], support[1]]],
            dtype=np.float64,
        )
        return float(np.max(vertices @ neg_delta_n))

    rng = np.random.default_rng(42)
    checked = 0
    max_difference = 0.0
    while checked < 20000:
        support = rng.random(2)
        if support.sum() < 1.0:
            continue  # infeasible region: rejected before evaluation
        delta_n = rng.normal(size=2) * 2.0

        greedy = _compute_wx_worst_case(delta_n, np.eye(2), support)
        legacy = legacy_two_vertex(delta_n, support)

        max_difference = max(max_difference, abs(greedy - legacy))
        checked += 1

    assert max_difference < 1e-12, f"max difference {max_difference:.3e}"


def test_full_simplex_greedy_matches_compute_worst_case_motive():
    """The full simplex is the s = 1-vector special case of the same greedy.

    Confirms the two code paths agree instead of silently diverging, so the
    full-simplex branch needs no separate worst-case implementation.
    """
    from utils.cone_utils import compute_worst_case_motive
    from utils.support_geometry import worst_case_over_support_region

    rng = np.random.default_rng(11)
    for num_objectives in (2, 3, 5, 10):
        for _ in range(50):
            delta_n = rng.normal(size=num_objectives) * 3.0

            greedy = worst_case_over_support_region(
                delta_n, np.ones(num_objectives)
            )
            reference = compute_worst_case_motive(delta_n)

            assert abs(greedy - reference) < 1e-12


def test_greedy_solver_is_permutation_symmetric():
    """Relabelling objectives must not change the worst-case value."""
    from library.skill_library import _compute_wx_worst_case

    rng = np.random.default_rng(5)
    for num_objectives in (3, 5, 8):
        support = rng.random(num_objectives) * 0.5 + 0.5  # sum >= 1 by construction
        delta_n = rng.normal(size=num_objectives)
        permutation = rng.permutation(num_objectives)

        original = _compute_wx_worst_case(delta_n, np.eye(num_objectives), support)
        permuted = _compute_wx_worst_case(
            delta_n[permutation], np.eye(num_objectives), support[permutation]
        )

        assert abs(original - permuted) < 1e-12


# Runtime Feasibility Telemetry Tests
def test_infeasible_support_counter_increments_and_excludes_wx():
    """Infeasible support must be counted, not silently logged and dropped.

    Invisibility was the worst property of the original bug: MDN_WX skills
    disappeared from selection with only a log line. The counter makes any
    recurrence loud.
    """
    from library.skill_metadata import MDN_WX

    lib = SkillLibrary()
    cert = make_cds_certificate("fs-skill")
    lib.add_skill(cert.skill_id, cert, make_dummy_policy())

    wx_cert = Certificate(
        skill_id="wx-skill",
        gate_type="CDS",
        delta_r=0.8,
        delta_n=(0.1, 0.6),
        admission_margin=0.9,
        epsilon=0.0,
        timestamp=datetime.now().isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="baseline-noop",
        environment="MO-LunarLander-v2",
        episode_length=200,
        version="0.1.0",
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 8,
        mdn_alpha=(3.0, 2.0),
        wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
        wx_support_values=(0.8, 0.4),
    )
    assert lib.add_skill(
        wx_cert.skill_id,
        wx_cert,
        make_dummy_policy(),
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 8,
        mdn_alpha=(3.0, 2.0),
        wx_support_directions=((1.0, 0.0), (0.0, 1.0)),
        wx_support_values=(0.8, 0.4),
    ) is True

    assert lib.infeasible_support_events == 0

    # sum(s) = 0.8 < 1 -> empty region, MDN_WX must be excluded and counted.
    admissible = lib.query_admissible(
        current_weight=np.array([0.5, 0.5]),
        support_directions=np.eye(2),
        support_values=np.array([0.4, 0.4]),
    )

    assert lib.infeasible_support_events == 1
    assert {entry.skill_id for entry in admissible} == {"fs-skill"}

    # Feasible support: no further events, MDN_WX becomes eligible again.
    lib.query_admissible(
        current_weight=np.array([0.5, 0.5]),
        support_directions=np.eye(2),
        support_values=np.array([0.8, 0.7]),
    )
    assert lib.infeasible_support_events == 1


def test_public_support_values_feasible_helper():
    """One shared definition of feasibility, exported for outside consumers."""
    from library import support_values_feasible

    assert support_values_feasible(np.array([0.8, 0.4])) is True
    assert support_values_feasible(np.array([0.4, 0.4])) is False   # sum < 1
    assert support_values_feasible(np.array([1.4, 0.4])) is False   # s_i > 1
    assert support_values_feasible(np.array([-0.1, 1.2])) is False  # s_i < 0
    assert support_values_feasible(np.array([np.nan, 1.0])) is False
    assert support_values_feasible(np.array([1.0])) is False        # M < 2
    assert support_values_feasible(np.array([[0.8, 0.4]])) is False  # not 1D
    # Generalizes past two objectives.
    assert support_values_feasible(np.array([0.4, 0.4, 0.4, 0.4, 0.4])) is True


# M > 2 Certification Path Tests
def _make_m5_wx_certificate(skill_id: str = "wx-m5") -> Certificate:
    """A five-objective MDN_WX certificate that passes CDS under its region.

    delta_n is all-positive, so the worst case over any W_x is positive and any
    non-negative delta_r admits. This exercises the greedy path end to end.
    """
    from library.skill_metadata import MDN_WX

    return Certificate(
        skill_id=skill_id,
        gate_type="CDS",
        delta_r=0.9,
        delta_n=(0.3, 0.4, 0.5, 0.2, 0.6),
        admission_margin=1.1,
        epsilon=0.0,
        timestamp=datetime.now().isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="baseline-noop",
        environment="MO-LunarLander-v2",
        episode_length=200,
        version="0.1.0",
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 8,
        mdn_alpha=(2.0, 2.0, 2.0, 2.0, 2.0),
        wx_support_directions=tuple(
            tuple(float(v) for v in row) for row in np.eye(5)
        ),
        wx_support_values=(0.5, 0.5, 0.5, 0.5, 0.5),
    )


def test_add_skill_accepts_m5_mdn_wx_certificate():
    """add_skill must certify an M=5 MDN_WX skill, not raise.

    Regression guard for the real gap: add_skill used to reconstruct a
    two-vertex WeightSet for every MDN_WX skill, so fixing only the network
    head would still have hard-failed here at M > 2.
    """
    from library.skill_metadata import MDN_WX

    lib = SkillLibrary()
    cert = _make_m5_wx_certificate()

    added = lib.add_skill(
        cert.skill_id,
        cert,
        make_dummy_policy(),
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 8,
        mdn_alpha=(2.0, 2.0, 2.0, 2.0, 2.0),
        wx_support_directions=tuple(
            tuple(float(v) for v in row) for row in np.eye(5)
        ),
        wx_support_values=(0.5, 0.5, 0.5, 0.5, 0.5),
    )

    assert added is True
    assert lib.count() == 1


def test_query_admissible_at_m5():
    """Runtime admissibility must work at M = 5 through the greedy solver."""
    from library.skill_metadata import MDN_WX

    lib = SkillLibrary()
    cert = _make_m5_wx_certificate()
    lib.add_skill(
        cert.skill_id,
        cert,
        make_dummy_policy(),
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 8,
        mdn_alpha=(2.0, 2.0, 2.0, 2.0, 2.0),
        wx_support_directions=tuple(
            tuple(float(v) for v in row) for row in np.eye(5)
        ),
        wx_support_values=(0.5, 0.5, 0.5, 0.5, 0.5),
    )

    admissible = lib.query_admissible(
        current_weight=np.full(5, 0.2),
        support_directions=np.eye(5),
        support_values=np.full(5, 0.5),
    )

    assert {entry.skill_id for entry in admissible} == {"wx-m5"}
    assert lib.infeasible_support_events == 0


def test_add_skill_rejects_m5_wx_certificate_that_fails_its_gate():
    """The chain of safety must still reject bad math at M > 2.

    Generalizing the solver must not weaken verification: delta_n has a
    strongly negative coordinate the region can put full allowed mass on, so a
    small delta_r cannot cover the worst case.
    """
    from library.skill_metadata import MDN_WX

    directions = tuple(tuple(float(v) for v in row) for row in np.eye(5))
    support = (0.5, 0.5, 0.5, 0.5, 0.5)
    delta_n = (0.3, 0.4, 0.5, 0.2, -2.0)

    cert = Certificate(
        skill_id="wx-m5-bad",
        gate_type="CDS",
        delta_r=0.05,
        delta_n=delta_n,
        admission_margin=0.0,
        epsilon=0.0,
        timestamp=datetime.now().isoformat(),
        seed=42,
        gamma=0.99,
        baseline_id="baseline-noop",
        environment="MO-LunarLander-v2",
        episode_length=200,
        version="0.1.0",
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 8,
        mdn_alpha=(2.0,) * 5,
        wx_support_directions=directions,
        wx_support_values=support,
    )

    lib = SkillLibrary()
    added = lib.add_skill(
        cert.skill_id,
        cert,
        make_dummy_policy(),
        weight_region_type=MDN_WX,
        certification_context=(0.0,) * 8,
        mdn_alpha=(2.0,) * 5,
        wx_support_directions=directions,
        wx_support_values=support,
    )

    assert added is False
    assert lib.count() == 0


def test_certificate_schema_rejects_empty_wx_region():
    """An empty W_x must be impossible to certify at construction."""
    from library.skill_metadata import MDN_WX

    with pytest.raises(ValueError, match="sum"):
        Certificate(
            skill_id="wx-empty",
            gate_type="CDS",
            delta_r=0.5,
            delta_n=(0.3, 0.2, 0.1),
            admission_margin=0.5,
            epsilon=0.0,
            timestamp=datetime.now().isoformat(),
            seed=42,
            gamma=0.99,
            baseline_id="baseline-noop",
            environment="MO-LunarLander-v2",
            episode_length=200,
            version="0.1.0",
            weight_region_type=MDN_WX,
            certification_context=(0.0,) * 8,
            mdn_alpha=(2.0, 2.0, 2.0),
            wx_support_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            wx_support_values=(0.2, 0.2, 0.2),  # sum 0.6 < 1
        )


def test_certificate_schema_rejects_length_mismatch():
    """All MDN_WX vectors must describe the same objective count."""
    from library.skill_metadata import MDN_WX

    with pytest.raises(ValueError, match="mdn_alpha must have length"):
        Certificate(
            skill_id="wx-mismatch",
            gate_type="CDS",
            delta_r=0.5,
            delta_n=(0.3, 0.2, 0.1),
            admission_margin=0.5,
            epsilon=0.0,
            timestamp=datetime.now().isoformat(),
            seed=42,
            gamma=0.99,
            baseline_id="baseline-noop",
            environment="MO-LunarLander-v2",
            episode_length=200,
            version="0.1.0",
            weight_region_type=MDN_WX,
            certification_context=(0.0,) * 8,
            mdn_alpha=(2.0, 2.0),  # length 2 against M = 3
            wx_support_directions=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            wx_support_values=(0.5, 0.5, 0.5),
        )


# Gate Support-Geometry Path Tests
def test_gates_support_values_path_matches_weight_set_path_at_m2():
    """The new support_values keyword must agree with the legacy vertex path."""
    from certification.cds_test import CDSGate
    from certification.pds_test import PDSGate
    from utils.weight_set_store import WeightSet

    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=0.1)

    rng = np.random.default_rng(3)
    checked = 0
    while checked < 2000:
        support = rng.random(2)
        if support.sum() < 1.0:
            continue
        delta_n = rng.normal(size=2)
        delta_r = float(rng.normal())

        weight_set = WeightSet()
        weight_set.add_vertex(
            np.array([support[0], 1.0 - support[0]], dtype=np.float32)
        )
        weight_set.add_vertex(
            np.array([1.0 - support[1], support[1]], dtype=np.float32)
        )

        for gate in (cds_gate, pds_gate):
            assert gate.admit(
                delta_r, delta_n, weight_set
            ) == gate.admit(delta_r, delta_n, None, support)
            assert abs(
                gate.get_admission_margin(delta_r, delta_n, weight_set)
                - gate.get_admission_margin(delta_r, delta_n, None, support)
            ) < 1e-6
        checked += 1


def test_greedy_solver_rejects_infeasible_support_values():
    """The solver must reject a region that has no maximum, not approximate one.

    An empty region (sum(s) < 1) cannot have full mass placed on it. Allocating
    only what the caps allow would return a value computed over partial mass --
    smaller than any true worst case -- which makes an admission gate strictly
    MORE permissive. Rejecting is the only safe behavior.
    """
    from utils.support_geometry import greedy_support_function

    coefficients = np.array([-0.5, 2.0])

    with pytest.raises(ValueError, match=r"sum\(s\) >= 1"):
        greedy_support_function(coefficients, np.array([0.4, 0.4]))   # sum 0.8

    with pytest.raises(ValueError, match="0 <= s_i <= 1"):
        greedy_support_function(coefficients, np.array([1.4, 0.2]))   # s_i > 1

    with pytest.raises(ValueError, match="0 <= s_i <= 1"):
        greedy_support_function(coefficients, np.array([-0.1, 1.2]))  # s_i < 0

    with pytest.raises(ValueError, match="finite"):
        greedy_support_function(coefficients, np.array([np.nan, 1.0]))

    # A feasible region is still evaluated normally.
    assert greedy_support_function(coefficients, np.array([1.0, 1.0])) == 2.0


def test_gates_reject_infeasible_support_values_instead_of_over_admitting():
    """Regression: an empty W_x must not make the gates more permissive.

    Before validation reached the greedy solver, delta_n=(0.5, -2.0) with the
    empty region s=(0.4, 0.4) produced h_Wx = 0.6, so CDS admitted at
    delta_r >= 0.6 -- while the correct full-simplex bound requires
    delta_r >= 2.0. The empty region was strictly easier to pass than the whole
    simplex, which is exactly backwards for a safety gate.
    """
    from certification.cds_test import CDSGate
    from certification.pds_test import PDSGate

    delta_n = np.array([0.5, -2.0])
    empty_region = np.array([0.4, 0.4])  # sum 0.8 < 1

    for gate in (CDSGate(), PDSGate(epsilon=0.1)):
        with pytest.raises(ValueError, match=r"sum\(s\) >= 1"):
            gate.admit(1.0, delta_n, None, empty_region)
        with pytest.raises(ValueError, match=r"sum\(s\) >= 1"):
            gate.get_admission_margin(1.0, delta_n, None, empty_region)

    # The full simplex is the honest comparison point and still works.
    assert CDSGate().admit(2.0, delta_n, None, np.array([1.0, 1.0])) is True
    assert CDSGate().admit(1.9, delta_n, None, np.array([1.0, 1.0])) is False


def test_wx_worst_case_and_greedy_agree_on_rejection():
    """Both entry points must reject the same inputs.

    _compute_wx_worst_case validates via _validate_wx_geometry; the gates reach
    the solver directly. Both must refuse an empty region so there is no path
    into the worst-case computation that accepts one.
    """
    from library.skill_library import _compute_wx_worst_case
    from utils.support_geometry import worst_case_over_support_region

    delta_n = np.array([0.5, -2.0])
    empty_region = np.array([0.4, 0.4])

    with pytest.raises(ValueError):
        _compute_wx_worst_case(delta_n, np.eye(2), empty_region)

    with pytest.raises(ValueError):
        worst_case_over_support_region(delta_n, empty_region)


def test_gates_support_values_path_works_at_m5():
    """Both gates must evaluate a five-objective region without vertices."""
    from certification.cds_test import CDSGate
    from certification.pds_test import PDSGate

    support = np.full(5, 0.5)
    delta_n = np.array([0.3, 0.4, 0.5, 0.2, 0.6])

    assert CDSGate().admit(0.9, delta_n, None, support) is True
    assert PDSGate(epsilon=0.1).admit(0.9, delta_n, None, support) is True

    # A coordinate the region can load mass onto defeats a small delta_r.
    hostile = np.array([0.3, 0.4, 0.5, 0.2, -2.0])
    assert CDSGate().admit(0.05, hostile, None, support) is False

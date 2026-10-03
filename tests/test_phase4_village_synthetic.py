"""
tests/test_phase4_village_synthetic.py
Phase 4 exit-gate tests for the Synthetic Six-Objective Village Pipeline.

These tests verify that:
  - The pipeline produces all required artifacts
  - At least one option was admitted and at least one rejected
  - All admitted certificates carry correct 6D schema identity
  - No rejected option entered the library
  - Artifacts round-trip correctly from disk
  - SkillGenerator and MDN forward-pass shapes are correct for 6D (torch-gated)
  - Paired rollouts are reproducible (same seed -> same delta)

Run with the project venv (hyperon + torch required for all tests):
    .venv\\Scripts\\python.exe -m pytest tests/test_phase4_village_synthetic.py -v

Run without torch (skips generator/MDN smoke tests):
    python -m pytest tests/test_phase4_village_synthetic.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# ── Paths produced by the pipeline ────────────────────────────────────────────
CERT_FILE    = Path("data/village_synthetic/v1/certificates.metta")
LIBRARY_FILE = Path("data/village_synthetic/v1/library.json")
REPORT_JSON  = Path("demo/artifacts/village_synthetic/v1/admission_report.json")
REPORT_MD    = Path("demo/artifacts/village_synthetic/v1/admission_report.md")

EXPECTED_DOMAIN        = "minecraft_village_defense_trade"
EXPECTED_SCHEMA_VER    = "1.0"
EXPECTED_MOTIVE_NAMES  = [
    "Safety", "Reputation", "DeadlineSlack",
    "InventoryValue", "Sustainability", "Infrastructure",
]
EXPECTED_MOTIVE_DIM    = 6


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_report() -> dict:
    """Load the admission report JSON produced by the pipeline."""
    assert REPORT_JSON.exists(), (
        f"Admission report not found at {REPORT_JSON}. "
        "Run: .venv\\Scripts\\python.exe -m demo.run_village_synthetic_pipeline"
    )
    return json.loads(REPORT_JSON.read_text(encoding="utf-8"))


def _load_admitted_certs() -> list[dict]:
    """Return admitted episode records from the report."""
    report = _load_report()
    # The AdmissionReport stores example_admitted_skill, not all records.
    # We re-run the pipeline logic at the cert-file level for full verification.
    return []  # used only as a helper marker; actual tests use cert_store


def _load_cert_store():
    """Load the MeTTa certificate store from the generated file."""
    from certification.metta_storage import CertificateStore
    store = CertificateStore()
    store.load_from_file(str(CERT_FILE))
    return store


def _load_library_json() -> dict:
    """Load the raw library JSON."""
    assert LIBRARY_FILE.exists(), f"Library file not found at {LIBRARY_FILE}"
    return json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))


# ── Artifact existence tests ───────────────────────────────────────────────────

class TestArtifactsExist:
    """Verify the pipeline produced all required output files."""

    def test_cert_file_exists(self):
        assert CERT_FILE.exists(), (
            f"{CERT_FILE} missing — run the pipeline first"
        )

    def test_cert_file_not_empty(self):
        assert CERT_FILE.exists()
        assert CERT_FILE.stat().st_size > 0, "certificates.metta is empty"

    def test_library_file_exists(self):
        assert LIBRARY_FILE.exists(), f"{LIBRARY_FILE} missing"

    def test_report_json_exists(self):
        assert REPORT_JSON.exists(), f"{REPORT_JSON} missing"

    def test_report_md_exists(self):
        assert REPORT_MD.exists(), f"{REPORT_MD} missing"

    def test_report_json_is_valid(self):
        data = _load_report()
        assert isinstance(data, dict)
        assert "admitted" in data
        assert "rejected" in data
        assert "total_attempted" in data


# ── Admission outcome tests ────────────────────────────────────────────────────

class TestAdmissionOutcomes:
    """Verify the pipeline admitted and rejected the correct number of options."""

    def test_at_least_one_admitted(self):
        report = _load_report()
        assert report["admitted"] >= 1, (
            f"Expected at least 1 admission, got {report['admitted']}"
        )

    def test_at_least_one_rejected(self):
        report = _load_report()
        assert report["rejected"] >= 1, (
            f"Expected at least 1 rejection, got {report['rejected']}"
        )

    def test_admitted_plus_rejected_equals_total(self):
        report = _load_report()
        assert report["admitted"] + report["rejected"] == report["total_attempted"]

    def test_library_size_matches_admitted_count(self):
        report = _load_report()
        lib = _load_library_json()
        lib_skills = lib.get("skills", lib.get("entries", []))
        assert len(lib_skills) == report["admitted"], (
            f"Library has {len(lib_skills)} entries but report says {report['admitted']} admitted"
        )

    def test_cert_store_count_matches_admitted_count(self):
        report = _load_report()
        store = _load_cert_store()
        assert store.count() == report["admitted"], (
            f"Cert store has {store.count()} certs but report says {report['admitted']} admitted"
        )


# ── Certificate content tests ──────────────────────────────────────────────────

class TestCertificateContent:
    """Verify every admitted certificate carries correct 6D schema identity."""

    def test_all_certs_use_full_simplex(self):
        store = _load_cert_store()
        for cert in store.load_all():
            assert cert.weight_region_type == "FULL_SIMPLEX", (
                f"cert '{cert.skill_id}' has weight_region_type={cert.weight_region_type!r}, "
                "expected 'FULL_SIMPLEX'"
            )

    def test_no_cert_uses_mdn_wx(self):
        store = _load_cert_store()
        for cert in store.load_all():
            assert cert.weight_region_type != "MDN_WX", (
                f"cert '{cert.skill_id}' incorrectly uses MDN_WX in a FULL_SIMPLEX pipeline"
            )

    def test_delta_n_is_six_dimensional(self):
        store = _load_cert_store()
        for cert in store.load_all():
            assert len(cert.delta_n) == EXPECTED_MOTIVE_DIM, (
                f"cert '{cert.skill_id}' has delta_n length={len(cert.delta_n)}, "
                f"expected {EXPECTED_MOTIVE_DIM}"
            )

    def test_cert_has_domain_id(self):
        store = _load_cert_store()
        for cert in store.load_all():
            assert cert.domain_id == EXPECTED_DOMAIN, (
                f"cert '{cert.skill_id}' domain_id={cert.domain_id!r}, "
                f"expected {EXPECTED_DOMAIN!r}"
            )

    def test_cert_has_motive_schema_version(self):
        store = _load_cert_store()
        for cert in store.load_all():
            assert cert.motive_schema_version == EXPECTED_SCHEMA_VER, (
                f"cert '{cert.skill_id}' motive_schema_version={cert.motive_schema_version!r}, "
                f"expected {EXPECTED_SCHEMA_VER!r}"
            )

    def test_cert_has_correct_motive_names(self):
        store = _load_cert_store()
        for cert in store.load_all():
            assert cert.motive_names is not None, (
                f"cert '{cert.skill_id}' has motive_names=None"
            )
            assert list(cert.motive_names) == EXPECTED_MOTIVE_NAMES, (
                f"cert '{cert.skill_id}' motive_names={list(cert.motive_names)}, "
                f"expected {EXPECTED_MOTIVE_NAMES}"
            )

    def test_cds_cert_margin_is_non_negative(self):
        """CDS admission margin = delta_r + min(delta_n) must be >= 0."""
        store = _load_cert_store()
        for cert in store.load_all():
            if cert.gate_type == "CDS":
                margin = float(cert.delta_r) + float(np.min(cert.delta_n))
                assert margin >= 0.0, (
                    f"CDS cert '{cert.skill_id}' has negative margin {margin:.6f}"
                )

    def test_all_certs_have_environment_field(self):
        store = _load_cert_store()
        for cert in store.load_all():
            assert cert.environment is not None and cert.environment != "", (
                f"cert '{cert.skill_id}' is missing environment field"
            )

    def test_all_certs_have_baseline_id(self):
        store = _load_cert_store()
        for cert in store.load_all():
            assert cert.baseline_id == "idle_policy_v1", (
                f"cert '{cert.skill_id}' baseline_id={cert.baseline_id!r}"
            )


# ── Round-trip / reload tests ──────────────────────────────────────────────────

class TestArtifactRoundTrip:
    """Verify saved artifacts reload correctly from disk."""

    def test_cert_file_round_trips_count(self):
        store1 = _load_cert_store()
        # Load a second independent store from the same file
        from certification.metta_storage import CertificateStore
        store2 = CertificateStore()
        store2.load_from_file(str(CERT_FILE))
        assert store2.count() == store1.count(), (
            f"Reload count mismatch: {store2.count()} != {store1.count()}"
        )

    def test_cert_file_round_trips_skill_ids(self):
        store1 = _load_cert_store()
        from certification.metta_storage import CertificateStore
        store2 = CertificateStore()
        store2.load_from_file(str(CERT_FILE))
        ids1 = {c.skill_id for c in store1.load_all()}
        ids2 = {c.skill_id for c in store2.load_all()}
        assert ids1 == ids2, f"Skill IDs differ on reload: {ids1} != {ids2}"

    def test_cert_file_round_trips_delta_n(self):
        """delta_n values survive MeTTa serialise → deserialise within float32 precision."""
        store1 = _load_cert_store()
        from certification.metta_storage import CertificateStore
        store2 = CertificateStore()
        store2.load_from_file(str(CERT_FILE))
        for c1 in store1.load_all():
            c2 = store2.get_certificate(c1.skill_id)
            assert c2 is not None
            np.testing.assert_allclose(
                np.array(c1.delta_n), np.array(c2.delta_n), rtol=1e-5,
                err_msg=f"delta_n mismatch for '{c1.skill_id}' after reload"
            )

    def test_library_file_round_trips_skill_ids(self):
        lib = _load_library_json()
        lib_skills = lib.get("skills", lib.get("entries", []))
        store = _load_cert_store()
        cert_ids = {c.skill_id for c in store.load_all()}
        lib_ids  = {s["skill_id"] if isinstance(s, dict) else s for s in lib_skills}
        assert lib_ids == cert_ids, (
            f"Library skill IDs {lib_ids} don't match cert store IDs {cert_ids}"
        )


# ── Rejected skills safety test ────────────────────────────────────────────────

class TestRejectedSkillsSafety:
    """Verify no rejected option entered the library."""

    KNOWN_REJECTED = {
        "torch_corridor_v1",
        "infrastructure_first_v1",
        "wasteful_fuel_v1",
        "aggressive_burn_v1",
    }

    def test_rejected_skills_not_in_cert_store(self):
        store = _load_cert_store()
        cert_ids = {c.skill_id for c in store.load_all()}
        overlap = self.KNOWN_REJECTED & cert_ids
        assert not overlap, (
            f"Rejected skills found in cert store: {overlap}"
        )

    def test_rejected_skills_not_in_library(self):
        lib = _load_library_json()
        lib_skills = lib.get("skills", lib.get("entries", []))
        lib_ids = {s["skill_id"] if isinstance(s, dict) else s for s in lib_skills}
        overlap = self.KNOWN_REJECTED & lib_ids
        assert not overlap, (
            f"Rejected skills found in library: {overlap}"
        )


# ── Reproducibility test ───────────────────────────────────────────────────────

class TestPairedRolloutReproducibility:
    """Same seed must produce identical delta values on re-run."""

    def test_same_seed_produces_same_delta_r(self):
        from village_sim.baseline import idle_policy, run_policy
        from demo.run_village_synthetic_pipeline import (
            _trade_focus_policy, GAMMA, MAX_STEPS,
        )
        seed = 42

        # Run twice with the same seed
        r_base_1, n_base_1 = run_policy(idle_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        r_cand_1, n_cand_1 = run_policy(_trade_focus_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)

        r_base_2, n_base_2 = run_policy(idle_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        r_cand_2, n_cand_2 = run_policy(_trade_focus_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)

        assert r_base_1 == r_base_2, "Baseline payoff not reproducible with same seed"
        assert r_cand_1 == r_cand_2, "Candidate payoff not reproducible with same seed"

    def test_same_seed_produces_same_delta_n(self):
        from village_sim.baseline import idle_policy, run_policy
        from demo.run_village_synthetic_pipeline import (
            _trade_focus_policy, GAMMA, MAX_STEPS,
        )
        seed = 123
        _, n_base_1 = run_policy(idle_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        _, n_cand_1 = run_policy(_trade_focus_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        _, n_base_2 = run_policy(idle_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
        _, n_cand_2 = run_policy(_trade_focus_policy, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)

        np.testing.assert_array_equal(n_base_1, n_base_2,
                                      err_msg="Baseline motives not reproducible")
        np.testing.assert_array_equal(n_cand_1, n_cand_2,
                                      err_msg="Candidate motives not reproducible")

    def test_different_seeds_produce_different_states(self):
        """Seeds must actually change the environment state (not degenerate)."""
        from village_sim.baseline import idle_policy, run_policy
        from demo.run_village_synthetic_pipeline import GAMMA, MAX_STEPS

        _, n1 = run_policy(idle_policy, seed=42,  max_steps=MAX_STEPS, gamma=GAMMA)
        _, n2 = run_policy(idle_policy, seed=999, max_steps=MAX_STEPS, gamma=GAMMA)
        # At least one motive should differ between seeds
        assert not np.array_equal(n1, n2), (
            "Seeds 42 and 999 produced identical motive vectors — RNG seeding is broken"
        )


# ── SkillGenerator and MDN 6D smoke tests (torch-gated) ───────────────────────

class TestGeneratorMDNSixDimensionalSmoke:
    """
    Verify that SkillGenerator and MotiveDecompositionNetwork accept 6D inputs.

    These are pure shape/forward-pass tests — no training, no quality claims.
    Skipped automatically when torch is not installed.
    """

    def test_skill_generator_6d_output_shapes(self):
        torch = pytest.importorskip("torch")
        from generator.skill_generator import SkillGenerator

        model = SkillGenerator(input_dim=6, hidden_dim=32, motive_dim=6)
        model.eval()

        # Single observation
        obs = torch.randn(6)
        with torch.no_grad():
            payoff, motives = model(obs)
        assert payoff.shape  == (1,), f"Expected payoff shape (1,), got {payoff.shape}"
        assert motives.shape == (6,), f"Expected motives shape (6,), got {motives.shape}"

    def test_skill_generator_6d_batch_output_shapes(self):
        torch = pytest.importorskip("torch")
        from generator.skill_generator import SkillGenerator

        model = SkillGenerator(input_dim=6, hidden_dim=32, motive_dim=6)
        model.eval()

        # Batched input
        batch = torch.randn(4, 6)
        with torch.no_grad():
            payoff, motives = model(batch)
        assert payoff.shape  == (4, 1), f"Expected (4, 1), got {payoff.shape}"
        assert motives.shape == (4, 6), f"Expected (4, 6), got {motives.shape}"

    def test_mdn_6d_forward_inference_shapes(self):
        torch = pytest.importorskip("torch")
        from generator.mdn import MotiveDecompositionNetwork

        # num_skills must NOT equal num_objectives (would be a magic-number coincidence)
        mdn = MotiveDecompositionNetwork(
            input_dim=6,
            num_objectives=6,
            hidden_dim=32,
            num_skills=10,
        )
        mdn.eval()

        obs = torch.randn(6)
        with torch.no_grad():
            alpha, support = mdn.forward_inference(obs)
        assert alpha.shape   == (6,), f"Expected alpha shape (6,), got {alpha.shape}"
        assert support.shape == (6,), f"Expected support shape (6,), got {support.shape}"

    def test_mdn_6d_forward_auxiliary_shapes(self):
        torch = pytest.importorskip("torch")
        from generator.mdn import MotiveDecompositionNetwork

        mdn = MotiveDecompositionNetwork(
            input_dim=6,
            num_objectives=6,
            hidden_dim=32,
            num_skills=10,
        )
        mdn.eval()

        obs      = torch.randn(6)
        skill_id = torch.tensor(0, dtype=torch.long)
        with torch.no_grad():
            gate_logit, q_hat = mdn.forward_auxiliary(obs, skill_id)
        assert gate_logit.shape == (), f"Expected scalar gate_logit, got {gate_logit.shape}"
        assert q_hat.shape      == (6,), f"Expected q_hat shape (6,), got {q_hat.shape}"

    def test_mdn_6d_alpha_is_positive(self):
        """Softplus output for alpha must be strictly positive (valid Dirichlet params)."""
        torch = pytest.importorskip("torch")
        from generator.mdn import MotiveDecompositionNetwork

        mdn = MotiveDecompositionNetwork(input_dim=6, num_objectives=6, num_skills=10)
        mdn.eval()
        obs = torch.randn(6)
        with torch.no_grad():
            alpha, _ = mdn.forward_inference(obs)
        assert torch.all(alpha > 0).item(), (
            f"All alpha values must be positive, got {alpha.tolist()}"
        )

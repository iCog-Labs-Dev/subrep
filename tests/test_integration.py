"""
test_integration.py — End-to-end integration tests for the SubRep pipeline.

Tests live system connectivity using real environment execution and disk
persistence. Unlike component tests, these verify the full chain:
    Execute → Certify → Store (MeTTa) → Library (Python)

Run with:
    python -m pytest tests/test_integration.py -v
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pytest

from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor
from baseline.idle_policy import IdlePolicy
from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from certification.certificate_schema import Certificate
from certification.metta_storage import CertificateStore
from library.skill_library import SkillLibrary
from library.skill_selector import SkillSelector

# ── Shared constants ──────────────────────────────────────────────────────────
GAMMA       = 0.99
SEED        = 42
ENV_NAME    = "MO-LunarLander-v3"
VERSION     = "0.1.0"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_baseline(env: SubRepEnv, num_episodes: int = 5) -> dict:
    """Run a quick baseline and return stats."""
    idle = IdlePolicy(env=env, idle_action=0, gamma=GAMMA)
    return idle.run_baseline_episodes(num_episodes=num_episodes, seed=SEED)


def _make_cert(
    skill_id: str,
    delta_r: float,
    delta_n: tuple[float, float],
    margin: float,
    gate_type: str = "CDS",
    epsilon: float = 0.0,
) -> Certificate:
    return Certificate(
        skill_id=skill_id,
        gate_type=gate_type,
        delta_r=delta_r,
        delta_n=delta_n,
        admission_margin=margin,
        epsilon=epsilon,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=SEED,
        gamma=GAMMA,
        baseline_id="idle_policy_v1",
        environment=ENV_NAME,
        episode_length=50,
        version=VERSION,
    )


def _random_policy(env: SubRepEnv):
    """A callable random policy for use as a skill placeholder."""
    return lambda obs: env.env.action_space.sample()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_real_execution_to_certification():
    """
    Run a real episode in MO-LunarLander, compute real Δr/Δn,
    and verify CDS gate produces a valid boolean decision.
    """
    env = SubRepEnv(seed=SEED)
    baseline_stats = _build_baseline(env, num_episodes=5)
    calculator = ImprovementCalculator(baseline_stats)
    gate = CDSGate()
    pds_gate = PDSGate(epsilon=0.1)

    # Run one real episode with a random policy.
    policy = _random_policy(env)
    executor = SkillExecutor(env=env, policy_fn=policy, gamma=GAMMA, max_steps=100)
    payoff, motives, _terminated = executor.run_episode()

    delta_r, delta_n = calculator.compute_improvements(payoff, motives)

    # Sanity: improvements are finite scalars/vectors.
    assert np.isfinite(delta_r), f"delta_r must be finite, got {delta_r}"
    assert np.all(np.isfinite(delta_n)), f"delta_n must be finite, got {delta_n}"
    assert delta_n.shape == (2,), f"Expected shape (2,), got {delta_n.shape}"

    # Gate must return a bool without raising.
    result = gate.admit(delta_r, delta_n)
    pds_result = pds_gate.admit(delta_r, delta_n)
    assert isinstance(result, bool), f"gate.admit() must return bool, got {type(result)}"
    assert isinstance(pds_result, bool), f"pds_gate.admit() must return bool, got {type(pds_result)}"

    # Margin must be consistent with Boolean decision.
    margin = gate.get_admission_margin(delta_r, delta_n)
    pds_margin = pds_gate.get_admission_margin(delta_r, delta_n)
    assert (margin >= 0.0) == result, (
        f"CDS Margin {margin:.4f} is inconsistent with admit={result}"
    )
    assert (pds_margin >= 0.0) == pds_result, (
        f"PDS Margin {pds_margin:.4f} is inconsistent with admit={pds_result}"
    )


def test_safety_rejection_logic():
    """
    Critical Test: A skill must pass BOTH the Store check and the Math check.
    
    1. Rejection due to missing certificate in Store (Identity check).
    2. Rejection due to failing math (Chain of Safety check).
    """
    cert_store = CertificateStore()
    library = SkillLibrary(cert_store=cert_store)
    env = SubRepEnv(seed=SEED)
    policy = _random_policy(env)

    # --- Scenario A: Identity-based rejection (Not in store) ---
    # Even if math passes schema, library rejects if ID is unknown.
    failing_cert_a = _make_cert(
        skill_id="bad_id_001",
        delta_r=10.0,
        delta_n=(5.0, 5.0),
        margin=15.0
    )
    assert library.add_skill("bad_id_001", failing_cert_a, policy) is False, \
        "Library must reject a skill with no certificate in store"

    # --- Scenario B: Mathematics-based rejection (In store, but bad math) ---
    # We purposefully 'smear' a bad cert into the store (simulating a compromise).
    failing_cert_b = _make_cert(
        skill_id="bad_math_001",
        delta_r=-10.0,
        delta_n=(-5.0, -5.0),
        margin=0.1 # Fake positive margin
    )
    cert_store.add(failing_cert_b) # Forced injection into store
    
    # The library must still reject it because our internal verify(math) fails.
    assert cert_store.contains("bad_math_001") is True
    result = library.add_skill("bad_math_001", failing_cert_b, policy)
    
    assert result is False, "Library must reject a skill with failing math, even if in Store"
    assert library.count() == 0, "No bad skill should enter the library"


def test_pds_admits_within_epsilon():
    """
    Verify PDS epsilon logic: admits skills that fall within (Δr + min(Δn) + epsilon >= 0)
    even if they fail the strict CDS test (Δr + min(Δn) < 0).
    """
    cert_store = CertificateStore()
    library = SkillLibrary(cert_store=cert_store)
    env = SubRepEnv(seed=SEED)
    policy = _random_policy(env)

    # CDS formula: Δr + min(Δn) >= 0
    # PDS formula: Δr + min(Δn) >= -epsilon
    
    eps = 2.0
    cert = _make_cert(
        skill_id="pds_boundary_skill",
        delta_r=10.0,
        delta_n=(-11.0, -11.0),
        margin=1.0, # 10 - 11 + 2.0 = 1.0 (Passing margin for PDS)
        gate_type="PDS",
        epsilon=eps
    )
    
    cert_store.add(cert)
    success = library.add_skill("pds_boundary_skill", cert, policy)
    
    assert success is True, "PDS must admit skills within the epsilon budget"
    entry = library.get_skill("pds_boundary_skill")
    assert entry is not None
    assert entry.gate_type == "PDS"
    assert entry.epsilon == eps
    assert library.count() == 1


def test_meTTA_to_python_handoff():
    """
    Add a certificate to CertificateStore (MeTTa), then verify
    SkillLibrary successfully validates against it before adding the skill.
    """
    cert_store = CertificateStore()
    library = SkillLibrary(cert_store=cert_store)

    # A clearly passing certificate (delta_r + min(delta_n) > 0).
    cert = _make_cert(
        skill_id="good_skill_001",
        delta_r=5.0,
        delta_n=(2.0, 3.0),
        margin=7.0,   # 5.0 + min(2.0, 3.0) = 7.0
    )

    # Step 1: Add to MeTTa store.
    added_to_store = cert_store.add(cert)
    assert added_to_store is True, "CertificateStore.add() must return True for new cert"
    assert cert_store.contains("good_skill_001"), "CertificateStore must contain the cert"

    # Step 2: Add to library — should succeed because store has the cert.
    env = SubRepEnv(seed=SEED)
    policy = _random_policy(env)
    added_to_library = library.add_skill("good_skill_001", cert, policy)

    assert added_to_library is True, "SkillLibrary must accept skill with valid cert"
    assert library.count() == 1, f"Library count must be 1, got {library.count()}"

    # Step 3: Retrieve and verify identity.
    entry = library.get_skill("good_skill_001")
    assert entry is not None
    assert entry.skill_id == "good_skill_001"
    assert entry.gate_type == "CDS"


def test_full_pipeline_cycle():
    """
    Run a mini end-to-end loop (5 episodes). Assert no exceptions,
    no type errors, no state corruption, and that the safety invariant
    holds: cert_store.count() == library.count() at all times.
    """
    env = SubRepEnv(seed=SEED)
    baseline_stats = _build_baseline(env, num_episodes=5)
    calculator = ImprovementCalculator(baseline_stats)
    gate = CDSGate()
    cert_store = CertificateStore()
    library = SkillLibrary(cert_store=cert_store)
    selector = SkillSelector(library=library, seed=SEED)

    for ep in range(1, 6):
        skill_id = f"integration_skill_{ep:03d}"

        # SELECT (or random fallback)
        # SubRepEnv.reset() uses its internal seed — no argument accepted.
        obs, _ = env.reset()
        selected = selector.select_random(obs)
        policy = _random_policy(env)

        # EXECUTE
        executor = SkillExecutor(
            env=env, policy_fn=policy, gamma=GAMMA, max_steps=100
        )
        payoff, motives, _terminated = executor.run_episode()
        episode_length = executor.last_run_info.get("steps", 100)

        # CERTIFY
        delta_r, delta_n = calculator.compute_improvements(payoff, motives)
        admitted = gate.admit(delta_r, delta_n)
        margin = gate.get_admission_margin(delta_r, delta_n)

        if admitted:
            cert = _make_cert(
                skill_id=skill_id,
                delta_r=float(delta_r),
                delta_n=(float(delta_n[0]), float(delta_n[1])),
                margin=max(0.0, float(margin)),
            )
            cert_store.add(cert)
            library.add_skill(skill_id, cert, policy)

        # Invariant: store and library counts must be in sync.
        assert cert_store.count() == library.count(), (
            f"Ep {ep}: cert_store.count()={cert_store.count()} != "
            f"library.count()={library.count()}"
        )

        # Type safety: admitted skills must have valid delta values.
        if admitted:
            entry = library.get_skill(skill_id)
            assert entry is not None
            assert isinstance(entry.delta_r, float)
            assert len(entry.delta_n) == 2

    # Final: library count is non-negative and consistent.
    assert library.count() >= 0
    assert cert_store.count() >= 0
    assert cert_store.count() == library.count()

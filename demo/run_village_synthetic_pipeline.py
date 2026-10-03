"""
demo/run_village_synthetic_pipeline.py
Phase 4 — Synthetic Six-Objective Village SubRep Pipeline.

Runs paired baseline/candidate rollouts through the complete SubRep flow:

    same-seed paired rollouts → delta_r / delta_n[6]
    → CDS/PDS gate → Certificate (6D, FULL_SIMPLEX, Minecraft domain)
    → MeTTa cert store → SkillLibrary
    → AdmissionReport (JSON + Markdown)
    → reload-verification (artifacts round-trip)

This is a pure-Python, torch-free synthetic validation of the 6D SubRep
pipeline before any real Minecraft server is involved (Phase 5+).

Exit gates (asserted at end of run):
    - At least one option admitted to the library
    - At least one option rejected
    - cert_store.count() == library.count() after every admission
    - Reloaded cert file contains the same number of certificates

Usage:
    python -m demo.run_village_synthetic_pipeline
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from village_sim.baseline import idle_policy, run_policy
from village_sim.motives import MOTIVE_NAMES
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from certification.certificate_schema import Certificate
from certification.metta_storage import CertificateStore
from library.skill_library import SkillLibrary
from utils.admission_report import AdmissionReport


# ── Configuration ──────────────────────────────────────────────────────────────
# Motive order (canonical, fixed for Minecraft domain):
#   [Safety, Reputation, DeadlineSlack, InventoryValue, Sustainability, Infrastructure]

GAMMA         = 0.99
MAX_STEPS     = 100
# Three seeds per option — multi-seed evidence mirrors Phase 8 statistical pattern
SEEDS: list[int] = [42, 123, 7]
PDS_EPSILON   = 0.3
VERSION       = "0.1.0"
BASELINE_ID   = "idle_policy_v1"
DOMAIN        = "minecraft_village_defense_trade"
SCHEMA_VER    = "1.0"
ENVIRONMENT   = "village_sim_v1"

DATA_DIR      = Path("data/village_synthetic/v1")
CERT_FILE     = DATA_DIR / "certificates.metta"
LIBRARY_FILE  = DATA_DIR / "library.json"
REPORT_JSON   = Path("demo/artifacts/village_synthetic/v1/admission_report.json")
REPORT_MD     = Path("demo/artifacts/village_synthetic/v1/admission_report.md")
# ──────────────────────────────────────────────────────────────────────────────


# ── Candidate option policies ──────────────────────────────────────────────────
# Each policy takes a VillageState snapshot and returns an action string.
# Action set: "idle", "trade", "torch_corridor", "iron_golem_spawn",
#             "reputation_first", "discount_chain", "archer_kite"

def _torch_corridor_policy(state) -> str:
    """Improve infrastructure via torches up to 90%, then trade to complete the task."""
    if state.fuel >= 1 and state.infrastructure_pct < 0.9:
        return "torch_corridor"
    return "trade"


def _infrastructure_first_policy(state) -> str:
    """Prioritise iron-golem spawning for strong defence, then torch, then trade."""
    if state.fuel >= 3 and state.infrastructure_pct < 0.8:
        return "iron_golem_spawn"
    if state.fuel >= 1 and state.infrastructure_pct < 0.9:
        return "torch_corridor"
    return "trade"


def _trade_focus_policy(state) -> str:
    """Always trade — completes the delivery contract in the fewest steps."""
    return "trade"


def _reputation_builder_policy(state) -> str:
    """Build reputation and unlock discounts before trading heavily."""
    if state.reputation < 0.8:
        return "reputation_first"
    if state.emerald_price > 2.0:
        return "discount_chain"
    return "trade"


def _wasteful_fuel_policy(state) -> str:
    """Burn all fuel on iron-golem spawns then idle — never trades.

    Engineered to fail certification: Sustainability collapses (fuel exhausted)
    while the delivery task never completes (no trading), so delta_r is near 0
    and delta_n[Sustainability] << 0.  Both CDS and PDS gates will reject this.
    """
    if state.fuel >= 3:
        return "iron_golem_spawn"
    return "idle"  # fuel exhausted — no trading means task always fails


def _aggressive_burn_policy(state) -> str:
    """Burn all fuel on golems then trade — high Infrastructure, collapsed Sustainability.

    Completes the task (so delta_r > 0) but Sustainability suffers so severely
    that min(delta_n) << 0 and delta_r cannot compensate within CDS or PDS-0.3.
    This is a real-world anti-pattern: over-investing in defence infrastructure
    at the expense of sustainable resource usage.
    """
    if state.fuel >= 3:
        return "iron_golem_spawn"
    return "trade"  # exhausted fuel — finish task but Sustainability stays low


# Ordered list of (option_id, policy_fn) pairs evaluated by the pipeline.
CANDIDATES: list[tuple[str, object]] = [
    ("torch_corridor_v1",       _torch_corridor_policy),
    ("infrastructure_first_v1", _infrastructure_first_policy),
    ("trade_focus_v1",          _trade_focus_policy),
    ("reputation_builder_v1",   _reputation_builder_policy),
    ("wasteful_fuel_v1",        _wasteful_fuel_policy),
    ("aggressive_burn_v1",      _aggressive_burn_policy),
]


# ── Rollout helpers ────────────────────────────────────────────────────────────

def _run_paired(candidate_fn, seed: int) -> tuple[float, np.ndarray]:
    """Run baseline then candidate from the SAME seed.

    Identical reset seed guarantees both policies start from the same world
    state, mirroring the real Minecraft snapshot-restore paired-rollout
    pattern defined in Phase 8.

    Returns:
        delta_r: Scalar payoff improvement over baseline.
        delta_n: 6D motive improvement vector over baseline.
    """
    r_base, n_base = run_policy(idle_policy,  seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
    r_cand, n_cand = run_policy(candidate_fn, seed=seed, max_steps=MAX_STEPS, gamma=GAMMA)
    delta_r = float(r_cand) - float(r_base)
    delta_n = (np.asarray(n_cand, dtype=np.float32)
               - np.asarray(n_base, dtype=np.float32))
    return delta_r, delta_n


def _aggregate_seeds(
    candidate_fn,
    seeds: list[int],
) -> tuple[float, np.ndarray, list[tuple[float, np.ndarray]]]:
    """Aggregate paired rollouts over multiple seeds.

    Returns:
        mean_delta_r: Mean payoff improvement across seeds.
        mean_delta_n: Mean 6D motive improvement across seeds.
        pairs:        Raw (delta_r, delta_n) per seed for audit logging.
    """
    pairs: list[tuple[float, np.ndarray]] = [
        _run_paired(candidate_fn, seed) for seed in seeds
    ]
    mean_delta_r = float(np.mean([p[0] for p in pairs]))
    mean_delta_n = np.mean([p[1] for p in pairs], axis=0).astype(np.float32)
    return mean_delta_r, mean_delta_n, pairs


# ── Certificate builder ────────────────────────────────────────────────────────

def _make_certificate(
    option_id: str,
    delta_r: float,
    delta_n: np.ndarray,
    margin: float,
    gate_type: str,
    epsilon: float,
) -> Certificate:
    """Build a fully-validated 6D Certificate with Minecraft domain identity.

    The domain_id / motive_schema_version / motive_names triple is required
    by the Phase 3 schema contract so the SkillLibrary can enforce cross-domain
    runtime rejection and legacy quarantine rules.
    """
    return Certificate(
        skill_id=option_id,
        gate_type=gate_type,
        delta_r=float(delta_r),
        delta_n=tuple(float(v) for v in delta_n),
        admission_margin=float(margin),
        epsilon=float(epsilon),
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=SEEDS[0],
        gamma=GAMMA,
        baseline_id=BASELINE_ID,
        environment=ENVIRONMENT,
        episode_length=MAX_STEPS,
        version=VERSION,
        weight_region_type="FULL_SIMPLEX",
        # Minecraft domain identity — marks this as a 6D Minecraft artifact
        domain_id=DOMAIN,
        motive_schema_version=SCHEMA_VER,
        motive_names=tuple(MOTIVE_NAMES),
        # MDN audit fields not used in synthetic Phase 4
        certification_context=None,
        mdn_alpha=None,
        wx_support_directions=None,
        wx_support_values=None,
    )


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline() -> dict:
    """Run the Phase 4 synthetic six-objective village pipeline.

    Returns a dict of summary statistics suitable for use in exit-gate tests.
    """
    print("=" * 68)
    print("  Phase 4 — Synthetic Six-Objective Village SubRep Pipeline")
    print("=" * 68)
    print(f"  Domain   : {DOMAIN}")
    print(f"  Schema   : v{SCHEMA_VER}")
    print(f"  Motives  : {MOTIVE_NAMES}")
    print(f"  Seeds    : {SEEDS}  ({len(SEEDS)} per option)")
    print(f"  Gamma    : {GAMMA}  |  Max steps : {MAX_STEPS}  |  PDS e : {PDS_EPSILON}")
    print(f"  Artifacts: {DATA_DIR}")
    print()

    # ── 1. Setup: gates and stores ─────────────────────────────────────────────
    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=PDS_EPSILON)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Fresh stores — namespaced to village_synthetic/v1, separate from LunarLander data/
    cert_store = CertificateStore()
    library    = SkillLibrary(cert_store=cert_store, save_path=str(LIBRARY_FILE))
    report     = AdmissionReport()

    # ── 2. Candidate evaluation loop ───────────────────────────────────────────
    admitted  = 0
    rejected  = 0
    cds_count = 0
    pds_count = 0
    episode_records: list[dict] = []

    print(
        f"{'Option':<28}  {'dR':>8}  {'min(dN)':>9}  "
        f"{'CDS':>4}  {'PDS':>4}  {'Result':<16}  {'Lib':>4}"
    )
    print("-" * 82)

    for option_id, policy_fn in CANDIDATES:
        # Multi-seed paired rollouts
        mean_dr, mean_dn, pairs = _aggregate_seeds(policy_fn, SEEDS)

        admitted_cds  = cds_gate.admit(mean_dr, mean_dn)
        admitted_pds  = pds_gate.admit(mean_dr, mean_dn)
        admitted_flag = admitted_cds or admitted_pds
        failure_reason: str | None = None

        if admitted_flag:
            gate_type = "CDS" if admitted_cds else "PDS"
            epsilon   = 0.0 if admitted_cds else PDS_EPSILON
            margin    = (cds_gate.get_admission_margin(mean_dr, mean_dn)
                         if admitted_cds
                         else pds_gate.get_admission_margin(mean_dr, mean_dn))

            cert = _make_certificate(option_id, mean_dr, mean_dn, margin, gate_type, epsilon)

            store_added = cert_store.add(cert)
            lib_added   = False

            if store_added:
                lib_added = library.add_skill(option_id, cert, policy_fn)
                if lib_added:
                    cds_count += int(admitted_cds)
                    pds_count += int(not admitted_cds)
                    admitted  += 1
                    result_str = f"ADMITTED [{gate_type}]"
                else:
                    # Library re-verification failed — roll back cert_store to stay in sync
                    cert_store.remove_skill(option_id)
                    failure_reason = "library.add_skill() re-verification failed"
                    admitted_flag  = False
                    rejected      += 1
                    result_str     = "REJECTED"
            else:
                failure_reason = "duplicate option_id in cert_store"
                admitted_flag  = False
                rejected      += 1
                result_str     = "REJECTED"
        else:
            gate_type = None
            epsilon   = 0.0
            margin    = cds_gate.get_admission_margin(mean_dr, mean_dn)
            worst     = float(mean_dr) + float(np.min(mean_dn))
            failure_reason = (
                f"dr + min(dn) = {worst:.4f} < -e={PDS_EPSILON} (fails PDS)"
            )
            rejected  += 1
            result_str = "REJECTED"

        # Invariant: cert_store and library must always be in sync
        assert cert_store.count() == library.count(), (
            f"SYNC ERROR after '{option_id}': "
            f"cert_store={cert_store.count()} != library={library.count()}"
        )

        print(
            f"{option_id:<28}  {mean_dr:>8.4f}  {float(np.min(mean_dn)):>9.4f}  "
            f"{'Y' if admitted_cds else 'N':>4}  {'Y' if admitted_pds else 'N':>4}  "
            f"{result_str:<18}  {library.count():>4}"
        )

        record: dict = {
            "skill_id":         option_id,
            "candidate_policy": option_id,
            "admitted":         admitted_flag,
            "gate_type":        gate_type if admitted_flag else None,
            "delta_r":          float(mean_dr),
            "delta_n":          tuple(float(v) for v in mean_dn),
            "margin":           float(margin),
            "epsilon":          float(epsilon),
            "failure_reason":   failure_reason,
            "seeds":            list(SEEDS),
            "n_pairs":          len(pairs),
        }
        episode_records.append(record)
        report.add_from_dict(record)

    # ── 3. Persist artifacts ───────────────────────────────────────────────────
    print()
    cert_store.save_to_file(str(CERT_FILE))
    library.save(str(LIBRARY_FILE))
    print(f"[Save] certificates -> {CERT_FILE}")
    print(f"[Save] library      -> {LIBRARY_FILE}")

    report.save_json(str(REPORT_JSON))
    report.save_markdown(str(REPORT_MD))
    print(f"[Save] report JSON  -> {REPORT_JSON}")
    print(f"[Save] report MD    -> {REPORT_MD}")

    # ── 4. Reload verification — artifacts must round-trip ─────────────────────
    print("\n[Verify] Reloading artifacts from disk...")

    reload_store = CertificateStore()
    reload_store.load_from_file(str(CERT_FILE))
    assert reload_store.count() == cert_store.count(), (
        f"Cert reload count mismatch: "
        f"got {reload_store.count()}, expected {cert_store.count()}"
    )

    lib_raw    = json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
    # SkillLibrary.save() stores skills under "skills" or "entries" key
    lib_skills = lib_raw.get("skills", lib_raw.get("entries", []))
    assert len(lib_skills) == library.count(), (
        f"Library reload count mismatch: "
        f"got {len(lib_skills)}, expected {library.count()}"
    )

    report_raw = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    assert report_raw["admitted"] == admitted, (
        f"Report admitted mismatch: got {report_raw['admitted']}, expected {admitted}"
    )

    print(f"[Verify] {reload_store.count()} certificates reloaded OK")
    print(f"[Verify] {library.count()} library entries reloaded OK")
    print("[Verify] Admission report JSON valid OK")

    # ── 5. Exit-gate assertions ────────────────────────────────────────────────
    assert admitted >= 1, (
        "EXIT GATE FAILED: at least one option must be admitted to the library"
    )
    assert rejected >= 1, (
        "EXIT GATE FAILED: at least one option must be rejected"
    )

    # ── 6. Summary ─────────────────────────────────────────────────────────────
    total = admitted + rejected
    print("\n" + "=" * 68)
    print("  Pipeline Summary")
    print("=" * 68)
    print(f"  Options evaluated : {total}")
    print(f"  Admitted (CDS)    : {cds_count}")
    print(f"  Admitted (PDS)    : {pds_count}")
    print(f"  Rejected          : {rejected}")
    print(f"  Library size      : {library.count()}")
    print(f"  Weight region     : FULL_SIMPLEX (6-D)")
    print(f"  Domain            : {DOMAIN} v{SCHEMA_VER}")
    print(f"  Safety guarantee  : Zero rejected options entered the library")
    print("=" * 68 + "\n")

    return {
        "admitted":        admitted,
        "rejected":        rejected,
        "cds_count":       cds_count,
        "pds_count":       pds_count,
        "library_size":    library.count(),
        "cert_file":       str(CERT_FILE),
        "library_file":    str(LIBRARY_FILE),
        "report_json":     str(REPORT_JSON),
        "report_md":       str(REPORT_MD),
        "episode_records": episode_records,
        "domain":          DOMAIN,
        "schema_version":  SCHEMA_VER,
        "motive_names":    list(MOTIVE_NAMES),
    }


if __name__ == "__main__":
    run_pipeline()

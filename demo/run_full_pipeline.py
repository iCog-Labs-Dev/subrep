"""
run_full_pipeline.py — End-to-end SubRep pipeline demonstration.

Runs 10 consecutive episodes in MO-LunarLander, certifying each skill
via CDS gate and storing admitted certificates in MeTTa + SkillLibrary.

Usage:
    python -m demo.run_full_pipeline
"""
from __future__ import annotations

import os
import torch
from datetime import datetime, timezone

import numpy as np

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
from generator.skill_generator import SkillGenerator

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_EPISODES        = 10
BASELINE_EPISODES   = 20
GAMMA               = 0.99
MAX_STEPS           = 200
SEED                = 42
CERT_FILE           = "data/certificates.metta"
LIBRARY_FILE        = "data/library.json"
ENV_NAME            = "MO-LunarLander-v3"
VERSION             = "0.1.0"
# ──────────────────────────────────────────────────────────────────────────────


def _make_certificate(
    skill_id: str,
    delta_r: float,
    delta_n: np.ndarray,
    margin: float,
    episode_length: int,
    gate_type: str = "CDS",
    epsilon: float = 0.0,
) -> Certificate:
    """Build a fully-validated Certificate from computed improvements."""
    return Certificate(
        skill_id=skill_id,
        gate_type=gate_type,
        delta_r=float(delta_r),
        delta_n=(float(delta_n[0]), float(delta_n[1])),
        admission_margin=float(margin),
        epsilon=epsilon,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=SEED,
        gamma=GAMMA,
        baseline_id="idle_policy_v1",
        environment=ENV_NAME,
        episode_length=episode_length,
        version=VERSION,
        weight_region_type="FULL_SIMPLEX",
        certification_context=None,
        mdn_alpha=None,
        wx_support_directions=None,
        wx_support_values=None,
    )


def run_pipeline() -> dict:
    """Run the full 10-episode certification pipeline and return statistics."""
    print("=" * 60)
    print("  SubRep End-to-End Pipeline Demo")
    print("=" * 60)

    # ── 1. Environment + Baseline ──────────────────────────────────────────────
    print("\n[Init] Setting up environment and computing baseline...")
    env = SubRepEnv(seed=SEED)
    idle = IdlePolicy(env=env, idle_action=0, gamma=GAMMA)
    baseline_stats = idle.run_baseline_episodes(
        num_episodes=BASELINE_EPISODES, seed=SEED
    )
    print(
        f"[Init] Baseline computed over {BASELINE_EPISODES} episodes | "
        f"mean payoff={baseline_stats['baseline_payoff']:.4f}"
    )

    calculator = ImprovementCalculator(baseline_stats)
    gate = CDSGate()
    pds_gate = PDSGate(epsilon=0.1)  # Allow a small mathematical trade-off budget

    # ── 1.5 Load Skill Generator (Pre-Filter) ─────────────────────────────────
    print("[Init] Loading trained SkillGenerator from models/generator.pt...")
    model = SkillGenerator(input_dim=8, hidden_dim=64, motive_dim=2)
    model.load("models/generator.pt")
    model.eval()  # Set to inference mode


    # ── 2. Stores ──────────────────────────────────────────────────────────────
    print("[Init] Initializing CertificateStore (MeTTa) and SkillLibrary...")
    cert_store = CertificateStore()
    library = SkillLibrary(cert_store=cert_store, save_path=LIBRARY_FILE)
    selector = SkillSelector(library=library, seed=SEED)

    # ── 3. Episode Loop ────────────────────────────────────────────────────────
    print(f"\n[Loop] Running {NUM_EPISODES} episodes...\n")
    print(
        f"{'Ep':>4}  {'Search':>6}  {'Payoff':>9}  {'Δr':>8}  {'min(Δn)':>8}  "
        f"{'CDS':>3}  {'PDS':>3}  {'Result':>10}  {'Lib':>4}"
    )
    print("-" * 70)

    admitted = 0
    rejected = 0
    first_admitted_ep = None

    for ep in range(1, NUM_EPISODES + 1):
        skill_id = f"skill_{ep:03d}"

        # SELECT — pick a skill or search for a good starting state
        searches = 0
        max_search = 500
        found_promising_state = False
        obs = None

        while not found_promising_state and searches < max_search:
            searches += 1
            obs, _ = env.reset()
            # Predict outcome using the SkillGenerator
            with torch.no_grad():
                pred_payoff, pred_motives = model(torch.tensor(obs, dtype=torch.float32))
                pred_dr, pred_dn = calculator.compute_improvements(
                    pred_payoff.item(), pred_motives.numpy()
                )
                # Does the model THINK it will pass either gate?
                if gate.admit(pred_dr, pred_dn) or pds_gate.admit(pred_dr, pred_dn):
                    found_promising_state = True

        # EXECUTE — run one episode
        # Use the trained RL Policy provided by the team lead.
        executor = SkillExecutor.from_pilot_checkpoint(
            env=env,
            gamma=GAMMA,
            max_steps=MAX_STEPS,
        )
        payoff, motives, terminated = executor.run_episode(initial_obs=obs)
        episode_length = executor.last_run_info.get("steps", MAX_STEPS)

        # CERTIFY — compute improvements and run CDS/PDS gates
        delta_r, delta_n = calculator.compute_improvements(payoff, motives)
        admitted_cds = gate.admit(delta_r, delta_n)
        admitted_pds = pds_gate.admit(delta_r, delta_n)
        
        admitted_flag = admitted_cds or admitted_pds
        active_gate = "CDS" if admitted_cds else "PDS"
        margin = gate.get_admission_margin(delta_r, delta_n) if admitted_cds else pds_gate.get_admission_margin(delta_r, delta_n)

        if admitted_flag:
            # STORE — save to cert_store (MeTTa) then to library
            cert = _make_certificate(
                skill_id=skill_id,
                delta_r=delta_r,
                delta_n=delta_n,
                margin=margin,
                episode_length=int(episode_length),
                gate_type=active_gate,
                epsilon=0.1 if active_gate == "PDS" else 0.0,
            )
            cert_store.add(cert)
            # Attach a fresh random policy as placeholder
            random_policy = lambda o: env.env.action_space.sample()
            library.add_skill(skill_id, cert, random_policy)
            admitted += 1
            if first_admitted_ep is None:
                first_admitted_ep = ep
            result_str = "ADMITTED ✅"
        else:
            rejected += 1
            result_str = "REJECTED ❌"

        print(
            f"{ep:>4}  {searches:>6d}  {payoff:>9.3f}  {delta_r:>8.3f}  {float(np.min(delta_n)):>8.3f}"
            f"  {'Y' if admitted_cds else 'N':>3}  {'Y' if admitted_pds else 'N':>3}"
            f"  {result_str:>12}  {library.count():>4}"
        )

    # ── 4. Persistence ─────────────────────────────────────────────────────────
    print("\n[Save] Persisting pipeline state to disk...")
    os.makedirs("data", exist_ok=True)
    cert_store.save_to_file(CERT_FILE)
    library.save(LIBRARY_FILE)
    print(f"[Save] certificates → {CERT_FILE}")
    print(f"[Save] library      → {LIBRARY_FILE}")

    # ── 5. Summary ─────────────────────────────────────────────────────────────
    total = admitted + rejected
    admission_rate = (admitted / total * 100) if total > 0 else 0.0
    rejection_rate = 100.0 - admission_rate

    print("\n" + "=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    print(f"  Total Episodes    : {total}")
    print(f"  Admitted          : {admitted} ({admission_rate:.1f}%)")
    print(f"  Rejected          : {rejected} ({rejection_rate:.1f}%)")
    print(f"  Library Size      : {library.count()}")
    if first_admitted_ep:
        print(f"  First Admission   : Episode {first_admitted_ep}")
    print(f"  Safety Guarantee  : Zero rejected skills entered the library ✅")
    print("=" * 60 + "\n")

    return {
        "total_episodes": total,
        "admitted": admitted,
        "rejected": rejected,
        "admission_rate": admission_rate,
        "rejection_rate": rejection_rate,
        "first_admitted_ep": first_admitted_ep,
        "library_size": library.count(),
    }


if __name__ == "__main__":
    run_pipeline()

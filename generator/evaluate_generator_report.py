
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baseline.idle_policy import IdlePolicy
from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from env.lunar_lander_wrapper import SubRepEnv
from generator.skill_generator import SkillGenerator


def compute_idle_baseline_stats(episodes: int = 20, seed: int = 42, gamma: float = 0.99) -> dict[str, Any]:
    """Reuse the existing IdlePolicy baseline (same approach as train_mdn_candidate_sets.py)."""
    env = SubRepEnv(seed=seed)
    return IdlePolicy(env=env, gamma=gamma).run_baseline_episodes(num_episodes=episodes, seed=seed)


def load_candidate_set_files(eval_dir: str, pattern: str = "*.npz") -> list[dict]:
    paths = sorted(glob.glob(os.path.join(eval_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No candidate-set .npz files found in {eval_dir}. "
            f"Collect eval data first, e.g.:\n"
            f"  python -m data_collector.collect_candidate_sets "
            f"--contexts 1000 --save-dir {eval_dir} --seed 100 --prefix seed100"
        )
    records = []
    for path in paths:
        npz = np.load(path, allow_pickle=True)
        records.append({
            "context": npz["context"],
            "context_seed": int(npz["context_seed"]),
            "candidate_skill_ids": [str(s) for s in npz["candidate_skill_ids"]],
            "candidate_payoffs": npz["candidate_payoffs"],
            "candidate_motives": npz["candidate_motives"],
            "source_file": os.path.basename(path),
        })
    return records


def certify_candidate(
    payoff: float,
    motives: np.ndarray,
    calculator: ImprovementCalculator,
    cds_gate: CDSGate,
    pds_gate: PDSGate,
) -> dict[str, Any]:
    
    delta_r, delta_n = calculator.compute_improvements(payoff, motives)
    cds_admit = cds_gate.admit(delta_r, delta_n)
    cds_margin = cds_gate.get_admission_margin(delta_r, delta_n)

    try:
        pds_admit = pds_gate.admit(delta_r, delta_n)
    except Exception:
        # PDS may require extra context (e.g. a weight-set/support region)
        # not available for every candidate; treat as "not evaluated" rather
        # than silently failing the whole report.
        pds_admit = None

    admitted = bool(cds_admit) and (pds_admit is not False)
    if admitted:
        reason = "admitted"
    elif not cds_admit:
        reason = f"rejected_by_CDS (margin={cds_margin:.4f}, worst-case motive coordinate dominates payoff gain)"
    else:
        reason = "rejected_by_PDS"

    return {
        "delta_r": delta_r,
        "delta_n": delta_n.tolist(),
        "cds_admit": bool(cds_admit),
        "cds_margin": cds_margin,
        "pds_admit": pds_admit,
        "admitted": admitted,
        "reason": reason,
    }


def build_report(
    model: SkillGenerator,
    records: list[dict],
    baseline_stats: dict[str, Any],
) -> dict[str, Any]:
   
    calculator = ImprovementCalculator(baseline_stats)
    cds_gate = CDSGate()
    pds_gate = PDSGate()

    per_skill: dict[str, dict[str, Any]] = {}
    generator_predicted_payoffs = []
    generator_actual_payoffs = []  # actual ppo_deterministic payoff, for correlation check

    for record in records:
        context = torch.tensor(record["context"], dtype=torch.float32)
        with torch.no_grad():
            pred_payoff, pred_motives = model(context)
        pred_payoff = float(pred_payoff.item())

        skill_ids = record["candidate_skill_ids"]
        payoffs = record["candidate_payoffs"]
        motives = record["candidate_motives"]

        for skill_id, payoff, motive in zip(skill_ids, payoffs, motives):
            result = certify_candidate(float(payoff), np.asarray(motive), calculator, cds_gate, pds_gate)
            bucket = per_skill.setdefault(skill_id, {
                "n": 0, "n_admitted": 0, "delta_r_sum": 0.0,
                "delta_n_sum": np.zeros_like(np.asarray(motive), dtype=np.float64),
                "rejection_reasons": {},
            })
            bucket["n"] += 1
            bucket["n_admitted"] += int(result["admitted"])
            bucket["delta_r_sum"] += result["delta_r"]
            bucket["delta_n_sum"] += np.asarray(result["delta_n"])
            if not result["admitted"]:
                bucket["rejection_reasons"][result["reason"]] = bucket["rejection_reasons"].get(result["reason"], 0) + 1

            if skill_id == "ppo_deterministic":
                generator_predicted_payoffs.append(pred_payoff)
                generator_actual_payoffs.append(float(payoff))

    # Summarize per-skill stats
    per_skill_summary = {}
    for skill_id, bucket in per_skill.items():
        n = bucket["n"]
        per_skill_summary[skill_id] = {
            "n_contexts": n,
            "success_rate": bucket["n_admitted"] / n,               # candidate skill success rate
            "admission_rate": bucket["n_admitted"] / n,             # certification admission rate after CDS/PDS
            "rejection_rate": 1.0 - (bucket["n_admitted"] / n),
            "rejection_reasons": bucket["rejection_reasons"],
            "avg_payoff_improvement_over_baseline": bucket["delta_r_sum"] / n,   # avg reward/payoff improvement
            "avg_motive_improvement_over_baseline": (bucket["delta_n_sum"] / n).tolist(),  # motive-feature improvement
        }

    correlation = None
    if len(generator_predicted_payoffs) >= 2:
        correlation = float(np.corrcoef(generator_predicted_payoffs, generator_actual_payoffs)[0, 1])

    # Baseline comparison: does the neural-pre-filtered skill (ppo_deterministic)
    # actually beat the simple non-neural baselines (random, fixed policies)?
    comparison = {}
    if "ppo_deterministic" in per_skill_summary:
        target = per_skill_summary["ppo_deterministic"]
        for baseline_id in ("random", "noop", "left_engine", "main_engine", "right_engine"):
            if baseline_id in per_skill_summary:
                base = per_skill_summary[baseline_id]
                comparison[baseline_id] = {
                    "ppo_deterministic_success_rate": target["success_rate"],
                    f"{baseline_id}_success_rate": base["success_rate"],
                    "ppo_deterministic_avg_payoff_improvement": target["avg_payoff_improvement_over_baseline"],
                    f"{baseline_id}_avg_payoff_improvement": base["avg_payoff_improvement_over_baseline"],
                }

    return {
        "n_contexts_evaluated": len(records),
        "context_seeds_used": sorted({r["context_seed"] for r in records}),
        "per_skill": per_skill_summary,
        "generator_predicted_vs_actual_payoff_correlation": correlation,
        "baseline_comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Certification-focused evaluation report for SkillGenerator.")
    parser.add_argument("--model-path", type=str, default="models/generator.pt")
    parser.add_argument("--eval-dir", type=str, default="data/mdn_candidate_sets_eval",
                         help="Candidate-set directory collected from seeds NOT used in generator training.")
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="demo/artifacts/generator_evaluation_report.json")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at '{args.model_path}'. Run generator.train_generator first.")
        return

    model = SkillGenerator(input_dim=8, hidden_dim=64, motive_dim=2)
    model.load(args.model_path)
    model.eval()

    try:
        records = load_candidate_set_files(args.eval_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loaded {len(records)} held-out contexts from {args.eval_dir} "
          f"(seeds: {sorted({r['context_seed'] for r in records})})")

    baseline_stats = compute_idle_baseline_stats(episodes=args.baseline_episodes, seed=args.baseline_seed)
    report = build_report(model, records, baseline_stats)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("  Generator Evaluation Report (held-out seeds)")
    print("=" * 60)
    print(f"Contexts evaluated : {report['n_contexts_evaluated']}")
    print(f"Seeds              : {report['context_seeds_used']}")
    print(f"Predicted-vs-actual payoff correlation (ppo_deterministic): "
          f"{report['generator_predicted_vs_actual_payoff_correlation']}")
    print("\nPer-skill certification summary:")
    for skill_id, stats in report["per_skill"].items():
        print(f"  {skill_id:20s} success_rate={stats['success_rate']:.2%} "
              f"avg_payoff_gain={stats['avg_payoff_improvement_over_baseline']:.3f}")
    print(f"\nFull report saved -> {out_path}")


if __name__ == "__main__":
    main()
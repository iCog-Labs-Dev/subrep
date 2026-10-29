"""
Full certification-focused evaluation report for the SkillGenerator.

"""

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
from generator.skill_generator import SkillGenerator


def compute_idle_baseline_stats(episodes: int = 20, seed: int = 42, gamma: float = 0.99) -> dict[str, Any]:
    """Reuse the existing IdlePolicy baseline """
    from env.lunar_lander_wrapper import SubRepEnv
    env = SubRepEnv(seed=seed)
    return IdlePolicy(env=env, gamma=gamma).run_baseline_episodes(num_episodes=episodes, seed=seed)


def load_candidate_set_files(eval_dir: str, pattern: str = "*.npz") -> list[dict]:
    """
    Load candidate-set .npz files directly...).
    """
    paths = sorted(glob.glob(os.path.join(eval_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No candidate-set .npz files found in {eval_dir}. "
            f"Collect eval data first, e.g.:\n"
            f"  python -m data_collector.collect_candidate_sets "
            f"--contexts 1000 --save-dir {eval_dir} --seed 1000 --prefix seed1000"
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


def load_training_context_seeds(training_data_dir: str) -> set[int]:
    """Load all training seeds, refusing datasets whose seeds cannot be verified."""
    paths = sorted(glob.glob(os.path.join(training_data_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No training .npz files found in {training_data_dir}.")

    seeds = set()
    missing_seed_files = []
    for path in paths:
        with np.load(path, allow_pickle=True) as saved_record:
            if "context_seed" not in saved_record.files:
                missing_seed_files.append(os.path.basename(path))
            else:
                seeds.add(int(saved_record["context_seed"]))

    if missing_seed_files:
        examples = ", ".join(missing_seed_files[:5])
        remainder = len(missing_seed_files) - 5
        if remainder > 0:
            examples += f", and {remainder} more"
        raise ValueError(
            "Cannot verify evaluation seeds because "
            f"{len(missing_seed_files)} training .npz file(s) lack context_seed ({examples}). "
            "Collect a fresh training dataset with context_seed metadata before evaluation."
        )
    return seeds


def validate_eval_seeds_disjoint(records: list[dict], training_data_dir: str) -> None:
    """Raise when supplementary evaluation seeds occur in the training dataset."""
    training_seeds = load_training_context_seeds(training_data_dir)
    evaluation_seeds = {int(record["context_seed"]) for record in records}
    overlapping_seeds = training_seeds & evaluation_seeds
    if overlapping_seeds:
        examples = ", ".join(str(seed) for seed in sorted(overlapping_seeds)[:10])
        remainder = len(overlapping_seeds) - 10
        if remainder > 0:
            examples += f", and {remainder} more"
        raise ValueError(
            "Supplementary evaluation seeds overlap the training dataset "
            f"({len(overlapping_seeds)} seeds: {examples}). Collect evaluation data "
            "with seeds not present in the training dataset, then rerun the report."
        )


def _format_seed_range(seeds: set[int]) -> str:
    """Compact 'min-max' display, not a full listing of every value."""
    if not seeds:
        return "none"
    lo, hi = min(seeds), max(seeds)
    return f"{lo}-{hi}" if lo != hi else str(lo)


def certify_candidate(
    payoff: float,
    motives: np.ndarray,
    calculator: ImprovementCalculator,
    cds_gate: CDSGate,
    pds_gate: PDSGate,
) -> dict[str, Any]:
    """
    Run one candidate's real recorded outcome through CDS and PDS.
    """
    delta_r, delta_n = calculator.compute_improvements(payoff, motives)

    cds_admit = bool(cds_gate.admit(delta_r, delta_n))
    cds_margin = cds_gate.get_admission_margin(delta_r, delta_n)

    pds_admit = bool(pds_gate.admit(delta_r, delta_n))

    return {
        "delta_r": delta_r,
        "delta_n": delta_n.tolist(),
        "cds_admit": cds_admit,
        "cds_margin": cds_margin,
        "pds_admit": pds_admit,
        "pds_reason": "admitted" if pds_admit else "rejected_by_PDS",
    }


def build_report(
    model: SkillGenerator,
    records: list[dict],
    baseline_stats: dict[str, Any],
) -> dict[str, Any]:
    """
   Track cds_admission_rate and pds_admission_rate per skill --
      
    """
    calculator = ImprovementCalculator(baseline_stats)
    cds_gate = CDSGate()
    pds_gate = PDSGate()

    per_skill: dict[str, dict[str, Any]] = {}
    generator_predicted_payoffs = []
    generator_actual_payoffs = []
    generator_predicted_motives = []  
    generator_actual_motives = []

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
                "n": 0,
                "n_cds_admitted": 0,
                "n_pds_admitted": 0,
                "delta_r_sum": 0.0,
                "delta_n_sum": np.zeros_like(np.asarray(motive), dtype=np.float64),
                "rejection_reasons": {},
            })
            bucket["n"] += 1
            bucket["n_cds_admitted"] += int(result["cds_admit"])
            bucket["n_pds_admitted"] += int(result["pds_admit"])
            bucket["delta_r_sum"] += result["delta_r"]
            bucket["delta_n_sum"] += np.asarray(result["delta_n"])
            if not result["pds_admit"]:
                bucket["rejection_reasons"][result["pds_reason"]] = (
                    bucket["rejection_reasons"].get(result["pds_reason"], 0) + 1
                )
            if skill_id == "ppo_deterministic":
               generator_predicted_payoffs.append(pred_payoff)
               generator_actual_payoffs.append(float(payoff))
               generator_predicted_motives.append(pred_motives.tolist())   
               generator_actual_motives.append(np.asarray(motive).tolist())  
    per_skill_summary = {}
    for skill_id, bucket in per_skill.items():
        n = bucket["n"]
        per_skill_summary[skill_id] = {
            "n_contexts": n,
            "cds_admission_rate": bucket["n_cds_admitted"] / n,
            "pds_admission_rate": bucket["n_pds_admitted"] / n,
            "rejection_rate": 1.0 - (bucket["n_pds_admitted"] / n),
            "rejection_reasons": bucket["rejection_reasons"],
            "avg_payoff_improvement_over_baseline": bucket["delta_r_sum"] / n,
            "avg_motive_improvement_over_baseline": (bucket["delta_n_sum"] / n).tolist(),
        }

    correlation = None
    if len(generator_predicted_payoffs) >= 2:
        correlation = float(np.corrcoef(generator_predicted_payoffs, generator_actual_payoffs)[0, 1])
    generator_avg_predicted_payoff = None
    generator_avg_actual_payoff = None
    generator_avg_predicted_motives = None
    generator_avg_actual_motives = None
    if generator_predicted_payoffs:
            generator_avg_predicted_payoff = float(np.mean(generator_predicted_payoffs))
            generator_avg_actual_payoff = float(np.mean(generator_actual_payoffs))
            generator_avg_predicted_motives = np.mean(generator_predicted_motives, axis=0).tolist()
            generator_avg_actual_motives = np.mean(generator_actual_motives, axis=0).tolist()

    comparison = {}
    if "ppo_deterministic" in per_skill_summary:
        target = per_skill_summary["ppo_deterministic"]
        for baseline_id in ("random", "noop", "left_engine", "main_engine", "right_engine"):
            if baseline_id in per_skill_summary:
                base = per_skill_summary[baseline_id]
                comparison[baseline_id] = {
                    "ppo_deterministic_cds_admission_rate": target["cds_admission_rate"],
                    f"{baseline_id}_cds_admission_rate": base["cds_admission_rate"],
                    "ppo_deterministic_pds_admission_rate": target["pds_admission_rate"],
                    f"{baseline_id}_pds_admission_rate": base["pds_admission_rate"],
                    "ppo_deterministic_avg_payoff_improvement": target["avg_payoff_improvement_over_baseline"],
                    f"{baseline_id}_avg_payoff_improvement": base["avg_payoff_improvement_over_baseline"],
                }

    unique_seeds = {r["context_seed"] for r in records}
    return {
        "n_contexts_evaluated": len(records),
        "n_unique_context_seeds": len(unique_seeds),
        "context_seed_range": _format_seed_range(unique_seeds),
        "per_skill": per_skill_summary,
        "generator_predicted_vs_actual_payoff_correlation": correlation,
        "generator_avg_predicted_payoff": generator_avg_predicted_payoff,   
        "generator_avg_actual_payoff": generator_avg_actual_payoff,         
        "generator_avg_predicted_motives": generator_avg_predicted_motives, # [Safety, Fuel]
        "generator_avg_actual_motives": generator_avg_actual_motives, 
        "baseline_comparison": comparison,
    }


def build_multi_policy_report(test_rows, train_rows, model, observation_baseline, baseline_stats):
    """Held-out raw_mixed report; predictions never determine certification."""
    from generator.multi_policy_data import group_contexts
    from generator.multi_policy_pipeline import evaluate

    if not test_rows or not train_rows:
        raise ValueError("Multi-policy reporting requires nonempty train and test splits")
    groups = group_contexts(test_rows, complete=True)
    prediction = evaluate(test_rows, train_rows, {
        "multi_policy": (model, True),
        "observation_only": (observation_baseline, False),
    })
    calculator = ImprovementCalculator(baseline_stats)
    cds_gate, pds_gate = CDSGate(), PDSGate()
    per_policy = {}
    for context_rows in groups.values():
        for row in context_rows:
            measured = certify_candidate(row["payoff"], row["motives"],
                                         calculator, cds_gate, pds_gate)
            bucket = per_policy.setdefault(row["policy"], {
                "executed": 0, "cds_admitted": 0, "pds_admitted": 0,
                "certifiable": 0, "payoff_sum": 0.0,
                "motive_sum": np.zeros(2, dtype=np.float64),
                "payoff_improvement_sum": 0.0,
                "motive_improvement_sum": np.zeros(2, dtype=np.float64),
            })
            bucket["executed"] += 1
            bucket["cds_admitted"] += int(measured["cds_admit"])
            bucket["pds_admitted"] += int(measured["pds_admit"])
            bucket["certifiable"] += int(measured["cds_admit"] and measured["pds_admit"])
            bucket["payoff_sum"] += float(row["payoff"])
            bucket["motive_sum"] += np.asarray(row["motives"], dtype=np.float64)
            bucket["payoff_improvement_sum"] += float(measured["delta_r"])
            bucket["motive_improvement_sum"] += np.asarray(measured["delta_n"])
    certification = {}
    for policy, b in per_policy.items():
        n = b["executed"]
        certification[policy] = {
            "executed": n,
            "cds_admission_rate": b["cds_admitted"] / n,
            "cds_rejection_rate": 1 - b["cds_admitted"] / n,
            "pds_admission_rate": b["pds_admitted"] / n,
            "pds_rejection_rate": 1 - b["pds_admitted"] / n,
            "joint_certifiable_count": b["certifiable"],
            "measured_payoff_mean": b["payoff_sum"] / n,
            "measured_motives_mean": (b["motive_sum"] / n).tolist(),
            "measured_payoff_improvement_mean": b["payoff_improvement_sum"] / n,
            "measured_motive_improvement_mean": (b["motive_improvement_sum"] / n).tolist(),
        }
    return {
        "evaluation_mode": "multi_policy_held_out_raw_mixed",
        "n_contexts": len(groups), "n_candidate_outcomes": len(test_rows),
        "prediction_metrics": prediction,
        "measured_certification_per_policy": certification,
    }


def run_multi_policy_report(args):
    """Use the persisted grouped split, never a supplementary MDN dataset."""
    from generator.multi_policy_data import load_records, load_or_create_manifest, partition
    from generator.multi_policy_pipeline import load_model

    rows = load_records(args.data_dir)
    manifest = load_or_create_manifest(rows, args.split_manifest, seed=args.seed)
    parts = partition(rows, manifest)
    model, conditioned = load_model(args.model_path)
    baseline, baseline_conditioned = load_model(args.observation_baseline_path)
    if not conditioned or baseline_conditioned:
        raise ValueError("Expected conditioned model and observation-only baseline checkpoints")
    baseline_stats = compute_idle_baseline_stats(
        episodes=args.baseline_episodes, seed=args.baseline_seed)
    report = build_multi_policy_report(parts["test"], parts["train"],
                                       model, baseline, baseline_stats)
    report["split_manifest"] = str(args.split_manifest)
    report["model_path"] = str(args.model_path)
    report["observation_baseline_path"] = str(args.observation_baseline_path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Multi-policy held-out report saved -> {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Certification-focused evaluation report for SkillGenerator.")
    parser.add_argument("--model-path", type=str, default="models/generator.pt")
    parser.add_argument("--eval-dir", type=str, default="data/mdn_candidate_sets_eval",
                         help="Held-out candidate-set directory collected from seeds NOT used in generator training.")
    parser.add_argument("--training-data-dir", type=str, default="data/raw",
                        help="Training rollout directory used to verify evaluation seed separation.")
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="demo/artifacts/generator_evaluation_report.json")
    parser.add_argument("--multi-policy", action="store_true",
                        help="Evaluate held-out raw_mixed records with policy conditioning")
    parser.add_argument("--data-dir", default="data/raw_mixed")
    parser.add_argument("--split-manifest", default="data/mixed_split_manifest.json")
    parser.add_argument("--observation-baseline-path", default="models/observation_only_baseline.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.multi_policy:
        # Legacy CLI defaults must not silently load an observation-only checkpoint.
        if args.model_path == "models/generator.pt":
            args.model_path = "models/multi_policy_generator.pt"
        if args.output == "demo/artifacts/generator_evaluation_report.json":
            args.output = "reports/multi_policy/generator_evaluation_report.json"
        run_multi_policy_report(args)
        return

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at '{args.model_path}'. Run generator.train_generator first.")
        return

    model = SkillGenerator(input_dim=8, hidden_dim=64, motive_dim=2)
    model.load(args.model_path)
    model.eval()

    try:
        records = load_candidate_set_files(args.eval_dir)
        validate_eval_seeds_disjoint(records, args.training_data_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    unique_seeds = {r["context_seed"] for r in records}
    print(f"Loaded {len(records)} candidate-set files from {args.eval_dir} "
          f"({len(unique_seeds)} unique context seeds, range {_format_seed_range(unique_seeds)})")
    if len(unique_seeds) != len(records):
        print(f"  NOTE: {len(records)} files but only {len(unique_seeds)} unique context seeds -- "
              f"some contexts were collected more than once. See docs/GENERATOR_TRAINING_PIPELINE.md "
              f"for non-overlapping seed values to use on the next collection run.")

    baseline_stats = compute_idle_baseline_stats(episodes=args.baseline_episodes, seed=args.baseline_seed)
    report = build_report(model, records, baseline_stats)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("  Generator Evaluation Report (held-out seeds)")
    print("=" * 60)
    print(f"Contexts evaluated (files) : {report['n_contexts_evaluated']}")
    print(f"Unique context seeds       : {report['n_unique_context_seeds']} (range {report['context_seed_range']})")
    print(f"Predicted-vs-actual payoff correlation (ppo_deterministic): "
          f"{report['generator_predicted_vs_actual_payoff_correlation']}")
    print(f"Avg predicted payoff  : {report['generator_avg_predicted_payoff']:.3f}   "  
      f"Avg actual payoff  : {report['generator_avg_actual_payoff']:.3f}")          
    print(f"Avg predicted motives : {report['generator_avg_predicted_motives']}   "     
      f"Avg actual motives : {report['generator_avg_actual_motives']}")             
    print("\nPer-skill certification summary:")
    for skill_id, stats in report["per_skill"].items():
        print(f"  {skill_id:20s} CDS={stats['cds_admission_rate']:.2%}  "
              f"PDS={stats['pds_admission_rate']:.2%}  "
              f"avg_payoff_gain={stats['avg_payoff_improvement_over_baseline']:.3f}")
    print(f"\nFull report saved -> {out_path}")


if __name__ == "__main__":
    main()

"""Experimentally tune fixed Safety/Fuel weights on validation contexts at fixed top-k selection budget."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from generator.candidate_filter import (make_predicted_score_fn,select_top_k,)
from generator.evaluate_candidate_filter import assess
from generator.multi_policy_data import (
    group_contexts,
    load_manifest,
    load_records,
    partition,
)
from generator.multi_policy_pipeline import load_model
from generator.policy_registry import POLICY_NAMES
from generator.ranking_config import validate_motive_weights

def weight_grid(step: float = 0.1) -> list[tuple[float, float]]:
    """Build a deterministic simplex grid for Safety and Fuel weights."""
    if not np.isfinite(step) or not 0.0 < step <= 1.0:
        raise ValueError("grid step must be in (0, 1]")

    values = [
        round(float(value), 10)
        for value in np.arange(0.0, 1.0, step)
    ]
    values.append(1.0)
    values = sorted(set(min(1.0, value) for value in values))

    return [
        (safety, round(1.0 - safety, 10))
        for safety in values
    ]


def tune_weights(validation_groups,model,baseline_stats,fixed_top_k: int = 3,grid_step: float = 0.1,):
    """Choose weights by validation certification recall at fixed budget.

    Ties are broken by higher precision, weights closest to equal,
    and then lower Safety weight for deterministic selection.
    """
    if not validation_groups:
        raise ValueError("Validation split contains no contexts")

    first_rows = next(iter(validation_groups.values()))

    if not 1 <= fixed_top_k <= len(first_rows):
        raise ValueError(
            "fixed_top_k must be within the number of candidates per context"
        )

    candidates = []

    for safety_weight, fuel_weight in weight_grid(grid_step):
        weights = validate_motive_weights((safety_weight, fuel_weight))
        score_fn = make_predicted_score_fn( baseline_stats,weights,)
        selections = {}

        for context, rows in validation_groups.items():
            names = [row["policy"] for row in rows]
            selections[context] = select_top_k(model,rows[0]["obs"],names,fixed_top_k,
                score_fn=score_fn,
            )
        # Certification is assessed using measured validation outcomes.
        metrics = assess( validation_groups,selections,baseline_stats,)

        total_certifiable = int(metrics["genuinely_certifiable"])
        found = int(metrics["certifiable_found"])
        executed = int(metrics["executed"])

        if total_certifiable == 0:
            raise ValueError(
                "Validation split contains no certifiable candidates"
            )

        recall = found / total_certifiable
        precision = found / executed if executed else 0.0

        candidates.append({
            "safety_weight": weights[0],
            "fuel_weight": weights[1],
            "certifiable_found": found,
            "genuinely_certifiable": total_certifiable,
            "executed": executed,
            "recall": recall,
            "precision": precision,
        })

    best = max(
        candidates,
        key=lambda row: (
            row["recall"],
            row["precision"],
            -abs(row["safety_weight"] - 0.5),
            -row["safety_weight"],
        ),
    )

    return {
        "motive_weights": {
            "safety": best["safety_weight"],
            "fuel": best["fuel_weight"],
        },
        "fixed_top_k": int(fixed_top_k),
        "selection_objective": (
            "maximize_validation_certification_recall"
        ),
        "tie_breaking": [
            "higher_validation_precision",
            "weights_closest_to_equal",
            "lower_safety_weight",
        ],
        "validation_metrics": {
            "certifiable_found": best["certifiable_found"],
            "genuinely_certifiable": best["genuinely_certifiable"],
            "recall": best["recall"],
            "precision": best["precision"],
            "executed": best["executed"],
        },
        "grid_step": float(grid_step),
        "grid_results": candidates,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/raw_mixed")
    parser.add_argument(
        "--split-manifest",
        default="data/mixed_split_manifest.json",
    )
    parser.add_argument(
        "--model-path",
        default="models/multi_policy_generator.pt",
    )
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--fixed-top-k", type=int, default=3)
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="reports/multi_policy/motive_weights.json",
    )

    args = parser.parse_args(argv)

    if args.baseline_episodes <= 0:
        parser.error("--baseline-episodes must be positive")

    if not 1 <= args.fixed_top_k <= len(POLICY_NAMES):
        parser.error("--fixed-top-k is outside the candidate range")

    records = load_records(args.data_dir)
    manifest = load_manifest(records, args.split_manifest)

    validation_records = partition(records, manifest)["val"]
    validation_groups = group_contexts(
        validation_records,
        complete=True,
    )

    model, conditioned = load_model(
        args.model_path,
        expected_manifest=manifest,
    )

    if not conditioned:
        raise ValueError(
            "Motive-weight tuning requires a policy-conditioned model"
        )

    from generator.evaluate_generator_report import (
        compute_idle_baseline_stats,
    )

    baseline_stats = compute_idle_baseline_stats(
        episodes=args.baseline_episodes,
        seed=args.seed,
    )

    tuned = tune_weights(
        validation_groups,
        model,
        baseline_stats,
        fixed_top_k=args.fixed_top_k,
        grid_step=args.grid_step,
    )

    tuned.update({
        "split_signature": manifest["split_signature"],
        "dataset_signature": manifest["dataset_signature"],
        "baseline_episodes": args.baseline_episodes,
        "baseline_seed": args.seed,
        "model_path": str(args.model_path),
    })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(tuned, indent=2),
        encoding="utf-8",
    )

    print(
        "Selected validation weights: "
        f"Safety={tuned['motive_weights']['safety']:.3f}, "
        f"Fuel={tuned['motive_weights']['fuel']:.3f}",
        flush=True,
    )
    print(
        "Validation recall: "
        f"{tuned['validation_metrics']['recall']:.3f}; "
        "precision: "
        f"{tuned['validation_metrics']['precision']:.3f}; "
        f"fixed top-k: {args.fixed_top_k}",
        flush=True,
    )
    print(f"Saved ranking configuration -> {output}", flush=True)


if __name__ == "__main__":
    main()

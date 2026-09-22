"""Train and evaluate the MDN support head with context-disjoint splits."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from generator.evaluate_mdn_candidate_sets import compute_idle_baseline_stats
from generator.evaluate_mdn_support import (
    evaluate_support_models,
    write_json_report,
    write_support_evaluation_report,
)
from generator.mdn_support_trainer import MDNSupportTrainer, SupportTrainerConfig
from utils.mdn_checkpoint_loader import load_mdn_checkpoint
from utils.mdn_data_adapter import candidate_set_directory_to_prepared_candidate_outcomes
from utils.mdn_support_data import runtime_logs_to_support_data, split_weight_set_store
from utils.weight_set_store import WeightSetStore


def run_support_head_experiment(
    *,
    base_checkpoint_path: str | Path,
    weight_store_path: str | Path | None = None,
    runtime_log_dir: str | Path | None = None,
    runtime_log_pattern: str = "*.npz",
    output_dir: str | Path = "data/mdn_support_evaluation",
    output_checkpoint_path: str | Path = "models/mdn_support_best.pth",
    candidate_data_dir: str | Path | None = None,
    candidate_pattern: str = "*.npz",
    baseline_stats: dict[str, Any] | None = None,
    baseline_episodes: int = 20,
    baseline_seed: int = 900_000,
    gamma: float = 0.99,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 1.0,
    max_epochs: int = 500,
    early_stopping_patience: int = 50,
    min_delta: float = 1e-6,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run support-only fitting, validation selection, and untouched test evaluation."""
    if (weight_store_path is None) == (runtime_log_dir is None):
        raise ValueError("Provide exactly one of weight_store_path or runtime_log_dir")

    logged_candidate_sets = None
    if runtime_log_dir is not None:
        source_store, logged_candidate_sets = runtime_logs_to_support_data(
            str(runtime_log_dir),
            pattern=runtime_log_pattern,
        )
        source_description = str(runtime_log_dir)
    else:
        source_store = WeightSetStore.load(weight_store_path or "")
        source_description = str(weight_store_path)
    model = load_mdn_checkpoint(base_checkpoint_path, map_location=device)
    if model.num_objectives != source_store.num_objectives:
        raise ValueError(
            "Base checkpoint objective count does not match the support store: "
            f"{model.num_objectives} != {source_store.num_objectives}"
        )
    context_dimensions = {
        int(context.shape[0]) for context, _ in source_store.get_all_context_vertices()
    }
    if context_dimensions != {model.input_dim}:
        raise ValueError(
            "Support-store context dimensions do not match the base checkpoint: "
            f"store={sorted(context_dimensions)}, model={model.input_dim}"
        )
    split = split_weight_set_store(
        source_store,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split.train.save(output_path / "support_train.json")
    split.validation.save(output_path / "support_validation.json")
    split.test.save(output_path / "support_test.json")
    write_json_report(split.manifest, output_path / "split_manifest.json")

    trainer = MDNSupportTrainer(
        model,
        split.train,
        config=SupportTrainerConfig(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_norm=gradient_clip_norm,
            min_contexts_to_train=1,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            min_delta=min_delta,
            checkpoint_path=str(output_checkpoint_path),
        ),
        device=device,
    )
    training_metrics = trainer.fit(split.validation)
    checkpoint_path = trainer.save_checkpoint(
        output_checkpoint_path,
        metadata={
            "target_semantics": "coordinate-wise maxima of observed certified-selection weights",
            "base_checkpoint_path": str(base_checkpoint_path),
            "source_support_data": source_description,
            "source_type": "runtime_logs" if runtime_log_dir is not None else "weight_store",
            "split_manifest": split.manifest,
            "training_metrics": training_metrics,
        },
    )

    outcomes = None
    if candidate_data_dir is not None:
        if logged_candidate_sets is not None:
            raise ValueError(
                "Runtime logs already provide candidate metrics; do not also provide candidate_data_dir"
            )
        outcomes = candidate_set_directory_to_prepared_candidate_outcomes(
            candidate_data_dir,
            pattern=candidate_pattern,
        )
        if baseline_stats is None:
            baseline_stats = compute_idle_baseline_stats(
                episodes=baseline_episodes,
                seed=baseline_seed,
                gamma=gamma,
            )

    train_targets = split.train.get_all_support_targets()
    train_mean_support = np.mean(
        np.stack([target for _, target in train_targets], axis=0), axis=0
    )
    validation_report = evaluate_support_models(
        model,
        split.validation,
        constant_baseline_support=train_mean_support,
        device=device,
    )
    test_report = evaluate_support_models(
        model,
        split.test,
        constant_baseline_support=train_mean_support,
        candidate_outcomes=outcomes,
        candidate_sets=logged_candidate_sets,
        baseline_stats=baseline_stats,
        device=device,
    )
    report = {
        "status": "completed",
        "checkpoint_path": checkpoint_path,
        "base_checkpoint_path": str(base_checkpoint_path),
        "source_support_data": source_description,
        "source_type": "runtime_logs" if runtime_log_dir is not None else "weight_store",
        "candidate_data_dir": None if candidate_data_dir is None else str(candidate_data_dir),
        "training": training_metrics,
        "split": split.manifest,
        "validation": validation_report,
        "test": test_report,
        "limitations": [
            "Targets are derived from selected weights that passed certification, not independent safe-region boundaries.",
            "Selection and certification bias remain in the observed target vertices.",
            "StubMDN and FULL_SIMPLEX use the same all-ones support geometry.",
            "Downstream admission and reuse metrics require candidate contexts matching the held-out support contexts.",
        ],
    }
    write_json_report(report, output_path / "support_experiment.json")
    write_support_evaluation_report(test_report, output_path / "support_test_report.md")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate the MDN support head with context-disjoint splits."
    )
    parser.add_argument("--base-checkpoint", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--weight-store")
    source_group.add_argument(
        "--runtime-log-dir",
        help="Validated probability-aware logs used to build targets and downstream metrics.",
    )
    parser.add_argument("--runtime-log-pattern", default="*.npz")
    parser.add_argument("--output-dir", default="data/mdn_support_evaluation")
    parser.add_argument("--output-checkpoint", default="models/mdn_support_best.pth")
    parser.add_argument("--candidate-data-dir")
    parser.add_argument("--candidate-pattern", default="*.npz")
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--baseline-seed", type=int, default=900_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_support_head_experiment(
        base_checkpoint_path=args.base_checkpoint,
        weight_store_path=args.weight_store,
        runtime_log_dir=args.runtime_log_dir,
        runtime_log_pattern=args.runtime_log_pattern,
        output_dir=args.output_dir,
        output_checkpoint_path=args.output_checkpoint,
        candidate_data_dir=args.candidate_data_dir,
        candidate_pattern=args.candidate_pattern,
        baseline_episodes=args.baseline_episodes,
        baseline_seed=args.baseline_seed,
        gamma=args.gamma,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        min_delta=args.min_delta,
        device=args.device,
    )
    print("MDN support-head experiment complete")
    print(f"checkpoint: {report['checkpoint_path']}")
    print(f"report: {Path(args.output_dir) / 'support_test_report.md'}")


if __name__ == "__main__":
    main()

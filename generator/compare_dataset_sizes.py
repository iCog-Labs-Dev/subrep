
"""Compare user-selected training sizes on shared validation and test sets."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from generator.losses import GeneratorLoss
from generator.skill_generator import SkillGenerator
from generator.train_generator import (
    InMemorySkillDataset,
    SkillDataset,
    evaluate_loss,
    train_one_epoch,
)


def create_holdout_split(
    full_dataset: SkillDataset,
    seed: int,
    source_size: int | None = None,
    validation_fraction: float = 0.125,
    test_fraction: float = 0.125,
):
    """
    Split the source pool once into a training pool and fixed holdouts.
    """
    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("validation_fraction and test_fraction must be positive")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.0")

    all_files = sorted(full_dataset.files)
    if not all_files:
        raise ValueError("Cannot split an empty dataset")

    file_to_record = dict(zip(full_dataset.files, full_dataset.data))
    context_groups: dict[tuple[float, ...], list[tuple[str, object]]] = {}
    for file_path in all_files:
        record = file_to_record[file_path]
        observation = record[0].detach().cpu().numpy().reshape(-1)
        context_key = tuple(float(value) for value in np.round(observation, decimals=6))
        context_groups.setdefault(context_key, []).append((file_path, record))

    rng = random.Random(seed)
    shuffled_groups = list(context_groups.values())
    rng.shuffle(shuffled_groups)

    available_size = len(all_files)
    requested_source_size = available_size if source_size is None else int(source_size)
    if requested_source_size <= 0:
        raise ValueError("source_size must be positive")
    if requested_source_size > available_size:
        raise ValueError(
            f"Requested source size {requested_source_size} exceeds available records ({available_size})"
        )

    source_groups = []
    selected_source_size = 0
    for group in shuffled_groups:
        if selected_source_size >= requested_source_size:
            break
        source_groups.append(group)
        selected_source_size += len(group)
    rng.shuffle(source_groups)
    shuffled_groups = source_groups

    selected_source_files = [entry[0] for group in shuffled_groups for entry in group]
    total_size = len(selected_source_files)
    total_contexts = len(shuffled_groups)
    validation_contexts = int(round(total_contexts * validation_fraction))
    test_contexts = int(round(total_contexts * test_fraction))
    if validation_contexts == 0 or test_contexts == 0:
        raise ValueError("Dataset is too small for non-empty validation and test sets")

    test_groups = shuffled_groups[:test_contexts]
    validation_groups = shuffled_groups[test_contexts:test_contexts + validation_contexts]
    training_pool_groups = shuffled_groups[test_contexts + validation_contexts:]
    if not training_pool_groups:
        raise ValueError("The fixed holdout leaves no records for training")

    def flatten_groups(groups):
        entries = [entry for group in groups for entry in group]
        return [entry[0] for entry in entries], [entry[1] for entry in entries]

    test_files, test_records = flatten_groups(test_groups)
    validation_files, validation_records = flatten_groups(validation_groups)
    training_pool_files = [
        entry[0] for group in training_pool_groups for entry in group
    ]
    return {
        "total_size": total_size, "available_size": available_size,
        "total_contexts": total_contexts,
        "training_pool_contexts": len(training_pool_groups),
        "validation_contexts": len(validation_groups), "test_contexts": len(test_groups),
        "training_pool_groups": training_pool_groups, "training_pool_files": training_pool_files,
        "validation_files": validation_files, "test_files": test_files,
        "validation_records": validation_records, "test_records": test_records,
        "validation_fraction": len(validation_files) / total_size,
        "test_fraction": len(test_files) / total_size,
    }


def create_dataset_split(
    holdout: dict,
    size: int,
):
    """Select a nested training subset from a precomputed holdout."""
    if size <= 0:
        raise ValueError("Training subset size must be positive")
    if size > len(holdout["training_pool_files"]):
        raise ValueError(
            f"Requested training size {size} exceeds the available training pool "
            f"({len(holdout['training_pool_files'])})."
        )

    train_entries = []
    train_count = 0
    for group in holdout["training_pool_groups"]:
        if train_count >= size:
            break
        train_entries.extend(group)
        train_count += len(group)

    return {
        "train_files": [entry[0] for entry in train_entries],
        "validation_files": holdout["validation_files"], "test_files": holdout["test_files"],
        "train_records": [entry[1] for entry in train_entries],
        "validation_records": holdout["validation_records"], "test_records": holdout["test_records"],
    }


def _copy_split_files(files: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for file_path in files:
        shutil.copy2(file_path, output_dir / os.path.basename(file_path))


def save_dataset_files(splits: dict[int, dict], output_dir: Path) -> None:
    """Save shared holdouts once and each requested training subset separately."""
    first_split = next(iter(splits.values()))
    _copy_split_files(first_split["test_files"], output_dir / "fixed_split" / "test")
    _copy_split_files(first_split["validation_files"], output_dir / "fixed_split" / "validation")
    for size, split in splits.items():
        _copy_split_files(split["train_files"], output_dir / f"size_{size}" / "train")


# ============================================================
# Training loop
# ============================================================

def run_training_loop(
    model: SkillGenerator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: GeneratorLoss,
    device: torch.device,
    num_epochs: int,
    patience: int,
):
    """
    Train the model using the training set.

    Validation loss is used for:
        - selecting the best model
        - early stopping
    """

    history_train = []
    history_val = []

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0

    for epoch in range(num_epochs):
        train_loss, _, _ = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        val_loss, _, _ = evaluate_loss(
            model,
            val_loader,
            loss_fn,
            device,
        )

        history_train.append(train_loss)
        history_val.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return (
        best_state,
        history_train,
        history_val,
        best_epoch,
    )


# ============================================================
# Test evaluation
# ============================================================

def evaluate_on_test_set(
    model: SkillGenerator,
    train_records: list,
    test_records: list,
    device: torch.device | None = None,
):
    """
    Compare model and training-mean predictions on the same fixed test set.
    """

    if device is None:
        first_parameter = next(model.parameters(), None)
        device = first_parameter.device if first_parameter is not None else torch.device("cpu")
    else:
        device = torch.device(device)
    model.to(device)

    loader = DataLoader(
        InMemorySkillDataset(test_records),
        batch_size=64,
        shuffle=False,
    )

    preds_payoff = []
    targets_payoff = []

    preds_motives = []
    targets_motives = []

    model.eval()
    with torch.no_grad():
        for obs, payoff, motives in loader:
            obs = obs.to(device)
            payoff = payoff.to(device)
            motives = motives.to(device)
            predicted_payoff, predicted_motives = model(obs)

            preds_payoff.append(predicted_payoff)
            targets_payoff.append(payoff)
            preds_motives.append(predicted_motives)
            targets_motives.append(motives)

    predicted_payoff = torch.cat(preds_payoff)
    target_payoff = torch.cat(targets_payoff)
    predicted_motives = torch.cat(preds_motives)
    target_motives = torch.cat(targets_motives)

    training_payoffs = torch.stack([record[1] for record in train_records]).to(device)
    training_motives = torch.stack([record[2] for record in train_records]).to(device)
    mean_payoff = training_payoffs.mean(dim=0).expand_as(target_payoff)
    mean_motives = training_motives.mean(dim=0).expand_as(target_motives)

    model_mse = {
        "payoff_mse": F.mse_loss(predicted_payoff, target_payoff).item(),
        "safety_mse": F.mse_loss(predicted_motives[:, 0], target_motives[:, 0]).item(),
        "fuel_mse": F.mse_loss(predicted_motives[:, 1], target_motives[:, 1]).item(),
    }
    mean_mse = {
        "payoff_mse": F.mse_loss(mean_payoff, target_payoff).item(),
        "safety_mse": F.mse_loss(mean_motives[:, 0], target_motives[:, 0]).item(),
        "fuel_mse": F.mse_loss(mean_motives[:, 1], target_motives[:, 1]).item(),
    }
    improvement = {key: mean_mse[key] - model_mse[key] for key in model_mse}
    improvement_percent = {
        key: 100.0 * improvement[key] / mean_mse[key] if mean_mse[key] > 0 else None
        for key in model_mse
    }

    return {
        "model_mse": model_mse,
        "training_mean_mse": mean_mse,
        "improvement_mse": improvement,
        "improvement_percent": improvement_percent,
        "model_beats_training_mean": {
            key: model_mse[key] < mean_mse[key] for key in model_mse
        },
    }


# ============================================================
# Training curves
# ============================================================

def save_combined_curves_plot(
    histories: list[dict],
    num_epochs: int,
    path: Path,
):
    """
    Save one figure containing the training/validation curves
    for every dataset size.
    """

    n = len(histories)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(5 * n, 4),
        sharey=True,
    )

    if n == 1:
        axes = [axes]

    for ax, history in zip(axes, histories):
        ax.plot(
            range(1, len(history["train"]) + 1),
            history["train"],
            label="Train",
        )

        ax.plot(
            range(1, len(history["val"]) + 1),
            history["val"],
            label="Validation",
        )

        ax.axvline(
            history["best_epoch"],
            linestyle="--",
            label="Best epoch",
        )

        ax.set_title(f"Dataset size = {history['size']}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    axes[0].set_ylabel("Loss")

    fig.suptitle(
        "Training/Validation Curves by Dataset Size "
        f"(epoch ceiling = {num_epochs})"
    )

    fig.tight_layout()
    fig.savefig(path)

    plt.close(fig)


# ============================================================
# MSE vs dataset size
# ============================================================

def save_mse_vs_size_plot(
    results: list[dict],
    path: Path,
):
    """
    Plot paired model and training-mean test MSE bars by training-set size.
    """
    metric_labels = (("payoff", "Payoff"), ("safety", "Safety"), ("fuel", "Fuel"))
    sizes = [result["training_records"] for result in results]
    positions = np.arange(len(sizes), dtype=np.float32)
    width = 0.36
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    for axis, (metric, label) in zip(axes, metric_labels):
        model_values = [result[metric]["model_mse"] for result in results]
        baseline_values = [result[metric]["mean_baseline_mse"] for result in results]
        axis.bar(positions - width / 2, model_values, width, label="Model")
        axis.bar(positions + width / 2, baseline_values, width, label="Training mean")
        axis.set_title(label)
        axis.set_xlabel("Training records")
        axis.set_ylabel("Fixed test-set MSE")
        axis.set_xticks(positions, [f"{size:,}" for size in sizes])
        axis.grid(axis="y", alpha=0.3)

    axes[0].legend()
    figure.suptitle("Model vs. Training-Mean Baseline on the Fixed Test Set")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


# ============================================================
# Analyze trend
# ============================================================

def analyze_trend(
    results: list[dict],
):
    """Summarize training-size trends for the Markdown report."""
    metric_labels = ("payoff", "safety", "fuel")
    analysis = {}
    for metric in metric_labels:
        model_values = [result[metric]["model_mse"] for result in results]
        winning_sizes = [
            result["training_records"]
            for result in results
            if result[metric]["improvement_percent"] is not None
            and result[metric]["improvement_percent"] > 0.0
        ]
        analysis[metric] = {
            "best_training_records": results[int(np.argmin(model_values))]["training_records"],
            "model_beats_training_mean_at_sizes": winning_sizes,
            "model_beats_training_mean_for_all_sizes": len(winning_sizes) == len(results),
            "model_mse_decreases_monotonically_with_training_size": all(
                later <= earlier for earlier, later in zip(model_values, model_values[1:])
            ),
        }
    return analysis


def build_markdown_report(summary: dict) -> str:
    """Render the JSON comparison summary as a concise human-readable report."""
    split = summary["split"]
    lines = [
        "# Dataset-Size Comparison",
        "",
        f"Shared validation: {split['validation_records']:,} records / {split['validation_contexts']:,} contexts; "
        f"test: {split['test_records']:,} records / {split['test_contexts']:,} contexts (seed {split['seed']}).",
        "",
        "| Training records | Payoff model / mean MSE (improvement) | Safety model / mean MSE (improvement) | Fuel model / mean MSE (improvement) |",
        "|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        format_pair = lambda key: (
            f"{result[key]['model_mse']:.3f} / {result[key]['mean_baseline_mse']:.3f}"
            f" ({result[key]['improvement_percent']:+.1f}% improvement)"
            if result[key]["improvement_percent"] is not None
            else f"{result[key]['model_mse']:.3f} / {result[key]['mean_baseline_mse']:.3f} (n/a)"
        )
        lines.append(
            f"| {result['training_records']:,} | {format_pair('payoff')} "
            f"| {format_pair('safety')} | {format_pair('fuel')} |"
        )

    lines.extend(["", "Analysis:"])
    for outcome, details in analyze_trend(summary["results"]).items():
        sizes = ", ".join(f"{size:,}" for size in details["model_beats_training_mean_at_sizes"])
        if not sizes:
            sizes = "none"
        trend = "decreased monotonically" if details[
            "model_mse_decreases_monotonically_with_training_size"
        ] else "did not decrease monotonically"
        lines.append(
            f"- {outcome}: best at {details['best_training_records']:,} training records; "
            f"beats mean at {sizes}; MSE {trend}."
        )
    lines.append("")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compare SkillGenerator performance as training data grows.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Directory containing rollout files.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 3000, 5250],
        help="Training-subset sizes to compare (default: 1000 3000 5250).",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Maximum training epochs per size.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42, help="Seed for the shared context-disjoint split.")
    parser.add_argument("--validation-fraction", type=float, default=0.125)
    parser.add_argument("--test-fraction", type=float, default=0.125)
    parser.add_argument("--rollout-output-dir", type=str, default="data/dataset_size_comparison_rollouts")
    parser.add_argument("--output-dir", type=str, default="plots/dataset_size_comparison")
    parser.add_argument("--report-json", type=str, default="demo/artifacts/dataset_size_comparison_report.json")
    parser.add_argument("--report-md", type=str, default="demo/artifacts/dataset_size_comparison_report.md")

    args = parser.parse_args()
    if any(size <= 0 for size in args.sizes):
        raise ValueError("--sizes must contain positive training sizes")
    if args.validation_fraction <= 0.0 or args.test_fraction <= 0.0:
        raise ValueError("validation and test fractions must be positive")
    if args.validation_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation and test fractions must sum to less than 1.0")
    if len(set(args.sizes)) != len(args.sizes):
        print("Warning: duplicate training sizes were provided; duplicates will be removed.")
    args.sizes = sorted(set(args.sizes))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_output_dir = Path(args.rollout_output_dir)

    print("Loading full dataset...")
    full_dataset = SkillDataset(args.data_dir)
    print(f"Available rollout files: {len(full_dataset.files)}")

    available_records = len(full_dataset.files)
    training_fraction = 1.0 - args.validation_fraction - args.test_fraction
    required_source_size = int(math.ceil(max(args.sizes) / training_fraction))
    if required_source_size > available_records:
        additional_records = required_source_size - available_records
        collection_command = (
            "python -m data_collector.collect "
            f"--episodes {additional_records} "
            f'--save-dir "{args.data_dir}" '
            f"--seed {args.seed}"
        )
        separate_collection_command = (
            "python -m data_collector.collect "
            f"--episodes {required_source_size} "
            '--save-dir "data/dataset_size_comparison_source" '
            f"--seed {args.seed}"
        )
        print(
            f"Insufficient data: requested training sizes {args.sizes} need at least "
            f"{required_source_size:,} source records with the selected holdout fractions; "
            f"only {available_records:,} are available. Collect at least "
            f"{additional_records:,} more records, then rerun. This is an estimate; "
            "context-grouped splitting may require additional records. No models were trained.\n"
            "Recommended: append the missing records to the current dataset:\n"
            f"{collection_command}\n"
            "For an isolated dataset, collect the full minimum into a separate directory, "
            "then rerun with --data-dir set to that directory:\n"
            f"{separate_collection_command}"
        )
        return
    source_size = required_source_size
    print(
        f"Using the minimum estimated source pool of {source_size:,} records "
        "for the requested sizes and holdout fractions."
    )

    try:
        fixed_split = create_holdout_split(
            full_dataset,
            seed=args.seed,
            source_size=source_size,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
    except ValueError as error:
        print(
            f"Warning: cannot create the requested fixed validation/test split: {error}. "
            "Provide more records or reduce the holdout fractions."
        )
        return

    available_training = len(fixed_split["training_pool_files"])
    if available_training < max(args.sizes):
        print(
            f"Insufficient context groups: the estimated {source_size:,}-record pool "
            f"leaves {available_training:,} training records, fewer than the requested "
            f"maximum of {max(args.sizes):,}. Collect more data and rerun; no models were trained."
        )
        return
    if fixed_split["total_size"] != source_size:
        print(
            f"Selected {fixed_split['total_size']:,} records rather than {source_size:,} "
            "to keep all records from each matching starting context together."
        )
    splits = {
        size: create_dataset_split(fixed_split, size)
        for size in args.sizes
    }

    save_dataset_files(splits, rollout_output_dir)
    print(f"Split files saved -> {rollout_output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    histories = []

    for size, split in splits.items():
        print(f"\n{'=' * 60}")
        print(f"Training subset target: {size}")
        print(f"{'=' * 60}")

        print(f"  Train:      {len(split['train_records'])}")
        print(f"  Validation: {len(split['validation_records'])} (fixed)")
        print(f"  Test:       {len(split['test_records'])}")
        training_files_dir = rollout_output_dir / f"size_{len(split['train_records'])}" / "train"
        print(f"  Training files -> {training_files_dir}")

        torch.manual_seed(args.seed)

        train_loader = DataLoader(
            InMemorySkillDataset(split["train_records"]),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            InMemorySkillDataset(split["validation_records"]),
            batch_size=args.batch_size,
            shuffle=False,
        )

        model = SkillGenerator(
            input_dim=8,
            hidden_dim=args.hidden_dim,
            motive_dim=2,
        )
        model.to(device)

        loss_fn = GeneratorLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_state, hist_train, hist_val, best_epoch = run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_epochs=args.epochs,
            patience=args.patience,
        )

        model.load_state_dict(best_state)
        test_mse = evaluate_on_test_set(
            model,
            split["train_records"],
            split["test_records"],
            device=device,
        )

        print(f"  Best epoch: {best_epoch}/{args.epochs}")
        for metric, label in (("payoff_mse", "Payoff"), ("safety_mse", "Safety"), ("fuel_mse", "Fuel")):
            model_error = test_mse["model_mse"][metric]
            baseline_error = test_mse["training_mean_mse"][metric]
            print(f"  Test {label} MSE: model={model_error:.4f}, training-mean={baseline_error:.4f}")

        histories.append({
            "size": len(split["train_records"]),
            "train": hist_train,
            "val": hist_val,
            "best_epoch": best_epoch,
        })

        results.append({
            "training_records": len(split["train_records"]),
            "payoff": {
                "model_mse": test_mse["model_mse"]["payoff_mse"],
                "mean_baseline_mse": test_mse["training_mean_mse"]["payoff_mse"],
                "improvement_percent": test_mse["improvement_percent"]["payoff_mse"],
            },
            "safety": {
                "model_mse": test_mse["model_mse"]["safety_mse"],
                "mean_baseline_mse": test_mse["training_mean_mse"]["safety_mse"],
                "improvement_percent": test_mse["improvement_percent"]["safety_mse"],
            },
            "fuel": {
                "model_mse": test_mse["model_mse"]["fuel_mse"],
                "mean_baseline_mse": test_mse["training_mean_mse"]["fuel_mse"],
                "improvement_percent": test_mse["improvement_percent"]["fuel_mse"],
            },
        })

    if not results:
        print("\nNo dataset sizes could be run.")
        return

    curves_path = output_dir / "combined_training_curves.png"
    save_combined_curves_plot(histories, args.epochs, curves_path)
    print(f"\nTraining curves saved -> {curves_path}")

    mse_plot_path = output_dir / "model_vs_training_mean_mse.png"
    save_mse_vs_size_plot(results, mse_plot_path)
    print(f"MSE comparison plot saved -> {mse_plot_path}")

    report = {
        "split": {
            "seed": args.seed,
            "validation_records": len(fixed_split["validation_files"]),
            "validation_contexts": fixed_split["validation_contexts"],
            "test_records": len(fixed_split["test_files"]),
            "test_contexts": fixed_split["test_contexts"],
        },
        "results": results,
    }
    report_json_path = Path(args.report_json)
    report_md_path = Path(args.report_md)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    report_md_path.write_text(build_markdown_report(report), encoding="utf-8")
    print(f"JSON report saved -> {report_json_path}")
    print(f"Markdown report saved -> {report_md_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()

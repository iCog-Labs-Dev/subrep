
"""
Compare SkillGenerator performance across default total dataset sizes of 1,000,
3,000, and 7,000 episodes.

Each requested dataset size is split independently into:
75% training, 12.5% validation, 12.5% test
Example:
    python -m generator.compare_dataset_sizes --data-dir data/raw

The purpose is to study how SkillGenerator performance changes
as the TOTAL amount of available data increases.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

def create_dataset_split(
    full_dataset: SkillDataset,
    size: int,
    seed: int,
):
    """
    Select exactly `size` files from the full dataset and split them.
    """
    if size > len(full_dataset.files):
        raise ValueError(
            f"Requested dataset size {size} exceeds available files "
            f"({len(full_dataset.files)})."
        )

    all_files = sorted(full_dataset.files)
    rng = random.Random(seed)

    shuffled_files = all_files.copy()
    rng.shuffle(shuffled_files)

    selected_files = shuffled_files[:size]
    train_size = int(size * 0.75)
    val_size = int(size * 0.125)

    train_files = selected_files[:train_size]
    val_files = selected_files[train_size:train_size + val_size]
    test_files = selected_files[train_size + val_size:]

    file_to_record = dict(zip(full_dataset.files, full_dataset.data))

    train_records = [file_to_record[fp] for fp in train_files]
    val_records = [file_to_record[fp] for fp in val_files]
    test_records = [file_to_record[fp] for fp in test_files]

    return {
        "train_files": train_files,
        "val_files": val_files,
        "test_files": test_files,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
    }


# ============================================================
# Save the dataset files so they can be inspected
# ============================================================

def save_dataset_files(
    split: dict,
    full_dataset: SkillDataset,
    output_dir: Path,
):
    """
    Copy the files belonging to this dataset into:
    size_N/
         train/
         val/
         test/
    """

    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for fp in split["train_files"]:
        shutil.copy2(fp, train_dir / os.path.basename(fp))

    for fp in split["val_files"]:
        shutil.copy2(fp, val_dir / os.path.basename(fp))

    for fp in split["test_files"]:
        shutil.copy2(fp, test_dir / os.path.basename(fp))


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
    test_records: list,
):
    """
    Evaluate the trained model on this dataset's OWN test set.

    Returns:
        payoff MSE
        safety MSE
        fuel MSE
    """

    loader = DataLoader(
        InMemorySkillDataset(test_records),
        batch_size=64,
        shuffle=False,
    )

    preds_payoff = []
    targets_payoff = []

    preds_motives = []
    targets_motives = []

    with torch.no_grad():
        for obs, payoff, motives in loader:
            predicted_payoff, predicted_motives = model(obs)

            preds_payoff.append(predicted_payoff)
            targets_payoff.append(payoff)
            preds_motives.append(predicted_motives)
            targets_motives.append(motives)

    predicted_payoff = torch.cat(preds_payoff)
    target_payoff = torch.cat(targets_payoff)
    predicted_motives = torch.cat(preds_motives)
    target_motives = torch.cat(targets_motives)

    return {
        "payoff_mse": F.mse_loss(
            predicted_payoff,
            target_payoff,
        ).item(),
        "safety_mse": F.mse_loss(
            predicted_motives[:, 0],
            target_motives[:, 0],
        ).item(),
        "fuel_mse": F.mse_loss(
            predicted_motives[:, 1],
            target_motives[:, 1],
        ).item(),
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
    num_epochs: int,
    path: Path,
):
    """
    Plot test MSE against TOTAL dataset size.
    """

    sizes = [
        result["dataset_size"]
        for result in results
    ]

    plt.figure(figsize=(8, 5))

    for key, label in [
        ("payoff_mse", "Payoff"),
        ("safety_mse", "Safety"),
        ("fuel_mse", "Fuel"),
    ]:
        plt.plot(
            sizes,
            [
                result["test_mse"][key]
                for result in results
            ],
            marker="o",
            label=f"{label} MSE",
        )

    plt.xlabel("Total dataset size (episodes)")
    plt.ylabel("Test-set MSE")
    plt.title(
        "Test MSE vs. Total Dataset Size "
        f"(epoch ceiling = {num_epochs})"
    )

    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()


# ============================================================
# Analyze trend
# ============================================================

def analyze_trend(
    results: list[dict],
):
    """
    Report whether payoff MSE decreases as total dataset
    size increases.
    """

    if len(results) < 2:
        return (
            "Need at least two dataset sizes "
            "to compare a trend."
        )

    payoff_mses = [
        result["test_mse"]["payoff_mse"]
        for result in results
    ]

    improved = all(
        later <= earlier
        for earlier, later in zip(
            payoff_mses,
            payoff_mses[1:],
        )
    )

    if improved:
        return (
            "Payoff MSE decreased monotonically "
            "as total dataset size increased."
        )

    return (
        "Payoff MSE did NOT consistently decrease "
        "as total dataset size increased."
    )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare SkillGenerator performance across different total dataset sizes."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing the complete pool of rollout files.",
    )

    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 3000, 7000],
        help="TOTAL dataset sizes to compare. Default: 1000 3000 7000.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Maximum number of training epochs for every dataset size.",
    )

    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to make dataset selection and splitting reproducible.",
    )

    parser.add_argument(
        "--rollout-output-dir",
        type=str,
        default="data/dataset_size_comparison_rollouts",
        help="Where each complete dataset will be saved.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/dataset_size_comparison",
        help="Where plots and results JSON are saved.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_output_dir = Path(args.rollout_output_dir)

    print("Loading full dataset...")
    full_dataset = SkillDataset(args.data_dir)
    print(f"Available rollout files: {len(full_dataset.files)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = []
    histories = []

    for size in args.sizes:
        print(f"\n{'=' * 60}")
        print(f"Dataset size: {size}")
        print(f"{'=' * 60}")

        try:
            split = create_dataset_split(full_dataset, size, args.seed)
        except ValueError as e:
            print(f"Skipping dataset size {size}: {e}")
            continue

        print(f"  Total:      {size}")
        print(f"  Train:      {len(split['train_records'])}")
        print(f"  Validation: {len(split['val_records'])}")
        print(f"  Test:       {len(split['test_records'])}")

        dataset_output_dir = rollout_output_dir / f"size_{size}"
        save_dataset_files(split, full_dataset, dataset_output_dir)
        print(f"  Dataset saved -> {dataset_output_dir}")

        torch.manual_seed(args.seed)

        train_loader = DataLoader(
            InMemorySkillDataset(split["train_records"]),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            InMemorySkillDataset(split["val_records"]),
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
        test_mse = evaluate_on_test_set(model, split["test_records"])

        print(f"  Best epoch: {best_epoch}/{args.epochs}")
        print(f"  Test payoff MSE: {test_mse['payoff_mse']:.4f}")
        print(f"  Test safety MSE: {test_mse['safety_mse']:.4f}")
        print(f"  Test fuel MSE: {test_mse['fuel_mse']:.4f}")

        histories.append({
            "size": size,
            "train": hist_train,
            "val": hist_val,
            "best_epoch": best_epoch,
        })

        results.append({
            "dataset_size": size,
            "train_size": len(split["train_records"]),
            "validation_size": len(split["val_records"]),
            "test_size": len(split["test_records"]),
            "train_fraction": len(split["train_records"]) / size,
            "validation_fraction": len(split["val_records"]) / size,
            "test_fraction": len(split["test_records"]) / size,
            "epoch_ceiling": args.epochs,
            "stopped_at_epoch": best_epoch,
            "test_mse": test_mse,
        })

    if not results:
        print("\nNo dataset sizes could be run.")
        return

    suffix = f"epochs{args.epochs}"
    curves_path = output_dir / f"combined_training_curves_{suffix}.png"
    save_combined_curves_plot(histories, args.epochs, curves_path)
    print(f"\nTraining curves saved -> {curves_path}")

    mse_plot_path = output_dir / f"test_mse_vs_dataset_size_{suffix}.png"
    save_mse_vs_size_plot(results, args.epochs, mse_plot_path)
    print(f"MSE plot saved -> {mse_plot_path}")

    trend_note = analyze_trend(results)
    print(f"\nTrend: {trend_note}")

    summary_path = output_dir / f"dataset_size_comparison_results_{suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "seed": args.seed,
                "train_fraction": 0.75,
                "validation_fraction": 0.125,
                "test_fraction": 0.125,
                "epoch_ceiling": args.epochs,
                "patience": args.patience,
                "results": results,
                "trend_analysis": trend_note,
            },
            f,
            indent=2,
        )

    print(f"Summary saved -> {summary_path}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()

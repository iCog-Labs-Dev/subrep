"""
Evaluate SkillGenerator Prediction Error (MSE) -- on the held-out TEST split.

Reports payoff MSE and EACH motive feature's MSE separately (not combined
into one number), and compares the trained model against a simple
mean-outcome baseline (predict the training set's mean payoff/motives for
every input, ignoring the state entirely). 


"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from generator.skill_generator import SkillGenerator
from generator.train_generator import SkillDataset, InMemorySkillDataset
from generator.dataset_split import load_split_manifest, apply_split_manifest, DEFAULT_MANIFEST_PATH


def _stack_payoffs_and_motives(records: list) -> tuple[torch.Tensor, torch.Tensor]:
    payoffs = torch.stack([r[1] for r in records])
    motives = torch.stack([r[2] for r in records])
    return payoffs, motives


def _mse_per_feature(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    """MSE computed independently per column (feature), not averaged together."""
    return [F.mse_loss(pred[:, i], target[:, i]).item() for i in range(pred.shape[1])]


def evaluate_dataset(
    model: SkillGenerator,
    data_dir: str,
    manifest: dict,
    output_dir: str,
) -> dict | None:
    print(f"Loading dataset from: {data_dir}/ ...")
    if not os.path.exists(data_dir):
        print("  -> Directory not found, skipping.")
        return None

    try:
        full_dataset = SkillDataset(data_dir)
    except FileNotFoundError:
        print("  -> No .npz files found, skipping.")
        return None

    split = apply_split_manifest(full_dataset.files, full_dataset.data, manifest)
    test_dataset = InMemorySkillDataset(split.test)
    print(f"  -> Using {len(test_dataset)} held-out TEST records "
          f"(of {len(full_dataset)} total; train/val excluded per saved manifest).")

    # --- Real model, evaluated on the test split ---
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    all_pred_payoff, all_target_payoff = [], []
    all_pred_motives, all_target_motives = [], []
    with torch.no_grad():
        for batch_obs, batch_payoff, batch_motives in loader:
            pred_payoff, pred_motives = model(batch_obs)
            all_pred_payoff.append(pred_payoff)
            all_target_payoff.append(batch_payoff)
            all_pred_motives.append(pred_motives)
            all_target_motives.append(batch_motives)

    pred_payoff = torch.cat(all_pred_payoff)
    target_payoff = torch.cat(all_target_payoff)
    pred_motives = torch.cat(all_pred_motives)
    target_motives = torch.cat(all_target_motives)

    model_payoff_mse = F.mse_loss(pred_payoff, target_payoff).item()
    model_motive_mse = _mse_per_feature(pred_motives, target_motives)  # [Safety, Fuel]

    # --- Mean-outcome baseline: computed from the TRAIN split only, applied
    #     as a constant prediction (ignoring the state) on the TEST split ---
    train_payoff, train_motives = _stack_payoffs_and_motives(split.train)
    mean_payoff = train_payoff.mean(dim=0)
    mean_motives = train_motives.mean(dim=0)

    baseline_pred_payoff = mean_payoff.expand_as(target_payoff)
    baseline_pred_motives = mean_motives.expand_as(target_motives)

    baseline_payoff_mse = F.mse_loss(baseline_pred_payoff, target_payoff).item()
    baseline_motive_mse = _mse_per_feature(baseline_pred_motives, target_motives)

    result = {
        "data_dir": data_dir,
        "n_test_records": len(test_dataset),
        "model": {
            "payoff_mse": model_payoff_mse,
            "safety_mse": model_motive_mse[0],
            "fuel_mse": model_motive_mse[1],
        },
        "mean_baseline": {
            "payoff_mse": baseline_payoff_mse,
            "safety_mse": baseline_motive_mse[0],
            "fuel_mse": baseline_motive_mse[1],
        },
        "model_beats_baseline": {
            "payoff": model_payoff_mse < baseline_payoff_mse,
            "safety": model_motive_mse[0] < baseline_motive_mse[0],
            "fuel": model_motive_mse[1] < baseline_motive_mse[1],
        },
    }

    print(f"Results over {len(test_dataset)} TEST records (never seen during training):")
    print(f"  {'':12s} {'Model':>12s}   {'Mean-baseline':>14s}   Model beats baseline?")
    print(f"  {'Payoff MSE':12s} {model_payoff_mse:12.4f}   {baseline_payoff_mse:14.4f}   {result['model_beats_baseline']['payoff']}")
    print(f"  {'Safety MSE':12s} {model_motive_mse[0]:12.4f}   {baseline_motive_mse[0]:14.4f}   {result['model_beats_baseline']['safety']}")
    print(f"  {'Fuel MSE':12s} {model_motive_mse[1]:12.4f}   {baseline_motive_mse[1]:14.4f}   {result['model_beats_baseline']['fuel']}")
    print("-" * 60)

    out_path = Path(output_dir) / "generator_mse_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_path}")
    print("-" * 60)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SkillGenerator MSE on held-out test data.")
    parser.add_argument("--model-path", type=str, default="models/generator.pt")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                         help="Single-policy rollout directory (ppo_deterministic). "
                              "data/raw_mixed is out of scope -- see generator/README.md.")
    parser.add_argument("--split-manifest", type=str, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=str, default="demo/artifacts")
    args = parser.parse_args()

    print("=" * 60)
    print("  SkillGenerator MSE Evaluation (held-out test split)")
    print("  Predicts outcomes for the ppo_deterministic policy only.")
    print("=" * 60)

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at '{args.model_path}'")
        print("Run 'python -m generator.train_generator' first.")
        return

    try:
        manifest = load_split_manifest(args.split_manifest)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    model = SkillGenerator(input_dim=8, hidden_dim=64, motive_dim=2)
    model.load(args.model_path)
    model.eval()
    print(f"Loaded model from {args.model_path}")
    print(f"Loaded split manifest from {args.split_manifest} "
          f"(seed={manifest['seed']}, test_frac={manifest['test_frac']})")
    print("-" * 60)

    evaluate_dataset(model, args.data_dir, manifest, args.output_dir)


if __name__ == "__main__":
    main()
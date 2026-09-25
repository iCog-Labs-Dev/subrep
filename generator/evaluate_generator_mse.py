"""
Evaluate SkillGenerator Prediction Error (MSE) -- on the held-out TEST split.

Compare the Payoff and Motive Mean Squared Error (MSE) of the
trained SkillGenerator against data it was never trained on.

Usage:
    python -m generator.evaluate_generator_mse \
      --model-path models/generator.pt \
      --data-dirs data/raw \
      --split-manifest data/generator_split_manifest.json
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from generator.skill_generator import SkillGenerator
from generator.train_generator import SkillDataset, InMemorySkillDataset
from generator.dataset_split import load_split_manifest, apply_split_manifest, DEFAULT_MANIFEST_PATH


def evaluate_dataset(model: SkillGenerator, data_dir: str, manifest: dict) -> None:
    """
    Evaluate on the held-out TEST group only, identified by looking up each
    file's label in the manifest train_generator.py already saved to disk --
    never by recomputing a split from --seed/--frac flags, which could
    silently drift out of sync with whatever training actually used.
    """
    print(f"Loading dataset from: {data_dir}/ ...")
    if not os.path.exists(data_dir):
        print("  -> Directory not found, skipping.")
        return

    try:
        full_dataset = SkillDataset(data_dir)
    except FileNotFoundError:
        print("  -> No .npz files found, skipping.")
        return

    split = apply_split_manifest(full_dataset.files, full_dataset.data, manifest)
    test_dataset = InMemorySkillDataset(split.test)
    print(f"  -> Using {len(test_dataset)} held-out TEST records "
          f"(of {len(full_dataset)} total; train/val excluded per saved manifest).")

    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    total_payoff_mse = 0.0
    total_motive_mse = 0.0

    with torch.no_grad():
        for batch_obs, batch_payoff, batch_motives in loader:
            pred_payoff, pred_motives = model(batch_obs)

            payoff_mse = F.mse_loss(pred_payoff, batch_payoff, reduction='sum')
            motive_mse = F.mse_loss(pred_motives, batch_motives, reduction='sum')

            total_payoff_mse += payoff_mse.item()
            total_motive_mse += motive_mse.item()

    num_samples = len(test_dataset)
    mean_payoff_mse = total_payoff_mse / num_samples
    mean_motive_mse = total_motive_mse / num_samples

    print(f"Results over {num_samples} TEST records (never seen during training):")
    print(f"  Payoff MSE: {mean_payoff_mse:.4f}")
    print(f"  Motive MSE: {mean_motive_mse:.4f}")
    print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SkillGenerator MSE on held-out test data.")
    parser.add_argument("--model-path", type=str, default="models/generator.pt")
    parser.add_argument("--data-dirs", type=str, nargs="+", default=["data/raw", "data/raw_mixed"])
    parser.add_argument("--split-manifest", type=str, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()

    print("=" * 60)
    print("  SkillGenerator MSE Evaluation (held-out test split)")
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
    print("-" * 40)

    for data_dir in args.data_dirs:
        evaluate_dataset(model, data_dir, manifest)


if __name__ == "__main__":
    main()
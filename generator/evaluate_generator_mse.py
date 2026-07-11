"""
Evaluate SkillGenerator Prediction Error (MSE).

Compare the Payoff and Motive Mean Squared Error (MSE) of the 
trained SkillGenerator against a target dataset.

Usage:
    python -m generator.evaluate_generator_mse \
      --model-path models/generator.pt \
      --data-dir data/raw_mixed
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from generator.skill_generator import SkillGenerator
from generator.train_generator import SkillDataset


def evaluate_dataset(model: SkillGenerator, data_dir: str) -> None:
    print(f"Loading dataset from: {data_dir}/ ...")
    if not os.path.exists(data_dir):
        print("  -> Directory not found, skipping.")
        return
        
    try:
        dataset = SkillDataset(data_dir)
    except FileNotFoundError:
        print("  -> No .npz files found, skipping.")
        return

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    total_payoff_mse = 0.0
    total_motive_mse = 0.0
    
    with torch.no_grad():
        for batch_obs, batch_payoff, batch_motives in loader:
            pred_payoff, pred_motives = model(batch_obs)
            
            payoff_mse = F.mse_loss(pred_payoff, batch_payoff, reduction='sum')
            motive_mse = F.mse_loss(pred_motives, batch_motives, reduction='sum')
            
            total_payoff_mse += payoff_mse.item()
            total_motive_mse += motive_mse.item()
            
    num_samples = len(dataset)
    mean_payoff_mse = total_payoff_mse / num_samples
    mean_motive_mse = total_motive_mse / num_samples
    
    print(f"Results over {num_samples} records:")
    print(f"  Payoff MSE: {mean_payoff_mse:.4f}")
    print(f"  Motive MSE: {mean_motive_mse:.4f}")
    print("-" * 40)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SkillGenerator MSE.")
    parser.add_argument("--model-path", type=str, default="models/generator.pt")
    parser.add_argument("--data-dirs", type=str, nargs="+", default=["data/raw", "data/raw_mixed"])
    args = parser.parse_args()

    print("=" * 60)
    print("  SkillGenerator MSE Evaluation")
    print("=" * 60)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at '{args.model_path}'")
        print("Run 'python -m generator.train_generator' first.")
        return

    model = SkillGenerator(input_dim=8, hidden_dim=64, motive_dim=2)
    model.load(args.model_path)
    model.eval()
    print(f"Loaded model from {args.model_path}")
    print("-" * 40)

    for data_dir in args.data_dirs:
        evaluate_dataset(model, data_dir)


if __name__ == "__main__":
    main()

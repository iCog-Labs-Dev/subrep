import os
import glob
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from generator.skill_generator import SkillGenerator
from generator.losses import GeneratorLoss

# Hyperparameters
DATA_DIR = "data/raw"
MODEL_DIR = "models"
PLOT_DIR = "plots"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
SEED = 42


class SkillDataset(Dataset):
    """
    Dataset loader for SubRep .npz rollout records.
    Loads all episodes from data/raw and provides tensors for training.
    """
    def __init__(self, data_dir: str):
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}. Run DataCollector first.")
        
        self.data = []
        for file in self.files:
            record = np.load(file, allow_pickle=True)
            obs = torch.tensor(record['obs'], dtype=torch.float32)
            payoff = torch.tensor(float(record['payoff']), dtype=torch.float32).unsqueeze(0)
            motives = torch.tensor(record['motives'], dtype=torch.float32)
            self.data.append((obs, payoff, motives))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx]


def train_one_epoch(
    model: SkillGenerator,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: GeneratorLoss,
) -> tuple[float, float, float]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0.0
    total_payoff_loss = 0.0
    total_motive_loss = 0.0
    
    for obs, target_payoff, target_motives in loader:
        optimizer.zero_grad()
        
        pred_payoff, pred_motives = model(obs)
        losses = loss_fn.breakdown(pred_payoff, pred_motives, target_payoff, target_motives)
        
        loss = losses["total_loss"]
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * obs.size(0)
        total_payoff_loss += losses["payoff_loss"].item() * obs.size(0)
        total_motive_loss += losses["motive_loss"].item() * obs.size(0)
        
    num_samples = len(loader.dataset)
    return (
        total_loss / num_samples,
        total_payoff_loss / num_samples,
        total_motive_loss / num_samples,
    )


def train() -> None:
    """Main training loop."""
    # Seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Loading dataset from {DATA_DIR}/ ...")
    try:
        dataset = SkillDataset(DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    print(f"{len(dataset)} episodes found.")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model, loss, and optimizer
    model = SkillGenerator(input_dim=8, hidden_dim=HIDDEN_DIM, motive_dim=2)
    loss_fn = GeneratorLoss(payoff_weight=1.0, motive_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history_total = []
    
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        loss, p_loss, m_loss = train_one_epoch(model, loader, optimizer, loss_fn)
        history_total.append(loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == NUM_EPOCHS - 1:
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {loss:.6f} "
                  f"(payoff: {p_loss:.6f}, motives: {m_loss:.6f})")

    # Save trained model weights
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "generator.pt")
    model.save(model_path)
    print(f"\nModel saved -> {model_path}")

    # Generate and save loss plot
    os.makedirs(PLOT_DIR, exist_ok=True)
    plot_path = os.path.join(PLOT_DIR, "generator_training.png")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), history_total, label="Total Loss (MSE)", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Skill Generator Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved  -> {plot_path}")
    print("Training complete.")

if __name__ == "__main__":
    train()

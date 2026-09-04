import argparse
import copy
import csv
import os
import glob
import numpy as np
import torch
import torch.optim as optim
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from generator.skill_generator import SkillGenerator
from generator.losses import GeneratorLoss
from generator.dataset_split import (
    compute_split_assignment,
    save_split_manifest,
    apply_split_manifest,
    DEFAULT_MANIFEST_PATH,
)

# CHANGE 1: hyperparameters are no longer hardcoded module constants only.
# They remain here as defaults for argparse arguments 
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
SEED = 42


class SkillDataset(Dataset):
    """
    Dataset loader for SubRep .npz rollout records.
    Loads all episodes from a directory and provides tensors for training.
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

class InMemorySkillDataset(Dataset):
    """Wrap a list of (obs, payoff, motives) tuples (one split group) as a Dataset."""
    def __init__(self, records: list):
        self.data = records

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
    """Train the model for one epoch (unchanged from before)."""
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

def evaluate_loss(
    model: SkillGenerator,
    loader: DataLoader,
    loss_fn: GeneratorLoss,
) -> tuple[float, float, float]:
    """Compute loss on a held-out split WITHOUT updating any weights."""
    model.eval()
    total_loss = 0.0
    total_payoff_loss = 0.0
    total_motive_loss = 0.0

    with torch.no_grad():
        for obs, target_payoff, target_motives in loader:
            pred_payoff, pred_motives = model(obs)
            losses = loss_fn.breakdown(pred_payoff, pred_motives, target_payoff, target_motives)

            total_loss += losses["total_loss"].item() * obs.size(0)
            total_payoff_loss += losses["payoff_loss"].item() * obs.size(0)
            total_motive_loss += losses["motive_loss"].item() * obs.size(0)

    num_samples = len(loader.dataset)
    return (
        total_loss / num_samples,
        total_payoff_loss / num_samples,
        total_motive_loss / num_samples,
    )


def train(
    data_dir: str,
    output: str,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    hidden_dim: int = HIDDEN_DIM,
    seed: int = SEED,
    train_frac: float = 0.75,
    val_frac: float = 0.125,
    test_frac: float = 0.125,
    patience: int = 10,
    split_manifest_path: str = DEFAULT_MANIFEST_PATH,
) -> None:
    """Main training loop, now with a real train/val split and model selection."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Loading dataset from {data_dir}/ ...")
    try:
        full_dataset = SkillDataset(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"{len(full_dataset)} episodes found.")

    '''Training below only ever touches `split.train` and `split.val`; 
    `split.test` is loaded but never used past this point in this file. 
    Because the manifest is persisted, evaluate_generator_mse.py reads 
    the SAME assignment back later instead of recomputing it 
    from seed/fraction flags that could drift out of sync'''
    assignment = compute_split_assignment(
        full_dataset.files,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
    )
    save_split_manifest(
        assignment,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        path=split_manifest_path,
    )
    manifest = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "assignment": assignment,
    }
    split = apply_split_manifest(full_dataset.files, full_dataset.data, manifest)
    print(
        f"Split: {len(split.train)} train / {len(split.val)} val / "
        f"{len(split.test)} test -- manifest saved to {split_manifest_path} "
        f"(evaluate_generator_mse.py will read this back for the test set)"
    )

    train_loader = DataLoader(InMemorySkillDataset(split.train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(InMemorySkillDataset(split.val), batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = SkillGenerator(input_dim=8, hidden_dim=hidden_dim, motive_dim=2)
    loss_fn = GeneratorLoss(payoff_weight=1.0, motive_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history_train_loss = []
    history_val_loss = [] 

    # Instead of always saving whatever the model looks like after the final epoch, 
    # we keep a copy of the weights from whichever epoch had the best (lowest) validation loss.
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = -1
    epochs_without_improvement = 0

    print("\nStarting training...")
    for epoch in range(num_epochs):
        train_loss, train_p_loss, train_m_loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_p_loss, val_m_loss = evaluate_loss(model, val_loader, loss_fn)

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"train_loss: {train_loss:.6f} (payoff: {train_p_loss:.6f}, motives: {train_m_loss:.6f}) | "
                f"val_loss: {val_loss:.6f} (payoff: {val_p_loss:.6f}, motives: {val_m_loss:.6f})"
            )

        # early stopping. If validation loss doesn't improve 
        # for `patience` epochs in a row, stop early.
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch+1}: "
                    f"no val_loss improvement for {patience} epochs "
                    f"(best was {best_val_loss:.6f} at epoch {best_epoch})."
                )
                break

    model.load_state_dict(best_model_state)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = str(out_path)
    model.save(model_path)
    print(f"\nBest model (epoch {best_epoch}, val_loss={best_val_loss:.6f}) saved -> {model_path}")

    # log both curves to CSV (not just a PNG) so validation
    # numbers are machine-readable for later reporting, not just a picture.
    log_dir = Path("plots")
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "generator_training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (t, v) in enumerate(zip(history_train_loss, history_val_loss), start=1):
            writer.writerow([i, t, v])
    print(f"Training log saved -> {csv_path}")

    plot_path = str(log_dir / "generator_training.png")
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history_train_loss) + 1), history_train_loss, label="Train Loss", color="blue")
    plt.plot(range(1, len(history_val_loss) + 1), history_val_loss, label="Val Loss", color="orange")
    plt.axvline(best_epoch, color="green", linestyle="--", label=f"Best epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Skill Generator Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved  -> {plot_path}")
    print("Training complete. (Test split was not touched -- run evaluate_generator_mse.py for final numbers.)")


def main():
    parser = argparse.ArgumentParser(description="Train SkillGenerator with train/val split and model selection.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to input .npz data dir")
    parser.add_argument("--output", type=str, default="models/generator.pt", help="Path to output model .pt file")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--val-frac", type=float, default=0.125)
    parser.add_argument("--test-frac", type=float, default=0.125)
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs")
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=DEFAULT_MANIFEST_PATH,
        help="Where to save the train/val/test file assignment for evaluate_generator_mse.py to reuse.",
    )
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output=args.output,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        patience=args.patience,
        split_manifest_path=args.split_manifest,
    )


if __name__ == "__main__":
    main()
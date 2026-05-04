import os
import shutil
import numpy as np
import torch
import pytest
from torch.utils.data import DataLoader

from generator.skill_generator import SkillGenerator
from generator.losses import GeneratorLoss
from generator.train_generator import SkillDataset, train_one_epoch

def create_synthetic_dataset(data_dir: str, num_episodes: int = 50):
    """Generate synthetic .npz files for testing without the environment."""
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(42)
    for i in range(num_episodes):
        # Fake observations and targets
        obs = np.random.randn(8).astype(np.float32)
        # Make targets somewhat predictable from obs for testing learning
        payoff = float(np.sum(obs) * 0.5)
        motives = (obs[:2] * 2.0).astype(np.float32)
        
        filepath = os.path.join(data_dir, f"test_ep{i:03d}.npz")
        np.savez(
            filepath,
            obs=obs,
            payoff=payoff,
            motives=motives,
            skill_id=f"test_skill_{i}",
            terminated=True
        )

@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    """Sets up a temporary directory with synthetic data and patches train_generator paths."""
    data_dir = tmp_path / "data" / "raw"
    model_dir = tmp_path / "models"
    plot_dir = tmp_path / "plots"
    
    create_synthetic_dataset(str(data_dir), num_episodes=50)
    
    monkeypatch.setattr("generator.train_generator.DATA_DIR", str(data_dir))
    monkeypatch.setattr("generator.train_generator.MODEL_DIR", str(model_dir))
    monkeypatch.setattr("generator.train_generator.PLOT_DIR", str(plot_dir))
    monkeypatch.setattr("generator.train_generator.NUM_EPOCHS", 10)
    
    return tmp_path

def test_training_runs_without_error(temp_workspace):
    """Verify the full train() function runs and completes."""
    from generator.train_generator import train
    try:
        train()
    except Exception as e:
        pytest.fail(f"train() raised {type(e).__name__} unexpectedly: {str(e)}")

def test_loss_decreases_over_epochs(temp_workspace):
    """Train for a few epochs and ensure final loss is less than initial loss."""
    data_dir = os.path.join(temp_workspace, "data", "raw")
    dataset = SkillDataset(data_dir)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = SkillGenerator(input_dim=8, hidden_dim=32, motive_dim=2)
    loss_fn = GeneratorLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    initial_loss, _, _ = train_one_epoch(model, loader, optimizer, loss_fn)
    
    for _ in range(5):
        final_loss, _, _ = train_one_epoch(model, loader, optimizer, loss_fn)
        
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

def test_trained_model_beats_random(temp_workspace):
    """Verify a trained model predicts better than an untrained model on the same data."""
    data_dir = os.path.join(temp_workspace, "data", "raw")
    dataset = SkillDataset(data_dir)
    
    # We test on first 10 items
    loader = DataLoader(dataset, batch_size=10, shuffle=False)
    batch_obs, batch_payoff, batch_motives = next(iter(loader))
    
    # Random model
    random_model = SkillGenerator(input_dim=8, hidden_dim=32, motive_dim=2)
    loss_fn = GeneratorLoss()
    
    with torch.no_grad():
        rand_p, rand_m = random_model(batch_obs)
        random_loss = loss_fn(rand_p, rand_m, batch_payoff, batch_motives).item()
        
    # Trained model
    trained_model = SkillGenerator(input_dim=8, hidden_dim=32, motive_dim=2)
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=0.01)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for _ in range(10):
        train_one_epoch(trained_model, train_loader, optimizer, loss_fn)
        
    with torch.no_grad():
        trained_p, trained_m = trained_model(batch_obs)
        trained_loss = loss_fn(trained_p, trained_m, batch_payoff, batch_motives).item()
        
    assert trained_loss < random_loss, f"Trained {trained_loss:.4f} not better than Random {random_loss:.4f}"

def test_model_saves_and_loads(temp_workspace):
    """Verify saved models can be loaded and produce identical predictions."""
    model_path = os.path.join(temp_workspace, "test_model.pt")
    
    model = SkillGenerator()
    # Modify weights slightly to ensure it's not just checking zero state
    with torch.no_grad():
        for param in model.parameters():
            param.add_(1.0)
            
    model.save(model_path)
    
    loaded_model = SkillGenerator()
    loaded_model.load(model_path)
    
    test_obs = torch.randn(5, 8)
    
    with torch.no_grad():
        p1, m1 = model(test_obs)
        p2, m2 = loaded_model(test_obs)
        
    assert torch.allclose(p1, p2)
    assert torch.allclose(m1, m2)

def test_loss_plot_is_generated(temp_workspace):
    """Verify the plot gets saved correctly."""
    from generator.train_generator import train
    train()
    
    plot_file = os.path.join(temp_workspace, "plots", "generator_training.png")
    assert os.path.exists(plot_file), "Loss plot was not generated."
    assert os.path.getsize(plot_file) > 0, "Plot file is empty."

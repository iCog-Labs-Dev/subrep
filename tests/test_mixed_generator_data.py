import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from data_collector.collect_candidate_sets import (
    PpoThenSideTradeoff,
    PpoNoisyActions,
)
from data_collector.collect_mixed_generator_data import MixedGeneratorDataCollector
from generator.train_generator import SkillDataset


@pytest.fixture
def mixed_data_dir(tmp_path):
    save_dir = tmp_path / "mixed_data"
    yield str(save_dir)
    if save_dir.exists():
        shutil.rmtree(save_dir)


class MockPilot:
    """A tiny mock just to test stateful policy resets without loading heavy models."""
    def predict(self, obs, deterministic=True, return_probability=True):
        return 0, 1.0


def test_mixed_collector_creates_valid_npz_records(mixed_data_dir):
    collector = MixedGeneratorDataCollector(
        seed=42,
        save_dir=mixed_data_dir,
    )
    # Collect a tiny amount for testing (1 episode context)
    collector.collect(num_episodes=1)

    # With 9 candidates, running 1 context episode should yield 9 .npz files
    files = list(Path(mixed_data_dir).glob("*.npz"))
    assert len(files) == 9

    for file in files:
        data = np.load(file, allow_pickle=True)
        assert "obs" in data
        assert "payoff" in data
        assert "motives" in data
        assert "skill_id" in data
        assert "terminated" in data
        
        assert data["obs"].shape == (8,)
        assert data["motives"].shape == (2,)


def test_records_contain_finite_values(mixed_data_dir):
    collector = MixedGeneratorDataCollector(
        seed=42,
        save_dir=mixed_data_dir,
    )
    collector.collect(num_episodes=1)

    files = list(Path(mixed_data_dir).glob("*.npz"))
    for file in files:
        data = np.load(file, allow_pickle=True)
        
        assert np.all(np.isfinite(data["obs"]))
        assert np.isfinite(data["payoff"])
        assert np.all(np.isfinite(data["motives"]))


def test_stateful_policy_resets_per_episode():
    # Test PpoThenSideTradeoff
    mock_pilot = MockPilot()
    policy = PpoThenSideTradeoff(pilot=mock_pilot, switch_step=5, side_action=1)
    
    assert policy._step_counter == 0
    obs = np.zeros(8)
    
    for _ in range(3):
        policy(obs)
    
    assert policy._step_counter == 3
    
    # Simulate a new episode reset
    policy.reset()
    assert policy._step_counter == 0

    # Test PpoNoisyActions
    class MockActionSpace:
        n = 4
    
    noisy_policy = PpoNoisyActions(pilot=mock_pilot, noise_std=1.0, action_space=MockActionSpace())
    noisy_policy.reset(seed=123)
    val1 = noisy_policy.rng.random()
    noisy_policy.reset(seed=123)
    val2 = noisy_policy.rng.random()
    assert val1 == val2  # Prove the seed reset the RNG appropriately


def test_train_generator_loads_mixed_records(mixed_data_dir):
    os.makedirs(mixed_data_dir, exist_ok=True)
    
    # Write 3 mock .npz files imitating the collector output
    for i in range(3):
        filepath = Path(mixed_data_dir) / f"mock_{i}.npz"
        np.savez(
            filepath,
            obs=np.zeros(8, dtype=np.float32),
            payoff=10.0,
            motives=np.array([1.0, -1.0], dtype=np.float32),
            skill_id="mock_skill"
        )
    
    # Verify the SkillDataset inside train_generator can natively load them
    dataset = SkillDataset(mixed_data_dir)
    assert len(dataset) == 3
    
    obs, payoff, motives = dataset[0]
    assert isinstance(obs, torch.Tensor)
    assert isinstance(payoff, torch.Tensor)
    assert isinstance(motives, torch.Tensor)
    assert obs.shape == (8,)
    assert payoff.shape == (1,)
    assert motives.shape == (2,)

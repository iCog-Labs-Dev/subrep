import os
import shutil
import numpy as np
import pytest
from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor
from utils.data_collector import DataCollector

@pytest.fixture
def temp_data_dir(tmp_path):
    save_dir = tmp_path / "data" / "raw"
    yield str(save_dir)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

def test_collector_runs_without_error(temp_data_dir):
    """DataCollector.collect_n_episodes(3) completes without raising."""
    env = SubRepEnv(seed=42)
    policy = lambda obs: env.env.action_space.sample()
    executor = SkillExecutor(env=env, policy_fn=policy, gamma=0.99, max_steps=20)
    collector = DataCollector(executor=executor, seed=42, save_dir=temp_data_dir)
    
    records = collector.collect_n_episodes(3, print_summary=False)
    assert len(records) == 3

def test_saved_files_have_correct_keys(temp_data_dir):
    """
    Load each saved .npz and assert keys:
    {'obs', 'payoff', 'motives', 'skill_id', 'terminated'} all present.
    Assert shapes: obs=(8,), motives=(2,), payoff is scalar.
    """
    env = SubRepEnv(seed=42)
    policy = lambda obs: env.env.action_space.sample()
    executor = SkillExecutor(env=env, policy_fn=policy, gamma=0.99, max_steps=10)
    collector = DataCollector(executor=executor, seed=42, save_dir=temp_data_dir)
    
    collector.collect_n_episodes(1, print_summary=False, skill_prefix="test")
    
    saved_file = os.path.join(temp_data_dir, "test_ep001.npz")
    assert os.path.exists(saved_file)
    
    data = np.load(saved_file, allow_pickle=True)
    
    assert 'obs' in data
    assert 'payoff' in data
    assert 'motives' in data
    assert 'skill_id' in data
    assert 'terminated' in data
    
    assert data['obs'].shape == (8,)
    assert data['motives'].shape == (2,)
    assert np.isscalar(data['payoff'][()])

def test_summary_statistics_are_correct(temp_data_dir, capsys):
    """
    Collect 5 episodes. Manually compute mean payoff.
    Assert DataCollector's summary matches manual calculation (within 1e-5).
    """
    env = SubRepEnv(seed=42)
    policy = lambda obs: env.env.action_space.sample()
    executor = SkillExecutor(env=env, policy_fn=policy, gamma=0.99, max_steps=10)
    collector = DataCollector(executor=executor, seed=42, save_dir=temp_data_dir)
    
    records = collector.collect_n_episodes(5, print_summary=True)
    
    manual_mean_payoff = np.mean([r['payoff'] for r in records])
    
    captured = capsys.readouterr()
    assert f"Mean Payoff          : {manual_mean_payoff:.4f}" in captured.out

def test_seed_produces_consistent_results(temp_data_dir):
    """
    Run collect_n_episodes(5, seed=42) twice.
    Assert that saved payoff values are identical across both runs.
    """
    # Run 1
    env1 = SubRepEnv(seed=123)
    env1.env.action_space.seed(123)
    policy1 = lambda obs: env1.env.action_space.sample()
    executor1 = SkillExecutor(env=env1, policy_fn=policy1, gamma=0.99, max_steps=15)
    collector1 = DataCollector(executor=executor1, seed=123, save_dir=temp_data_dir)
    records1 = collector1.collect_n_episodes(2, print_summary=False)
    payoffs1 = [r['payoff'] for r in records1]
    
    # Run 2
    env2 = SubRepEnv(seed=123)
    env2.env.action_space.seed(123)
    policy2 = lambda obs: env2.env.action_space.sample()
    executor2 = SkillExecutor(env=env2, policy_fn=policy2, gamma=0.99, max_steps=15)
    collector2 = DataCollector(executor=executor2, seed=123, save_dir=temp_data_dir)
    records2 = collector2.collect_n_episodes(2, print_summary=False)
    payoffs2 = [r['payoff'] for r in records2]
    
    assert np.allclose(payoffs1, payoffs2)

def test_custom_prefix_prevents_overwriting(temp_data_dir):
    """
    Run 1 episode with prefix 'A'.
    Run 1 episode with prefix 'B'.
    Assert both files exist in the directory (proving 'B' didn't delete 'A').
    """
    env = SubRepEnv(seed=42)
    policy = lambda obs: env.env.action_space.sample()
    executor = SkillExecutor(env=env, policy_fn=policy, gamma=0.99, max_steps=10)
    collector = DataCollector(executor=executor, seed=42, save_dir=temp_data_dir)
    
    # Run prefix A
    collector.collect_n_episodes(1, print_summary=False, skill_prefix="A")
    # Run prefix B
    collector.collect_n_episodes(1, print_summary=False, skill_prefix="B")
    
    assert os.path.exists(os.path.join(temp_data_dir, "A_ep001.npz"))
    assert os.path.exists(os.path.join(temp_data_dir, "B_ep001.npz"))

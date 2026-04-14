"""
Environment Validation Test
Verifies that MO-LunarLander returns correct observation and reward shapes.
This test must pass before any Generator or Certification work begins.
"""

import numpy as np
from env.lunar_lander_wrapper import SubRepEnv

def test_env_structure():
    """Test observation and reward space shapes."""
    print("Testing Environment Structure...")
    env = SubRepEnv(seed=42)
    
    # Check Observation Space
    assert env.observation_space.shape == (8,), \
        f"Obs shape failed: expected (8,), got {env.observation_space.shape}"
    print("Observation space shape: (8,)")
    
    # Check Reward Space (SubRep's mapped 2D space)
    assert env.reward_space.shape == (2,), \
        f"Reward shape failed: expected (2,), got {env.reward_space.shape}"
    print("Reward space shape: (2,) [Safety, Fuel]")
    
    env.close()
    print("Structure tests passed.\n")

def test_env_execution():
    """Test step function and reward vector output."""
    print("Testing Environment Execution...")
    env = SubRepEnv(seed=42)
    obs, _ = env.reset()
    
    # Run 10 random steps
    for step in range(10):
        action = env.env.action_space.sample()
        obs, reward_vector, terminated, truncated, info = env.step(action)
        
        # Validate Observation
        assert isinstance(obs, np.ndarray), "Obs must be numpy array"
        assert obs.shape == (8,), f"Obs shape mismatch at step {step}"
        
        # Validate Reward Vector (Critical for SubRep)
        assert isinstance(reward_vector, np.ndarray), "Reward must be numpy array"
        assert reward_vector.shape == (2,), f"Reward shape mismatch at step {step}"
        assert np.isfinite(reward_vector).all(), "Reward contains NaN or Inf"
        
        # Print first step for manual verification
        if step == 0:
            print(f"   Step 0 Mapped Reward: {reward_vector} (Safety={reward_vector[0]:.2f}, Fuel={reward_vector[1]:.2f})")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step}")
            obs, _ = env.reset()
    
    env.close()
    print("Execution tests passed.\n")

if __name__ == "__main__":
    try:
        test_env_structure()
        test_env_execution()
        print("All Environment Tests Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
        raise
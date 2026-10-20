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

def test_raw_engine_costs_are_non_positive():
    """Engine usage is reported as a cost, never a gain.

    Pins the upstream contract `_map_rewards` relies on: if mo-gymnasium ever
    flipped these positive, the mapping would have to flip with it.
    """
    env = SubRepEnv(seed=42)
    try:
        env.reset(seed=42)
        for action in (0, 1, 2, 3):
            _, _, terminated, truncated, info = env.step(action)
            raw = info["raw_rewards"]
            assert raw.shape == (4,), f"expected 4 raw objectives, got {raw.shape}"
            assert raw[2] <= 0.0, f"main engine cost must be <= 0, got {raw[2]}"
            assert raw[3] <= 0.0, f"side engine cost must be <= 0, got {raw[3]}"
            if terminated or truncated:
                env.reset(seed=42)
    finally:
        env.close()


def test_fuel_objective_prefers_less_fuel():
    """Burning less fuel must score higher on the fuel objective.

    A sign error here does not crash; it silently turns fuel waste into an
    admission benefit inside `min_i(delta_n_i)`.
    """
    steps = 60

    def total_fuel_objective(action: int) -> float:
        env = SubRepEnv(seed=42)
        try:
            env.reset(seed=42)
            total = 0.0
            for _ in range(steps):
                _, reward_vector, terminated, truncated, _ = env.step(action)
                total += float(reward_vector[1])
                if terminated or truncated:
                    break
            return total
        finally:
            env.close()

    idle_fuel = total_fuel_objective(0)        # noop: fires no engine
    burning_fuel = total_fuel_objective(2)     # main engine at full power

    assert idle_fuel == 0.0, f"noop must burn nothing, got {idle_fuel}"
    assert burning_fuel < idle_fuel, (
        f"fuel objective is inverted: burning={burning_fuel} >= idle={idle_fuel}"
    )


def test_map_rewards_passes_engine_costs_through_unchanged():
    """Unit-level check of the mapping arithmetic, independent of the sim."""
    env = SubRepEnv(seed=42)
    try:
        raw = np.array([100.0, 5.0, -1.0, -0.5], dtype=np.float32)
        safety, fuel = env._map_rewards(raw)
        assert safety == 105.0, f"safety must be raw[0] + raw[1], got {safety}"
        assert fuel == -1.5, f"fuel must be raw[2] + raw[3], got {fuel}"

        # A no-engine step is the best attainable fuel score, not the worst.
        _, idle_fuel = env._map_rewards(np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        assert idle_fuel > fuel, "not firing an engine must beat firing one"
    finally:
        env.close()


if __name__ == "__main__":
    try:
        test_env_structure()
        test_env_execution()
        print("All Environment Tests Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
        raise
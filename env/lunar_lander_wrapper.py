"""
MO-LunarLander Environment Wrapper for SubRep.

This wrapper standardizes the MO-Gymnasium interface to ensure consistent 
vector reward output (Safety, Fuel) required for CDS/PDS certification.

"""

import numpy as np
import mo_gymnasium as mo_gym
from gymnasium.spaces import Box  
class SubRepEnv:
    """
    Wraps mo-lunar-lander-v3 to enforce SubRep reward structure.
    
    Attributes:
        env: The underlying mo-gymnasium environment.
        observation_space: Shape (8,) state vector.
        reward_space: Shape (2,) vector [Safety, Fuel].
    """
    
    def __init__(self, seed: int = 42, render_mode: str = None):
        """Initialize the environment."""
        # Create MO-LunarLander environment (wrapped with TimeLimit by default)
        self.env = mo_gym.make('mo-lunar-lander-v3', render_mode=render_mode)
        self.env.reset(seed=seed)
        
        # Access the unwrapped base environment
        base_env = self.env.unwrapped
        
        # Validate observation space
        assert base_env.observation_space.shape == (8,), \
            f"Expected obs shape (8,), got {base_env.observation_space.shape}"
        
        # MO-LunarLander-v3 returns 4 objectives:
        # [0] Shaping reward, [1] Landing success, [2] Fuel usage, [3] Crash penalty
        # We map these to SubRep's 2 objectives: [Safety, Fuel]
        assert base_env.reward_space.shape[0] == 4, \
            f"Expected 4 raw objectives from MO-LunarLander, got {base_env.reward_space.shape[0]}"
        
        # Store base spaces (for reference)
        self._base_observation_space = base_env.observation_space
        self._base_reward_space = base_env.reward_space
        
        # Define SubRep's 2D reward space (Safety, Fuel)
        self.observation_space = base_env.observation_space
        self.reward_space = Box( 
            low=np.array([-10.0, -10.0], dtype=np.float32),
            high=np.array([10.0, 10.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        self.seed = seed

    def _map_rewards(self, raw_rewards: np.ndarray) -> np.ndarray:
        """
        Map 4 raw MO-LunarLander rewards → 2 SubRep objectives.
        
        Raw rewards (index):
          [0] Shaping (potential-based guidance)
          [1] Landing success (1 if landed safely, 0 otherwise)
          [2] Fuel usage (negative value, more negative = more fuel used)
          [3] Crash penalty (negative if crashed)
        
        SubRep objectives:
          [0] Safety = Landing success + Crash penalty (combined)
          [1] Fuel = Fuel usage (inverted so positive = good)
        """
        safety = raw_rewards[1] + raw_rewards[3]  # Landing + Crash
        fuel = -raw_rewards[2]                     # Invert: positive = fuel saved
        return np.array([safety, fuel], dtype=np.float32)

    def reset(self):
        """Reset the environment and return initial observation."""
        obs, info = self.env.reset(seed=self.seed)
        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        obs, raw_rewards, terminated, truncated, info = self.env.step(action)
        
        # Map 4 raw rewards → 2 SubRep objectives
        reward_vector = self._map_rewards(np.array(raw_rewards, dtype=np.float32))
        
        # Validate reward shape at runtime (Safety check)
        if reward_vector.shape != (2,):
            raise ValueError(f"Reward vector shape mismatch: expected (2,), got {reward_vector.shape}")
            
        return obs, reward_vector, terminated, truncated, info

    def close(self):
        """Close the environment."""
        self.env.close()
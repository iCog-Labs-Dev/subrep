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
        # [0] Terminal result, [1] dense shaping, [2] main engine cost, [3] side engine cost
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
          [0] Terminal result reward (+100 if landed, -100 if crashed)
          [1] Shaping reward (potential-based guidance)
          [2] Main engine usage cost (negative)
          [3] Side engine usage cost (negative)
        
        SubRep objectives:
          [0] Safety = Terminal result + dense shaping
          [1] Fuel = Engine costs inverted so positive = fuel saved
        """
        # Safety should reflect both terminal outcome and dense flight
        # progress. Fuel is inverted because MO-LunarLander reports engine
        # usage as negative costs while SubRep motives are better when larger.
        safety = raw_rewards[0] + raw_rewards[1]
        fuel = -(raw_rewards[2] + raw_rewards[3])
        return np.array([safety, fuel], dtype=np.float32)

    def reset(self, seed=None):
        """Reset the environment and return initial observation."""
        # Only update the stored seed when the caller explicitly provides one.
        # Otherwise Gym continues from its current RNG stream, so repeated
        # `reset()` calls are stochastic instead of replaying one start state.
        if seed is not None:
            self.seed = int(seed)
            obs, info = self.env.reset(seed=self.seed)
        else:
            obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        obs, raw_rewards, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        raw_rewards = np.array(raw_rewards, dtype=np.float32)
        
        # Preserve the raw vector for PPO reward shaping and strict success
        # checks, then expose the mapped 2D vector for SubRep certification.
        reward_vector = self._map_rewards(raw_rewards)
        info["raw_rewards"] = raw_rewards.copy()
        info["terminal_reward"] = float(raw_rewards[0])
        info["shaping_reward"] = float(raw_rewards[1])
        # A real landing is a terminal, non-truncated episode with +100 result.
        # Positive dense shaping alone must not count as success.
        info["landing_success"] = bool(terminated and not truncated and raw_rewards[0] >= 100.0)
        info["crashed"] = bool(terminated and raw_rewards[0] <= -100.0)
        info["main_engine_cost"] = float(raw_rewards[2])
        info["side_engine_cost"] = float(raw_rewards[3])
        info["fuel_usage"] = float(raw_rewards[2] + raw_rewards[3])
        info["subrep_reward"] = reward_vector.copy()
        
        # Validate reward shape at runtime (Safety check)
        if reward_vector.shape != (2,):
            raise ValueError(f"Reward vector shape mismatch: expected (2,), got {reward_vector.shape}")
            
        return obs, reward_vector, terminated, truncated, info

    def close(self):
        """Close the environment."""
        self.env.close()

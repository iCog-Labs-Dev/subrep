from village_sim.env import VillageEnv
from village_sim.motives import MOTIVE_NAMES
import numpy as np

def run_policy(policy_fn, seed=None, max_steps=100, gamma=0.99):
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    # Create the env without seeding — seed is passed exclusively to reset()
    # so it is set exactly once and the RNG state is deterministic from there.
    env = VillageEnv(seed=None)
    state, _ = env.reset(seed=seed)
    total_payoff = 0.0
    total_motives = np.zeros(len(MOTIVE_NAMES), dtype=np.float32)

    for t in range(max_steps):
        action = policy_fn(state)
        state, motives, terminated, truncated, info = env.step(action)
        discount = gamma ** t
        # Scalar task payoff and motive vector are accumulated
        # independently: delta_r comes only from the scalar task payoff,
        # never from motives.sum(), so delta_r != sum(delta_n) in general.
        total_payoff += discount * float(info.get("task_payoff", info.get("task_reward", 0.0)))
        total_motives += discount * motives
        if terminated or truncated:
            break

    return total_payoff, total_motives


def idle_policy(state):
    return "idle"
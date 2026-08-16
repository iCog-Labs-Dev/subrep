from village_sim.env import VillageEnv
import numpy as np

def run_policy(policy_fn, seed=None, max_steps=100, gamma=0.99):
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    env = VillageEnv(seed=seed)
    state = env.reset()
    total_payoff = 0.0
    total_motives = np.zeros(6, dtype=np.float32)

    for t in range(max_steps):
        action = policy_fn(state)
        state, motives, done, info = env.step(action)
        discount = gamma ** t
        # Scalar task payoff and motive vector are accumulated
        # independently: delta_r comes only from the scalar task reward,
        # never from motives.sum(), so delta_r != sum(delta_n) in general.
        total_payoff += discount * float(info["task_reward"])
        total_motives += discount * motives
        if done:
            break

    return total_payoff, total_motives


def idle_policy(state):
    return "idle"
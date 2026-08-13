from village_sim.env import VillageEnv
import numpy as np

def run_policy(policy_fn, seed=None, max_steps=100):
    env = VillageEnv(seed=seed)
    state = env.reset()
    total_payoff = 0.0
    total_motives = np.zeros(6, dtype=np.float32)
    gamma = 0.99

    for t in range(max_steps):
        action = policy_fn(state)
        state, motives, done, info = env.step(action)
        total_payoff += (gamma ** t) * 0.0
        total_motives += (gamma ** t) * motives
        if done:
            break

    return total_payoff, total_motives


def idle_policy(state):
    return "idle"
import numpy as np
from data_collector import run_filtered_candidates as live

def test_live_baseline_uses_candidate_rollout_settings(monkeypatch):
    captured = {}

    class FakeIdlePolicy:
        def __init__(self, env, gamma):
            captured["gamma"] = gamma

        def run_baseline_episodes(self, num_episodes, seed, max_steps):
            captured["episodes"] = num_episodes
            captured["seed"] = seed
            captured["max_steps"] = max_steps
            return {
                "baseline_payoff": 1.0,
                "baseline_motives": np.array([0.5, -0.5]),
            }

    monkeypatch.setattr(live, "IdlePolicy", FakeIdlePolicy)
    monkeypatch.setattr(live, "SubRepEnv", lambda seed: object())

    live.measure_idle_baseline(
        episodes=20,
        seed=42,
        gamma=0.8,
        max_steps=100,
    )

    assert captured == {
        "episodes": 20,
        "seed": 42,
        "gamma": 0.8,
        "max_steps": 100,
    }
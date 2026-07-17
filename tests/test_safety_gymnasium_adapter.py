from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium.spaces import Box

from data_collector.collect_safety_gymnasium_rollouts import SafetyGymnasiumRolloutCollector
from env.safety_gymnasium_wrapper import SafeRLGymnasiumEnv


class FakeSafetyEnv:
    def __init__(self) -> None:
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )
        self._step_count = 0
        self._seed = 0

    def reset(self, seed=None):
        self._step_count = 0
        if seed is not None:
            self._seed = int(seed)
        obs = np.full(4, float(self._seed % 10), dtype=np.float32)
        return obs, {"seed": self._seed}

    def step(self, action):
        self._step_count += 1
        action = np.asarray(action, dtype=np.float32)
        obs = np.full(4, float(self._seed % 10) + self._step_count, dtype=np.float32)
        reward = 1.0 - 0.1 * float(np.linalg.norm(action))
        cost = 0.25 * float(np.any(np.abs(action) > 0.5))
        terminated = False
        truncated = self._step_count >= 3
        return obs, reward, cost, terminated, truncated, {"fake": True}

    def close(self):
        return None


def _make_fake_env(_env_id, render_mode=None):
    return FakeSafetyEnv()


def _fake_wrapper_factory(env_id: str, seed: int):
    return SafeRLGymnasiumEnv(env_id=env_id, seed=seed, make_env=_make_fake_env)


def test_safety_wrapper_maps_reward_and_cost_to_subrep_motives():
    env = SafeRLGymnasiumEnv(
        env_id="FakeSafety-v0",
        seed=7,
        make_env=_make_fake_env,
    )

    obs, _ = env.reset(seed=7)
    assert obs.shape == (4,)

    _, reward_vector, _, _, info = env.step(np.array([0.8, 0.0], dtype=np.float32))

    assert reward_vector.shape == (2,)
    assert reward_vector[0] == -0.25
    assert reward_vector[1] < 1.0
    assert info["safety_cost"] == 0.25
    assert np.isclose(info["task_reward"], float(reward_vector[1]))


def test_safety_rollout_collector_saves_payoff_cost_and_motive_returns(tmp_path):
    collector = SafetyGymnasiumRolloutCollector(
        env_id="FakeSafety-v0",
        seed=42,
        save_dir=str(tmp_path),
        max_steps=3,
        env_factory=_fake_wrapper_factory,
    )

    records = collector.collect(1, prefix="fake_safety")
    collector.close()

    assert len(records) == 1
    record = records[0]
    assert record["candidate_motives"].shape == (7, 2)
    assert record["candidate_payoffs"].shape == (7,)
    assert record["candidate_safety_costs"].shape == (7,)
    assert record["candidate_task_returns"].shape == (7,)
    assert np.all(np.isfinite(record["candidate_motives"]))
    assert np.all(record["candidate_safety_costs"] >= 0.0)

    saved_path = Path(tmp_path) / "fake_safety_00001.npz"
    assert saved_path.exists()
    saved = np.load(saved_path, allow_pickle=True)
    for key in (
        "env_id",
        "context",
        "candidate_skill_ids",
        "candidate_payoffs",
        "candidate_motives",
        "candidate_safety_costs",
        "candidate_task_returns",
    ):
        assert key in saved

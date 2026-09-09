"""Contract tests for SubRepBaseEnv conforming environments."""

from __future__ import annotations

import numpy as np
import pytest

from env.base_env import SubRepBaseEnv, validate_env_metadata
from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor
from village_sim.env import VillageEnv


@pytest.fixture(params=["village", "lunar"])
def env_instance(request):
    """Parametrized fixture providing both standard environments."""
    if request.param == "village":
        env = VillageEnv(seed=42)
        yield env
        env.close()
    elif request.param == "lunar":
        env = SubRepEnv(seed=42)
        yield env
        env.close()


def _get_sample_action(env):
    """Return a valid action for the given environment."""
    if isinstance(env, VillageEnv):
        return "idle"
    return 0


def test_metadata_keys_present(env_instance):
    """Environment metadata contains all required keys with valid types."""
    meta = env_instance.metadata
    assert isinstance(meta, dict)
    validate_env_metadata(meta)


def test_reset_returns_two_tuple(env_instance):
    """reset() returns (observation, info) 2-tuple."""
    result = env_instance.reset(seed=123)
    assert isinstance(result, tuple)
    assert len(result) == 2
    obs, info = result
    assert obs is not None
    assert isinstance(info, dict)


def test_step_returns_five_tuple(env_instance):
    """step() returns (obs, motives, terminated, truncated, info) 5-tuple."""
    env_instance.reset(seed=123)
    action = _get_sample_action(env_instance)
    result = env_instance.step(action)
    assert isinstance(result, tuple)
    assert len(result) == 5
    obs, motives, terminated, truncated, info = result
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_task_payoff_present_in_info(env_instance):
    """step() info dict contains a finite scalar task_payoff float."""
    env_instance.reset(seed=123)
    action = _get_sample_action(env_instance)
    _, _, _, _, info = env_instance.step(action)
    assert "task_payoff" in info
    payoff = info["task_payoff"]
    assert isinstance(payoff, (int, float, np.floating))
    assert np.isfinite(payoff)


def test_motive_vector_matches_motive_names(env_instance):
    """Motive vector length strictly equals length of metadata['motive_names']."""
    env_instance.reset(seed=123)
    action = _get_sample_action(env_instance)
    _, motives, _, _, _ = env_instance.step(action)
    expected_dim = len(env_instance.metadata["motive_names"])
    assert len(motives) == expected_dim


def test_motive_vector_is_1d_finite_float32(env_instance):
    """Motive return vector is 1D, float32, and contains finite values."""
    env_instance.reset(seed=123)
    action = _get_sample_action(env_instance)
    _, motives, _, _, _ = env_instance.step(action)
    assert isinstance(motives, np.ndarray)
    assert motives.ndim == 1
    assert motives.dtype == np.float32
    assert np.all(np.isfinite(motives))


def test_close_does_not_raise(env_instance):
    """close() can be called safely without raising."""
    env_instance.close()


def test_village_env_satisfies_base_protocol():
    """VillageEnv satisfies runtime_checkable SubRepBaseEnv protocol."""
    env = VillageEnv()
    assert isinstance(env, SubRepBaseEnv)
    env.close()


def test_subrep_env_satisfies_base_protocol():
    """SubRepEnv satisfies runtime_checkable SubRepBaseEnv protocol."""
    env = SubRepEnv()
    assert isinstance(env, SubRepBaseEnv)
    env.close()


def test_validate_env_metadata_rejects_invalid():
    """validate_env_metadata raises ValueError on incomplete or malformed dicts."""
    with pytest.raises(ValueError, match="metadata must be a dict"):
        validate_env_metadata("not a dict")

    with pytest.raises(ValueError, match="Missing required metadata key"):
        validate_env_metadata({"environment_id": "test"})

    with pytest.raises(ValueError, match="must be of type"):
        validate_env_metadata({
            "environment_id": 123,  # wrong type
            "motive_names": ["a"],
            "motive_schema_version": "1.0.0",
            "payoff_schema_version": "1.0.0",
            "observation_schema_version": "1.0.0",
            "action_schema_version": "1.0.0",
        })

    with pytest.raises(ValueError, match="non-empty"):
        validate_env_metadata({
            "environment_id": "test",
            "motive_names": [],  # empty list
            "motive_schema_version": "1.0.0",
            "payoff_schema_version": "1.0.0",
            "observation_schema_version": "1.0.0",
            "action_schema_version": "1.0.0",
        })

    with pytest.raises(ValueError, match="non-empty strings"):
        validate_env_metadata({
            "environment_id": "test",
            "motive_names": [""],  # empty string entry
            "motive_schema_version": "1.0.0",
            "payoff_schema_version": "1.0.0",
            "observation_schema_version": "1.0.0",
            "action_schema_version": "1.0.0",
        })


def test_executor_works_on_village_env():
    """SkillExecutor runs full rollout on VillageEnv and outputs 6D motives."""
    env = VillageEnv(seed=1)
    policy = lambda state: "idle"
    executor = SkillExecutor(env=env, policy_fn=policy, max_steps=10)
    payoff, motives, terminated = executor.run_episode()

    assert np.isscalar(payoff)
    assert motives.shape == (6,)
    assert executor.last_run_info["motive_names"] == env.metadata["motive_names"]
    env.close()


def test_executor_works_on_lunar_env():
    """SkillExecutor runs full rollout on SubRepEnv and outputs 2D motives."""
    env = SubRepEnv(seed=1)
    policy = lambda obs: 0
    executor = SkillExecutor(env=env, policy_fn=policy, max_steps=10)
    payoff, motives, terminated = executor.run_episode()

    assert np.isscalar(payoff)
    assert motives.shape == (2,)
    assert executor.last_run_info["motive_names"] == ["Safety", "Fuel"]
    env.close()


class _CustomPayoffEnv:
    def __init__(self):
        self._step = 0
        self.metadata = {
            "environment_id": "custom_payoff_env",
            "motive_names": ["m1", "m2"],
            "motive_schema_version": "1.0.0",
            "payoff_schema_version": "1.0.0",
            "observation_schema_version": "1.0.0",
            "action_schema_version": "1.0.0",
        }

    def reset(self, seed=None):
        self._step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.zeros(4, dtype=np.float32)
        reward = np.array([1.0, 1.0], dtype=np.float32)  # sum is 2.0
        terminated = self._step >= 2
        truncated = False
        info = {"task_payoff": 50.0}  # explicitly different from sum(reward)
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


def test_executor_uses_task_payoff_key():
    """SkillExecutor accumulates info['task_payoff'] rather than sum(reward)."""
    env = _CustomPayoffEnv()
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, gamma=1.0)
    payoff, motives, _ = executor.run_episode()

    # 2 steps at 50.0 = 100.0 (not 2 * 2.0 = 4.0)
    assert payoff == 100.0


class _LegacyEnvWithoutPayoff:
    def __init__(self):
        self._step = 0

    def reset(self, seed=None):
        self._step = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.zeros(4, dtype=np.float32)
        reward = np.array([3.0, 4.0], dtype=np.float32)
        terminated = self._step >= 2
        truncated = False
        info = {}  # missing task_payoff
        return obs, reward, terminated, truncated, info

    def close(self):
        pass


def test_executor_falls_back_when_task_payoff_missing():
    """SkillExecutor falls back to payoff_fn with warning when task_payoff is absent."""
    env = _LegacyEnvWithoutPayoff()
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, gamma=1.0)
    with pytest.warns(RuntimeWarning, match="missing 'task_payoff'"):
        payoff, motives, _ = executor.run_episode()

    # Fallback to sum([3.0, 4.0]) * 2 = 14.0
    assert payoff == 14.0

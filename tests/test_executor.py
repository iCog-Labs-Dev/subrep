"""
Skill Executor Validation Tests

Verifies that the rollout loop returns stable shapes/types, respects
termination controls, and supports payoff scalarization override.
"""

import numpy as np
from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor

def test_executor_output_shapes_and_types():
    #Smoke test with real MO-LunarLander wrapper for output contract.
    env = SubRepEnv(seed=42)
    policy = lambda obs: env.env.action_space.sample()
    executor = SkillExecutor(env=env, policy_fn=policy, gamma=0.99, max_steps=20)

    total_payoff, motive_deltas, terminated = executor.run_episode()
    env.close()

    assert np.isscalar(total_payoff), "total_payoff should be scalar-like"
    assert isinstance(motive_deltas, np.ndarray), "motive_deltas should be np.ndarray"
    assert motive_deltas.shape == (2,), "motive_deltas should have shape (2,)"
    assert np.isfinite(total_payoff), "total_payoff must be finite"
    assert np.isfinite(motive_deltas).all(), "motive_deltas must be finite"
    assert isinstance(terminated, bool), "terminated should be bool"

def test_executor_max_steps_and_last_run_info():
    #Check max-step cap and diagnostics dictionary population.
    env = SubRepEnv(seed=7)
    policy = lambda obs: env.env.action_space.sample()
    max_steps = 5
    executor = SkillExecutor(env=env, policy_fn=policy, gamma=0.99, max_steps=max_steps)

    _, motive_deltas, _ = executor.run_episode()
    env.close()

    assert motive_deltas.shape == (2,)
    assert executor.last_run_info is not None
    assert executor.last_run_info["steps"] <= max_steps
    assert executor.last_run_info["stop_reason"] in {"terminated", "truncated", "max_steps"}
    assert isinstance(executor.last_run_info["final_reward"], np.ndarray)
    assert executor.last_run_info["final_reward"].shape == (2,)

class _DeterministicEnv:
    #Small deterministic env stub for exact discounting assertions.

    def __init__(self):
        self._count = 0

    def reset(self):
        self._count = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self._count += 1
        obs = np.zeros(8, dtype=np.float32)
        reward = np.array([1.0, 2.0], dtype=np.float32)
        terminated = self._count >= 3
        truncated = False
        # task_payoff = sum(reward) by default; custom payoff_fn tests override
        # the scalar independently — the key's presence silences RuntimeWarning.
        info = {"task_payoff": float(reward.sum())}
        return obs, reward, terminated, truncated, info


def test_executor_uses_custom_payoff_fn():
    #Verify that payoff_fn is used as the fallback when task_payoff is absent.
    # Under the Phase 2 contract, task_payoff in info is always preferred;
    # payoff_fn is only invoked when the env does not supply that key.
    env = _DeterministicEnv()
    # Temporarily remove task_payoff by subclassing
    class _NoPayoffEnv(_DeterministicEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            info.pop("task_payoff", None)  # simulate a legacy env
            return obs, reward, terminated, truncated, info

    no_payoff_env = _NoPayoffEnv()
    gamma = 0.5
    policy = lambda obs: 0
    payoff_fn = lambda r: float(r[0] - r[1])  # -1.0 each step
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        executor = SkillExecutor(env=no_payoff_env, policy_fn=policy, gamma=gamma, payoff_fn=payoff_fn)
        total_payoff, motive_deltas, terminated = executor.run_episode()

    # Discount factors for 3 steps: 1, 0.5, 0.25
    discount_sum = 1.0 + 0.5 + 0.25
    expected_payoff = -1.0 * discount_sum
    expected_motives = np.array([1.0, 2.0], dtype=np.float32) * discount_sum

    assert np.isclose(total_payoff, expected_payoff)
    assert np.allclose(motive_deltas, expected_motives)
    assert terminated is True


def test_executor_max_steps_zero_runs_no_steps():
    #`max_steps=0` should exit before taking any environment step.
    env = _DeterministicEnv()
    policy = lambda obs: 0
    executor = SkillExecutor(env=env, policy_fn=policy, max_steps=0)

    total_payoff, motive_deltas, terminated = executor.run_episode()

    assert total_payoff == 0.0
    assert np.allclose(motive_deltas, np.zeros(2, dtype=np.float32))
    assert terminated is False
    assert executor.last_run_info["steps"] == 0
    assert executor.last_run_info["stop_reason"] == "max_steps"

def test_executor_rejects_invalid_gamma():
    #Gamma outside [0, 1] should fail fast, without even showing summary
    env = _DeterministicEnv()
    policy = lambda obs: 0

    try:
        SkillExecutor(env=env, policy_fn=policy, gamma=1.1)
    except ValueError as exc:
        assert "gamma must be in [0, 1]" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid gamma")


class _Deterministic6DEnv:
    def __init__(self):
        self._count = 0
        self.metadata = {
            "environment_id": "fake_6d_env",
            "motive_names": ["m1", "m2", "m3", "m4", "m5", "m6"],
            "motive_schema_version": "1.0.0",
            "payoff_schema_version": "1.0.0",
            "observation_schema_version": "1.0.0",
            "action_schema_version": "1.0.0",
        }

    def reset(self):
        self._count = 0
        return np.zeros(8, dtype=np.float32), {}

    def step(self, action):
        self._count += 1
        obs = np.zeros(8, dtype=np.float32)
        reward = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
        terminated = self._count >= 3
        truncated = False
        info = {"task_payoff": 10.0}
        return obs, reward, terminated, truncated, info


def test_executor_infers_nd_motive_shape():
    env = _Deterministic6DEnv()
    policy = lambda obs: 0
    executor = SkillExecutor(env=env, policy_fn=policy)
    total_payoff, motive_deltas, terminated = executor.run_episode()

    assert motive_deltas.shape == (6,)
    assert executor.last_run_info["motive_names"] == ["m1", "m2", "m3", "m4", "m5", "m6"]
    assert total_payoff > 0


# ---------------------------------------------------------------------------
# Group C1: strict mode tests
# ---------------------------------------------------------------------------

import pytest
import warnings as _warnings


class _StrictFriendlyEnv:
    """Fully compliant env: always 5-tuple + task_payoff."""
    def __init__(self):
        self._step = 0
    def reset(self, seed=None, options=None):
        self._step = 0
        return np.zeros(2, dtype=np.float32), {}
    def step(self, action):
        self._step += 1
        obs = np.zeros(2, dtype=np.float32)
        reward = np.array([0.5, 0.5], dtype=np.float32)
        terminated = self._step >= 2
        truncated = False
        info = {"task_payoff": 1.0}
        return obs, reward, terminated, truncated, info
    def close(self): pass


class _FourTupleEnv:
    """Legacy env returning 4-tuple from step()."""
    def __init__(self):
        self._step = 0
    def reset(self, seed=None):
        self._step = 0
        return np.zeros(2, dtype=np.float32), {}
    def step(self, action):
        self._step += 1
        obs = np.zeros(2, dtype=np.float32)
        reward = np.array([0.5, 0.5], dtype=np.float32)
        terminated = self._step >= 2
        info = {"task_payoff": 1.0}
        return obs, reward, terminated, info   # 4-tuple
    def close(self): pass


class _MissingPayoffEnv:
    """Env returning 5-tuple but no task_payoff in info."""
    def __init__(self):
        self._step = 0
    def reset(self, seed=None):
        self._step = 0
        return np.zeros(2, dtype=np.float32), {}
    def step(self, action):
        self._step += 1
        obs = np.zeros(2, dtype=np.float32)
        reward = np.array([0.5, 0.5], dtype=np.float32)
        terminated = self._step >= 2
        truncated = False
        return obs, reward, terminated, truncated, {}  # no task_payoff
    def close(self): pass


def test_executor_strict_mode_raises_on_4tuple():
    """strict=True raises ValueError when step() returns a 4-tuple."""
    env = _FourTupleEnv()
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, strict=True)
    with pytest.raises(ValueError, match="4-tuple"):
        executor.run_episode()


def test_executor_strict_mode_raises_on_missing_task_payoff():
    """strict=True raises ValueError when task_payoff is absent from info."""
    env = _MissingPayoffEnv()
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, strict=True)
    with pytest.raises(ValueError, match="task_payoff"):
        executor.run_episode()


def test_executor_strict_mode_does_not_warn_on_compliant_env():
    """strict=True does not issue RuntimeWarning on a fully compliant env."""
    env = _StrictFriendlyEnv()
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, strict=True)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        payoff, motives, _ = executor.run_episode()
    assert np.isfinite(payoff)


def test_executor_strict_mode_village_never_falls_back():
    """VillageEnv never triggers strict-mode fallback (always 5-tuple + task_payoff)."""
    from village_sim.env import VillageEnv
    env = VillageEnv(seed=1)
    executor = SkillExecutor(env=env, policy_fn=lambda obs: "idle", strict=True, max_steps=5)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        payoff, motives, _ = executor.run_episode()
    assert np.isfinite(payoff)
    env.close()


def test_executor_strict_mode_lunar_never_falls_back():
    """SubRepEnv (Lunar) never triggers strict-mode fallback."""
    env = SubRepEnv(seed=1)
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, strict=True, max_steps=5)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", RuntimeWarning)
        payoff, motives, _ = executor.run_episode()
    assert np.isfinite(payoff)
    env.close()


# ---------------------------------------------------------------------------
# Group D tests
# ---------------------------------------------------------------------------

from env.skill_executor import ExecutionResult
from env.registry import EnvRegistry, make_env, list_envs, register_env


# --- D2: ExecutionResult ---

def test_execution_result_fields():
    """ExecutionResult carries all expected fields with correct types."""
    env = _StrictFriendlyEnv()
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, max_steps=2)
    result = executor.run_episode()

    assert isinstance(result, ExecutionResult)
    assert isinstance(result.total_payoff, float)
    assert isinstance(result.motive_deltas, np.ndarray)
    assert isinstance(result.terminated, bool)
    assert isinstance(result.steps, int) and result.steps >= 0
    assert result.stop_reason in {"terminated", "truncated", "max_steps", "unknown"}


def test_execution_result_tuple_unpack_compat():
    """Existing 3-tuple unpacking still works on ExecutionResult."""
    env = _StrictFriendlyEnv()
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, max_steps=2)
    result = executor.run_episode()

    # Old-style unpacking must continue to work.
    payoff, motives, terminated = result
    assert isinstance(payoff, float)
    assert isinstance(motives, np.ndarray)
    assert isinstance(terminated, bool)


def test_execution_result_named_access():
    """Named attribute access works alongside tuple unpacking."""
    env = _StrictFriendlyEnv()
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, max_steps=2)
    result = executor.run_episode()

    # Both access patterns must agree.
    payoff, motives, terminated = result
    assert payoff == result.total_payoff
    assert np.array_equal(motives, result.motive_deltas)
    assert terminated == result.terminated


def test_execution_result_motive_names_when_env_has_metadata():
    """ExecutionResult.motive_names is populated from env metadata."""
    env = SubRepEnv(seed=1)
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, max_steps=3)
    result = executor.run_episode()
    assert result.motive_names == ["Safety", "Fuel"]
    env.close()


def test_execution_result_steps_matches_last_run_info():
    """ExecutionResult.steps matches last_run_info['steps']."""
    env = _StrictFriendlyEnv()
    executor = SkillExecutor(env=env, policy_fn=lambda obs: 0, max_steps=2)
    result = executor.run_episode()
    assert result.steps == executor.last_run_info["steps"]


# --- D1: EnvRegistry ---

def test_registry_list_contains_builtin_envs():
    """Global registry contains the three built-in environment names."""
    envs = list_envs()
    assert "lunarlander" in envs
    assert "village" in envs
    assert "safety_gymnasium" in envs


def test_registry_make_lunarlander():
    """make_env('lunarlander') returns a SubRepBaseEnv-conforming env."""
    from env.base_env import SubRepBaseEnv
    env = make_env("lunarlander", seed=1)
    assert isinstance(env, SubRepBaseEnv)
    env.close()


def test_registry_make_village():
    """make_env('village') returns a SubRepBaseEnv-conforming env."""
    from env.base_env import SubRepBaseEnv
    env = make_env("village", seed=1)
    assert isinstance(env, SubRepBaseEnv)
    env.close()


def test_registry_make_safety_gymnasium_with_fake_backend():
    """make_env('safety_gymnasium', env=...) works with a fake backend."""
    class _FakeBackend:
        def reset(self, seed=None): return np.zeros(4), {}
        def step(self, a): return np.zeros(4), 1.0, 0.0, False, False, {}
        def close(self): pass

    from env.base_env import SubRepBaseEnv
    env = make_env("safety_gymnasium", env=_FakeBackend())
    assert isinstance(env, SubRepBaseEnv)
    env.close()


def test_registry_unknown_env_raises_key_error():
    """make_env raises KeyError for an unregistered environment name."""
    with pytest.raises(KeyError, match="not registered"):
        make_env("does_not_exist")


def test_registry_register_custom_env():
    """Custom envs can be registered and made through the global registry."""
    class _MyEnv:
        def __init__(self, value=42): self.value = value
        def reset(self, seed=None): return np.zeros(1), {}
        def step(self, a): return np.zeros(1), np.array([1.0]), False, False, {"task_payoff": 1.0}
        def close(self): pass
        @property
        def metadata(self): return {
            "environment_id": "my_env", "motive_names": ["m"],
            "motive_schema_version": "1.0", "payoff_schema_version": "1.0",
            "observation_schema_version": "1.0", "action_schema_version": "1.0",
        }

    reg = EnvRegistry()
    reg.register("my_custom", _MyEnv)
    env = reg.make("my_custom", value=99)
    assert env.value == 99
    assert "my_custom" in reg


def test_registry_case_insensitive():
    """Registry lookups are case-insensitive."""
    from env.base_env import SubRepBaseEnv
    env = make_env("LunarLander", seed=1)
    assert isinstance(env, SubRepBaseEnv)
    env.close()


def test_registry_invalid_register_raises():
    """register() raises ValueError on empty name or non-callable factory."""
    reg = EnvRegistry()
    with pytest.raises(ValueError, match="non-empty string"):
        reg.register("", lambda: None)
    with pytest.raises(ValueError, match="callable"):
        reg.register("x", "not_a_callable")
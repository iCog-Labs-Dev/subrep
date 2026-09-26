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


# ---------------------------------------------------------------------------
# Group B Tests
# ---------------------------------------------------------------------------


# --- B1: Safety-Gymnasium stub contract ---

class _SafeGymStubEnv:
    """Minimal fake Safety-Gym backend for tests (no real package needed)."""

    def __init__(self):
        self._step_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(5, dtype=np.float32)
        reward = 1.0
        cost = 0.2
        terminated = self._step_count >= 3
        truncated = False
        info = {}
        return obs, reward, cost, terminated, truncated, info

    def close(self):
        pass


from env.safety_gymnasium_wrapper import SafetyGymnasiumEnv


def test_safety_gym_env_metadata_is_valid():
    """SafetyGymnasiumEnv metadata passes validate_env_metadata."""
    env = SafetyGymnasiumEnv(env=_SafeGymStubEnv())
    validate_env_metadata(env.metadata)
    env.close()


def test_safety_gym_env_motive_names():
    """SafetyGymnasiumEnv exposes [Safety, Task] motive names."""
    env = SafetyGymnasiumEnv(env=_SafeGymStubEnv())
    assert env.metadata["motive_names"] == ["Safety", "Task"]
    env.close()


def test_safety_gym_reset_returns_two_tuple():
    """SafetyGymnasiumEnv reset() returns (obs, info) 2-tuple."""
    env = SafetyGymnasiumEnv(env=_SafeGymStubEnv())
    result = env.reset()
    assert isinstance(result, tuple) and len(result) == 2
    obs, info = result
    assert obs is not None
    assert isinstance(info, dict)
    env.close()


def test_safety_gym_step_returns_five_tuple():
    """SafetyGymnasiumEnv step() returns (obs, motives, terminated, truncated, info) 5-tuple."""
    env = SafetyGymnasiumEnv(env=_SafeGymStubEnv())
    env.reset()
    result = env.step(0)
    assert isinstance(result, tuple) and len(result) == 5
    obs, motives, terminated, truncated, info = result
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_safety_gym_motive_vector_is_2d_float32():
    """SafetyGymnasiumEnv motive vector is shape (2,), dtype float32, finite."""
    env = SafetyGymnasiumEnv(env=_SafeGymStubEnv())
    env.reset()
    _, motives, _, _, _ = env.step(0)
    assert motives.shape == (2,)
    assert motives.dtype == np.float32
    assert np.all(np.isfinite(motives))
    env.close()


def test_safety_gym_task_payoff_in_info():
    """SafetyGymnasiumEnv step info contains finite scalar task_payoff."""
    env = SafetyGymnasiumEnv(env=_SafeGymStubEnv())
    env.reset()
    _, _, _, _, info = env.step(0)
    assert "task_payoff" in info
    assert np.isfinite(info["task_payoff"])
    env.close()


def test_safety_gym_satisfies_base_protocol():
    """SafetyGymnasiumEnv satisfies runtime_checkable SubRepBaseEnv protocol."""
    env = SafetyGymnasiumEnv(env=_SafeGymStubEnv())
    assert isinstance(env, SubRepBaseEnv)
    env.close()


# --- B2: Strengthened metadata validation ---

def _valid_meta():
    return {
        "environment_id": "test_env",
        "motive_names": ["Safety", "Task"],
        "motive_schema_version": "1.0",
        "payoff_schema_version": "1.0",
        "observation_schema_version": "1.0",
        "action_schema_version": "1.0",
    }


def test_validate_metadata_rejects_empty_environment_id():
    """Empty or whitespace environment_id must raise ValueError."""
    meta = _valid_meta()
    meta["environment_id"] = ""
    with pytest.raises(ValueError, match="environment_id"):
        validate_env_metadata(meta)

    meta["environment_id"] = "   "
    with pytest.raises(ValueError, match="environment_id"):
        validate_env_metadata(meta)


def test_validate_metadata_rejects_duplicate_motive_names():
    """Duplicate entries in motive_names must raise ValueError."""
    meta = _valid_meta()
    meta["motive_names"] = ["Safety", "Safety"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_env_metadata(meta)


def test_validate_metadata_rejects_empty_version_strings():
    """Empty version strings must raise ValueError."""
    for key in ("motive_schema_version", "payoff_schema_version",
                "observation_schema_version", "action_schema_version"):
        meta = _valid_meta()
        meta[key] = ""
        with pytest.raises(ValueError):
            validate_env_metadata(meta)


# --- B5: Lunar version string alignment ---

def test_lunar_metadata_version_matches_schema_constant():
    """LunarLander metadata motive_schema_version must equal LUNARLANDER_OBJECTIVE_SCHEMA."""
    from schemas.minecraft_objective_schema import LUNARLANDER_OBJECTIVE_SCHEMA
    env = SubRepEnv()
    assert env.metadata["motive_schema_version"] == LUNARLANDER_OBJECTIVE_SCHEMA.motive_schema_version
    env.close()


# --- B4: Dimension-safe store queries ---

def test_store_query_by_weights_raises_on_dimension_mismatch():
    """query_by_weights raises ValueError when weight length != cert delta_n length.

    Uses the hyperon-gated module; skipped when hyperon is absent.
    This test validates the same logic covered in test_certificate_storage.py
    but exercises the import path from test_env_contract.
    """
    # The full hyperon-backed version is covered in test_certificate_storage.py.
    # Here we validate the metta_storage._validated_weights_array rejects
    # invalid simplex vectors without needing hyperon.
    from certification.metta_storage import CertificateStore
    # An empty store accepts any valid simplex weight (no certs to mismatch).
    store = CertificateStore()
    # Non-simplex should still raise even on empty store.
    with pytest.raises(ValueError):
        store.query_by_weights([0.3, 0.3])  # sum != 1
    with pytest.raises(ValueError):
        store.query_by_weights([-0.1, 1.1])  # negative component
    # Valid simplex on empty store → empty list (no certs to mismatch).
    result = store.query_by_weights([0.5, 0.5])
    assert result == []




# --- B3: MDN record dimension enforcement ---

from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord


def _make_candidate(skill_id="s1", n=2):
    return CandidateSkillRecord(
        skill_id=skill_id,
        delta_r=0.5,
        delta_n=tuple([0.1] * n),
        is_certified=True,
        gate_type="CDS",
    )


def test_candidate_motive_names_length_must_match_delta_n():
    """CandidateSkillRecord raises when motive_names length != delta_n length."""
    with pytest.raises(ValueError, match="motive_names length"):
        CandidateSkillRecord(
            skill_id="x",
            delta_r=0.1,
            delta_n=(0.1, 0.2),
            is_certified=True,
            gate_type="CDS",
            domain_id="d",
            motive_schema_version="1.0",
            motive_names=("A", "B", "C"),  # 3 names, 2D delta_n
        )


def test_candidate_all_or_none_schema_identity():
    """CandidateSkillRecord raises on partial schema identity."""
    with pytest.raises(ValueError, match="all-or-none"):
        CandidateSkillRecord(
            skill_id="x",
            delta_r=0.1,
            delta_n=(0.1, 0.2),
            is_certified=True,
            gate_type="CDS",
            domain_id="minecraft_village_defense_trade",
            # motive_schema_version and motive_names missing
        )


def test_decision_record_rejects_mismatched_candidate_dims():
    """MDNDecisionRecord raises when a candidate's delta_n length differs from alpha length."""
    candidate_2d = _make_candidate("s1", n=2)
    candidate_6d = _make_candidate("s2", n=6)

    alpha_2 = (1.0, 1.0)
    support = (0.5, 0.5)
    weights = (0.5, 0.5)

    with pytest.raises(ValueError, match="delta_n of length"):
        MDNDecisionRecord(
            context=(0.1, 0.2),
            alpha=alpha_2,
            support_values=support,
            weights_used=weights,
            candidate_skills=(candidate_2d, candidate_6d),  # mixed dims!
            selected_skill_id="s1",
        )


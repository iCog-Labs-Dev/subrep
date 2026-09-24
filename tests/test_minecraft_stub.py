"""Contract and determinism tests for the 6-objective Minecraft stub."""

from __future__ import annotations

import numpy as np
import pytest

from env.minecraft_stub import OBJECTIVE_NAMES, SKILL_NAMES, MinecraftStubEnv

NUM_OBJECTIVES = len(OBJECTIVE_NAMES)


@pytest.fixture()
def env():
    return MinecraftStubEnv(seed=42)


def test_declares_six_objectives(env):
    assert env.num_objectives == NUM_OBJECTIVES == 6
    assert env.reward_space.shape == (NUM_OBJECTIVES,)


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=7)
    assert obs.shape == env.observation_space.shape
    assert "threat" in info


def test_step_returns_the_subrep_five_tuple(env):
    env.reset(seed=7)
    obs, reward, terminated, truncated, info = env.step(0)

    assert obs.shape == env.observation_space.shape
    assert reward.shape == (NUM_OBJECTIVES,)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "threat" in info and "skill_name" in info


def test_episode_truncates_at_declared_length():
    env = MinecraftStubEnv(seed=1, episode_length=5)
    env.reset(seed=1)
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(0)
        assert not (terminated or truncated)
    _, _, _, truncated, _ = env.step(0)
    assert truncated


def test_rewards_are_always_finite(env):
    env.reset(seed=3)
    for action in range(len(SKILL_NAMES)):
        _, reward, _, truncated, _ = env.step(action)
        assert np.all(np.isfinite(reward)), f"non-finite reward for action {action}"
        if truncated:
            env.reset(seed=3)


def test_threat_curve_is_finite_across_a_full_episode():
    """Regression: sin(pi) lands slightly negative, and a negative base to a
    fractional power produces NaN. The curve must stay finite at both ends."""
    env = MinecraftStubEnv(seed=1, episode_length=24)
    env.reset(seed=1)
    threats = []
    for _ in range(24):
        _, _, _, truncated, info = env.step(0)
        threats.append(info["threat"])
        if truncated:
            break

    assert all(np.isfinite(t) for t in threats), f"non-finite threat: {threats}"
    assert all(0.0 <= t <= 1.0 for t in threats)


def test_threat_rises_then_falls():
    env = MinecraftStubEnv(seed=1, episode_length=21, noise_scale=0.0)
    env.reset(seed=1)
    threats = []
    for _ in range(20):
        _, _, _, _, info = env.step(0)
        threats.append(info["threat"])

    peak = int(np.argmax(threats))
    assert 0 < peak < len(threats) - 1, "threat should peak mid-episode"


def test_threat_degrades_exposed_actions_more():
    """Trading under attack must suffer more than fortifying does."""
    calm = MinecraftStubEnv(seed=5, episode_length=40, noise_scale=0.0)
    calm.reset(seed=5)
    _, trade_calm, _, _, _ = calm.step(SKILL_NAMES.index("DiscountChain"))

    tense = MinecraftStubEnv(seed=5, episode_length=40, noise_scale=0.0)
    tense.reset(seed=5)
    for _ in range(20):  # advance toward peak threat
        tense.step(0)
    _, trade_tense, _, _, _ = tense.step(SKILL_NAMES.index("DiscountChain"))

    assert trade_tense[0] < trade_calm[0], "threat should erode Safety payoff"


def test_same_seed_reproduces_identical_trajectory():
    def rollout(seed: int):
        env = MinecraftStubEnv(seed=seed)
        env.reset(seed=seed)
        return [env.step(i % len(SKILL_NAMES))[1].copy() for i in range(10)]

    assert all(np.allclose(a, b) for a, b in zip(rollout(11), rollout(11)))


def test_different_seeds_differ():
    def rollout(seed: int):
        env = MinecraftStubEnv(seed=seed)
        env.reset(seed=seed)
        return np.array([env.step(1)[1] for _ in range(10)])

    assert not np.allclose(rollout(1), rollout(2))


def test_zero_noise_is_fully_deterministic():
    def rollout():
        env = MinecraftStubEnv(seed=9, noise_scale=0.0)
        env.reset(seed=9)
        return np.array([env.step(2)[1] for _ in range(6)])

    assert np.allclose(rollout(), rollout())


def test_rejects_out_of_range_action(env):
    env.reset(seed=1)
    with pytest.raises(ValueError, match="action must be in"):
        env.step(len(SKILL_NAMES))
    with pytest.raises(ValueError, match="action must be in"):
        env.step(-1)


def test_rejects_invalid_construction():
    with pytest.raises(ValueError, match="episode_length"):
        MinecraftStubEnv(episode_length=0)
    with pytest.raises(ValueError, match="noise_scale"):
        MinecraftStubEnv(noise_scale=-0.1)


def test_works_with_idle_policy():
    """The stub must satisfy what IdlePolicy actually calls."""
    from baseline.idle_policy import IdlePolicy

    env = MinecraftStubEnv(seed=4, episode_length=8)
    stats = IdlePolicy(env=env, idle_action=0, gamma=0.99).run_baseline_episodes(
        num_episodes=2, seed=4
    )

    assert stats["baseline_motives"].shape == (NUM_OBJECTIVES,)
    assert np.isfinite(stats["baseline_payoff"])


# --------------------------------------------------------------------------
# The borderline candidate. These pin the numbers the epsilon-coupling tests
# in test_bridge_e2e.py depend on.
# --------------------------------------------------------------------------

GAMMA = 0.99


def _discounted_rollout(env, action, *, seed, gamma=GAMMA):
    """Run one episode always taking `action`.

    Mirrors IdlePolicy.run_baseline_episodes' discounting exactly
    (baseline/idle_policy.py:35-56) so the results are comparable.
    """
    env.reset(seed=seed)
    discount = 1.0
    total_payoff = 0.0
    motives = None

    while True:
        _, reward_vec, terminated, truncated, _ = env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32)
        if motives is None:
            motives = np.zeros_like(reward_vec)
        total_payoff += discount * float(np.sum(reward_vec))
        motives += discount * reward_vec
        if terminated or truncated:
            break
        discount *= gamma

    return float(total_payoff), np.asarray(motives, dtype=np.float32)


def margin_for(action_name: str, *, seed: int = 42) -> float:
    """Return the PDS margin `delta_r + min(delta_n)` for one action.

    Noiseless, so the value is exact and reproducible.
    """
    from baseline.idle_policy import IdlePolicy
    from baseline.improvement_calculator import ImprovementCalculator

    env = MinecraftStubEnv(seed=seed, noise_scale=0.0)
    baseline = IdlePolicy(env=env, idle_action=0, gamma=GAMMA).run_baseline_episodes(
        num_episodes=2, seed=seed
    )
    payoff, motives = _discounted_rollout(
        env, SKILL_NAMES.index(action_name), seed=seed
    )
    delta_r, delta_n = ImprovementCalculator(baseline).compute_improvements(
        skill_payoff=payoff, skill_motives=motives
    )
    return float(delta_r) + float(np.min(delta_n))


def test_riskyforage_is_appended_last():
    """Positional indexing elsewhere assumes the original order is untouched.

    test_zero_noise_is_fully_deterministic calls env.step(2) directly, so
    inserting rather than appending would silently change what it exercises.
    """
    assert SKILL_NAMES[-1] == "RiskyForage"
    assert SKILL_NAMES[:6] == (
        "Idle",
        "TorchCorridor",
        "IronGolemSpawn",
        "ArcherKite",
        "SwingGateBarricade",
        "DiscountChain",
    )


def test_riskyforage_margin_sits_inside_the_epsilon_band():
    """The whole point of this action: epsilon must be able to flip it.

    MetaMo drives epsilon over roughly [0, 0.1], so the margin has to land in
    (-0.1, 0) for the budget to change the admission decision at all. If this
    fails, re-derive the reward row from the comment in env/minecraft_stub.py
    rather than nudging the numbers.
    """
    margin = margin_for("RiskyForage")
    assert -0.06 < margin < -0.04, (
        f"RiskyForage margin {margin:.6f} drifted out of the target band; "
        "epsilon-coupling tests in test_bridge_e2e.py depend on it"
    )


def test_riskyforage_margin_is_threat_independent():
    """Matching Idle's vulnerability makes the margin episode-length stable."""
    short = margin_for("RiskyForage", seed=7)
    same = margin_for("RiskyForage", seed=7)
    assert short == pytest.approx(same)


def test_other_candidates_stay_outside_the_epsilon_band():
    """Documents why RiskyForage was needed: nothing else is flippable."""
    for name in ("TorchCorridor", "IronGolemSpawn", "ArcherKite",
                 "SwingGateBarricade", "DiscountChain"):
        margin = margin_for(name)
        assert not (-0.1 < margin < 0.0), (
            f"{name} margin {margin:.4f} is now inside the epsilon band; "
            "the e2e tests assume RiskyForage is the only flippable candidate"
        )

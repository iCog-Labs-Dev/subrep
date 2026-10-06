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


def test_accessors_report_the_current_state(env):
    obs, info = env.reset(seed=7)
    np.testing.assert_array_equal(env.observation(), obs)
    assert env.threat == pytest.approx(info["threat"])
    assert not env.episode_over

    obs, _, _, _, info = env.step(1)
    np.testing.assert_array_equal(env.observation(), obs)
    assert env.threat == pytest.approx(info["threat"])


def test_phi_is_the_running_objective_levels(env):
    """phi(x) is the objective levels -- the running totals the observation
    already carries (obs = [threat, progress, *totals])."""
    obs, _ = env.reset(seed=7)
    np.testing.assert_array_equal(env.phi(), obs[2:])
    np.testing.assert_array_equal(env.phi(), np.zeros(NUM_OBJECTIVES))

    for action in range(len(SKILL_NAMES)):
        obs, reward, _, _, info = env.step(action)
        np.testing.assert_array_equal(env.phi(), obs[2:])
        np.testing.assert_array_equal(env.phi(), info["motive_totals"])


def test_phi_returns_a_copy(env):
    env.reset(seed=7)
    snapshot = env.phi()
    snapshot += 100.0
    np.testing.assert_array_equal(env.phi(), np.zeros(NUM_OBJECTIVES))


def test_episode_over_flips_at_the_declared_length():
    env = MinecraftStubEnv(seed=1, episode_length=3)
    env.reset(seed=1)
    for _ in range(2):
        env.step(0)
        assert not env.episode_over
    env.step(0)
    assert env.episode_over


def test_task_reward_is_reported_and_only_trade_earns_it(env):
    env.reset(seed=7)
    for action, name in enumerate(SKILL_NAMES):
        _, reward, _, _, info = env.step(action)
        assert "task_reward" in info
        if name == "DiscountChain":
            assert info["task_reward"] > 0.0
        else:
            assert info["task_reward"] == 0.0
        # The objective vector is untouched: IdlePolicy's contract still holds.
        assert reward.shape == (NUM_OBJECTIVES,)


def test_task_reward_constants_match_the_rollout_and_the_demo():
    """The stub restates the horizon and discount rather than importing the
    rollout module; pin them so they cannot drift apart."""
    from demo.run_metamo_pipeline import GAMMA
    from env import minecraft_stub
    from env.minecraft_rollout import DEFAULT_HORIZON

    assert minecraft_stub._TASK_REWARD_HORIZON == DEFAULT_HORIZON
    assert minecraft_stub._TASK_REWARD_GAMMA == GAMMA


def test_per_action_tables_are_aligned():
    from env import minecraft_stub

    n = len(SKILL_NAMES)
    assert len(minecraft_stub._BASE_REWARDS) == n
    assert len(minecraft_stub._THREAT_VULNERABILITY) == n
    assert len(minecraft_stub._TASK_REWARD) == n


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
# Calibration. These pin what _REWARD_SCALE was measured to deliver; the
# epsilon-coupling tests in test_bridge_e2e.py depend on it. If one fails,
# re-measure the scale (see the comment on _REWARD_SCALE), don't nudge it.
# --------------------------------------------------------------------------

GAMMA = 0.99


def _margins_over_a_threat_cycle():
    """Noiseless PDS margins delta_r + min(delta_n), per candidate, at every
    decision of one episode (idling between decisions, so threat rises to its
    peak and falls again)."""
    from env.minecraft_rollout import DEFAULT_HORIZON, evaluate_candidates

    env = MinecraftStubEnv(seed=42, noise_scale=0.0)
    env.reset(seed=42)
    cycle = []
    while not env.episode_over:
        threat = env.threat
        _, records = evaluate_candidates(env, gamma=GAMMA, horizon=DEFAULT_HORIZON)
        cycle.append(
            (threat, {r.skill_id: r.delta_r + min(r.delta_n) for r in records})
        )
        for _ in range(DEFAULT_HORIZON):
            env.step(0)
    return cycle


def test_every_margin_sits_inside_the_epsilon_band_all_episode():
    """MetaMo drives epsilon over [0, 0.1], so epsilon can only decide options
    whose margin lies in (-0.1, 0). Every option must stay there, at every
    threat level -- otherwise it is always or never admitted and the budget
    cannot touch it."""
    for threat, margins in _margins_over_a_threat_cycle():
        for skill_id, margin in margins.items():
            assert -0.1 < margin < 0.0, (
                f"{skill_id} margin {margin:+.4f} at threat {threat:.2f}"
            )


def test_margins_spread_across_the_band():
    """Different epsilons must admit different subsets, so the margins cannot
    all bunch at one end of the band."""
    _, margins = _margins_over_a_threat_cycle()[0]
    assert min(margins.values()) < -0.07
    assert max(margins.values()) > -0.03


def test_trading_gets_riskier_as_threat_rises():
    """DiscountChain is the most threat-exposed option, so its margin must
    fall as threat peaks -- that is what lets a tightening epsilon stop the
    agent trading under attack."""
    cycle = _margins_over_a_threat_cycle()
    calm = cycle[0][1]["DiscountChain"]
    peak = min(cycle, key=lambda entry: -entry[0])[1]["DiscountChain"]
    assert peak < calm - 0.03


def test_iron_golem_cost_matches_the_specification():
    """Specified O5 IronGolemSpawn: worst objective change -0.08."""
    from env.minecraft_rollout import DEFAULT_HORIZON, evaluate_candidates

    env = MinecraftStubEnv(seed=42, noise_scale=0.0)
    env.reset(seed=42)
    _, records = evaluate_candidates(env, gamma=GAMMA, horizon=DEFAULT_HORIZON)
    golem = next(r for r in records if r.skill_id == "IronGolemSpawn")
    assert min(golem.delta_n) == pytest.approx(-0.08, abs=0.01)

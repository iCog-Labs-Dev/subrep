"""Automated tests for the village_sim environment, motives, baseline and gates."""

from __future__ import annotations

import numpy as np
import pytest

from village_sim.state import VillageState
from village_sim.motives import MOTIVE_NAMES, phi
from village_sim.env import VillageEnv, DELIVERY_TARGET
from village_sim.baseline import run_policy, idle_policy
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate


class _QueueRng:
    """Fake RNG that yields a fixed sequence so raid branches are deterministic."""

    def __init__(self, values):
        self._values = list(values)

    def random(self):
        return self._values.pop(0)

    def uniform(self, a, b):
        return a


def _controlled_env(seed: int = 0) -> VillageEnv:
    """Env with infrastructure_pct=1.0 so raids can never start (deterministic)."""
    env = VillageEnv(seed=seed)
    env.reset(seed=seed)
    env.state.infrastructure_pct = 1.0
    return env


# --- motive names and phi -------------------------------------------------


def test_motive_names_exact_and_ordered():
    assert MOTIVE_NAMES == [
        "Safety",
        "Reputation",
        "DeadlineSlack",
        "InventoryValue",
        "Sustainability",
        "Infrastructure",
    ]


def test_phi_shape_is_six():
    assert phi(VillageState()).shape == (6,)


def test_phi_dtype_is_float32():
    assert phi(VillageState()).dtype == np.float32


def test_phi_initial_values():
    expected = np.array([1.0, 0.5, 1.0, 0.0, 1.0, 0.2], dtype=np.float32)
    np.testing.assert_allclose(phi(VillageState()), expected, atol=1e-6)


def test_phi_is_finite_for_default_and_extreme_states():
    assert np.all(np.isfinite(phi(VillageState())))

    s = VillageState()
    s.time_step = 0
    s.villager_hp = 0.0
    s.player_hp = 0.0
    s.reputation = 1.0
    s.inventory_value = 1000.0
    s.fuel = 0.0
    s.infrastructure_pct = 1.0
    assert np.all(np.isfinite(phi(s)))


# --- reproducibility and reset --------------------------------------------


def test_reproducible_with_same_seed_and_actions():
    def torch_policy(state):
        return "torch_corridor" if state.fuel >= 1 else "idle"

    payoff_a, motives_a = run_policy(torch_policy, seed=42)
    payoff_b, motives_b = run_policy(torch_policy, seed=42)

    assert payoff_a == payoff_b
    np.testing.assert_array_equal(motives_a, motives_b)


def test_reset_accepts_seed_and_reseeds_rng():
    env_a = VillageEnv(seed=100)
    env_b = VillageEnv(seed=200)
    env_a.reset(seed=5)
    env_b.reset(seed=5)

    for _ in range(50):
        state_a, motives_a, terminated_a, truncated_a, info_a = env_a.step("idle")
        state_b, motives_b, terminated_b, truncated_b, info_b = env_b.step("idle")
        assert state_a == state_b
        np.testing.assert_array_equal(motives_a, motives_b)
        assert terminated_a == terminated_b
        assert truncated_a == truncated_b
        assert info_a["raid_active"] == info_b["raid_active"]


def test_reset_same_seed_reproduces_trace_on_same_env():
    def safe_policy(state):
        if state.fuel >= 3 and "GolemPresent" not in state.facts:
            return "iron_golem_spawn"
        if state.fuel >= 1:
            return "torch_corridor"
        if state.raid_active:
            return "archer_kite"
        if state.time_step % 2 == 0:
            return "trade"
        return "idle"

    def collect_trace(env):
        states, motives, raids, dones = [], [], [], []
        done = False
        while not done:
            action = safe_policy(env.state)
            state, motive_vec, terminated, truncated, info = env.step(action)
            states.append(state)
            motives.append(motive_vec)
            raids.append(info["raid_active"])
            done = terminated or truncated
            dones.append(done)
        return states, motives, raids, dones

    env = VillageEnv(seed=0)
    env.reset(seed=99)
    trace_a = collect_trace(env)

    # Re-running reset(seed=99) on the same env must reseed the RNG so the
    # same action sequence produces an identical trace (3.3).
    env.reset(seed=99)
    trace_b = collect_trace(env)

    assert trace_a[0] == trace_b[0]  # states
    for motives_a, motives_b in zip(trace_a[1], trace_b[1]):
        np.testing.assert_array_equal(motives_a, motives_b)
    assert trace_a[2] == trace_b[2]  # raid events
    assert trace_a[3] == trace_b[3]  # termination result


def test_reset_returns_fresh_default_state():
    env = VillageEnv(seed=0)
    env.reset()
    env.state.villager_hp = 3.0
    env.state.task_completed = True
    state, _ = env.reset()

    assert state.villager_hp == 20.0
    assert state.time_step == 0
    assert state.task_completed is False


def test_policy_cannot_mutate_env_state_via_returned_snapshot():
    env = _controlled_env()
    returned_state, _, _, _, _ = env.step("trade")

    # Policies only receive copies; mutating them must not corrupt the env.
    returned_state.inventory_value = 999.0
    returned_state.facts.add("GateOpen")

    assert env.state.inventory_value == pytest.approx(3.0)
    assert "GateOpen" not in env.state.facts


# --- actions --------------------------------------------------------------


@pytest.mark.parametrize(
    "action, checks",
    [
        ("idle", lambda s: (s.fuel == 10.0, s.infrastructure_pct == 1.0)),
        ("torch_corridor", lambda s: (s.fuel == 9.0, s.infrastructure_pct == 1.0)),
        (
            "iron_golem_spawn",
            lambda s: (s.fuel == 7.0, s.infrastructure_pct == 1.0, "GolemPresent" in s.facts),
        ),
        ("discount_chain", lambda s: (s.emerald_price == 6.5, s.reputation == 0.55)),
        ("reputation_first", lambda s: (s.reputation == 0.6,)),
        ("trade", lambda s: (s.inventory_value == 3.0,)),
    ],
)
def test_action_has_expected_state_effect(action, checks):
    env = _controlled_env()
    state, motives, done, _, info = env.step(action)
    assert all(checks(state))


def test_archer_kite_ends_raid_when_successful():
    env = _controlled_env()
    env.state.raid_active = True
    env.state.raid_intensity = 0.5
    env.rng = _QueueRng([1.0, 0.0])  # survives damage phase, then archer succeeds
    _, _, done, _, info = env.step("archer_kite")
    assert env.state.raid_active is False


def test_archer_kite_leaves_raid_when_unsuccessful():
    env = _controlled_env()
    env.state.raid_active = True
    env.state.raid_intensity = 0.5
    env.rng = _QueueRng([1.0, 0.9])  # survives damage phase, archer fails the 0.8 check
    _, _, done, _, info = env.step("archer_kite")
    assert env.state.raid_active is True


def test_unknown_action_is_rejected():
    env = _controlled_env()
    with pytest.raises(ValueError):
        env.step("fly_to_moon")


def test_unknown_action_does_not_advance_state():
    env = _controlled_env()
    time_before = env.state.time_step
    hp_before = env.state.villager_hp
    with pytest.raises(ValueError):
        env.step("not_a_real_action")
    assert env.state.time_step == time_before
    assert env.state.villager_hp == hp_before


def test_torch_corridor_requires_fuel():
    env = _controlled_env()
    env.state.fuel = 0
    with pytest.raises(ValueError):
        env.step("torch_corridor")


def test_iron_golem_spawn_requires_fuel():
    env = _controlled_env()
    env.state.fuel = 2
    with pytest.raises(ValueError):
        env.step("iron_golem_spawn")


def test_archer_kite_requires_active_raid():
    env = _controlled_env()
    with pytest.raises(ValueError):
        env.step("archer_kite")


def test_resource_precondition_failure_does_not_mutate_state():
    env = _controlled_env()
    env.state.fuel = 0
    time_before = env.state.time_step
    with pytest.raises(ValueError):
        env.step("torch_corridor")
    assert env.state.time_step == time_before


# --- deadline and terminal behavior ---------------------------------------


def test_deadline_slack_decreases_with_time():
    assert phi(VillageState())[2] == pytest.approx(1.0)

    s = VillageState()
    s.time_step = 50
    assert phi(s)[2] == pytest.approx(0.5)

    s.time_step = s.total_steps
    assert phi(s)[2] == pytest.approx(0.0)


def test_episode_terminates_exactly_at_deadline():
    env = _controlled_env()
    done = False
    steps = 0
    while not done:
        _, _, done, _, info = env.step("idle")
        steps += 1
    assert steps == env.state.total_steps
    assert env.state.time_step == env.state.total_steps


def test_terminal_when_villager_hp_depleted():
    s = VillageState()
    s.villager_hp = 0.0
    assert s.is_terminal() is True


def test_terminal_when_time_up():
    s = VillageState()
    s.time_step = s.total_steps
    assert s.is_terminal() is True


def test_not_terminal_early():
    assert VillageState().is_terminal() is False


# --- independent scalar payoff / motive accumulation ----------------------

def test_payoff_and_motives_accumulate_independently():
    env = _controlled_env()
    total_payoff = 0.0
    total_motives = np.zeros(6, dtype=np.float32)
    t = 0
    done = False
    while not done:
        state, motives, done, _, info = env.step("trade")
        discount = 0.99 ** t
        total_payoff += discount * float(info["task_reward"])
        total_motives += discount * motives
        t += 1

    # Eight trades of 3.0 deliver the 24.0 target on the 8th step (index 7).
    assert state.task_completed is True
    assert total_payoff == pytest.approx(0.99 ** 7)
    # Scalar payoff and motive accumulation are decoupled (3.1 fix).
    assert total_payoff != pytest.approx(float(total_motives.sum()), abs=1e-3)


def test_run_policy_payoff_is_not_motive_sum():
    payoff, motives = run_policy(idle_policy, seed=1)
    assert not np.isclose(payoff, float(np.sum(motives)), atol=1e-3)


# --- CDS / PDS / CVaR gates ------------------------------------------------

def test_cds_pass_six_dimensional():
    delta_r = 5.0
    delta_n = np.array([1.0, 2.0, 0.5, 3.0, 0.1, 4.0], dtype=np.float32)
    assert CDSGate().admit(delta_r, delta_n) is True


def test_cds_pass_at_zero_margin():
    delta_r = 0.5
    delta_n = np.array([-0.5, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert CDSGate().admit(delta_r, delta_n) is True


def test_cds_fail_six_dimensional():
    delta_r = 1.0
    delta_n = np.array([-5.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert CDSGate().admit(delta_r, delta_n) is False


def test_pds_pass_within_budget():
    delta_r = 1.0
    delta_n = np.array([-5.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert PDSGate(epsilon=5.0).admit(delta_r, delta_n) is True


def test_pds_pass_at_budget_boundary():
    delta_r = 1.0
    delta_n = np.array([-5.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert PDSGate(epsilon=4.0).admit(delta_r, delta_n) is True


def test_pds_fail_over_budget():
    delta_r = 1.0
    delta_n = np.array([-5.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    assert PDSGate(epsilon=3.0).admit(delta_r, delta_n) is False


def test_cvar_six_dimensional():
    torch = pytest.importorskip("torch")
    from certification.cvar_test import CVaRGate

    alpha = np.ones(6, dtype=np.float32)
    delta_n = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # For w ~ Dirichlet(ones), w^T delta_n == 1.0 regardless of the draw.
    assert CVaRGate(confidence=0.1).admit(1.0, delta_n, alpha) is True
    assert CVaRGate(confidence=0.1).admit(-10.0, -delta_n, alpha) is False


# --- trade behavior --------------------------------------------------------

def test_simplified_trade_behavior():
    env = _controlled_env()
    env.step("trade")
    assert env.state.inventory_value == pytest.approx(3.0)  # 24 / 8

    env.step("discount_chain")
    assert env.state.emerald_price == pytest.approx(6.5)
    before = env.state.inventory_value
    env.step("trade")
    assert env.state.inventory_value == pytest.approx(before + 24.0 / 6.5)


def test_emerald_price_floor_and_large_gains():
    env = _controlled_env()
    for _ in range(20):
        env.step("discount_chain")
    assert env.state.emerald_price == pytest.approx(1.0)

    before = env.state.inventory_value
    env.step("trade")
    assert env.state.inventory_value == pytest.approx(before + 24.0)


def test_delivery_completed_via_repeated_trades():
    env = _controlled_env()
    rewards = []
    done = False
    while not done:
        _, _, done, _, info = env.step("trade")
        rewards.append(info["task_reward"])

    assert env.state.task_completed is True
    assert env.state.inventory_value >= DELIVERY_TARGET
    assert rewards.count(1.0) == 1
    assert -1.0 not in rewards
    assert phi(env.state)[3] == pytest.approx(1.0)
import copy
import random
import numpy as np
from village_sim.state import VillageState
from village_sim.motives import MOTIVE_NAMES, phi

# VillageEnv is a STANDALONE SYNTHETIC PROXY for the SubRep certification
# workflow, updated in Phase 2 to conform to the SubRepBaseEnv contract.
#
# Interface contract:
#   reset(seed=None, options=None) -> (VillageState, info)
#   step(action) -> (VillageState, motives, terminated, truncated, info)
#       motives     : np.ndarray of shape (6,)  -- the motive vector (phi)
#       terminated  : bool                      -- terminal flag (time-limit or casualty)
#       truncated   : bool                      -- external truncation flag (False)
#       info        : dict with "task_payoff", "task_reward", and "raid_active"
#
# It provides:
#   * deterministic reset(seed=...) -- reseeds the RNG (see reset)
#   * action validation -- unknown actions and unmet resource preconditions
#     raise ValueError before any state is mutated (see _validate_action).
#   * metadata property exposing environment_id, motive_names, and schema versions.
#
# Defense against policy mutation (Section 4.6): reset() and step() return a
# COPY of the internal VillageState (see _snapshot). A policy therefore cannot
# corrupt the environment by mutating the object it is handed; only the env
# owns the live state.
#
# Documented simplifications vs. full Minecraft mechanics (Section 4):
#   * iron_golem_spawn (4.2): adds the "GolemPresent" fact, but that fact does
#     not affect raid damage or any other mechanic.
#   * trade (4.3): creates inventory value without consuming resources or
#     villager stock.
#   * archer_kite (4.4): can end a raid directly with no modeled combat cost.
#   * player_hp (4.5): included in the Safety motive but is never reduced by
#     any action or raid.

# Delivery target in emerald-equivalent value; matches the normalization
# target used by the InventoryValue motive (inventory_value / 24.0).
DELIVERY_TARGET = 24.0

VALID_ACTIONS = {
    "idle",
    "torch_corridor",
    "iron_golem_spawn",
    "discount_chain",
    "reputation_first",
    "trade",
    "archer_kite",
}

class VillageEnv:
    """Standalone synthetic proxy environment for village simulation."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.state = None

    @property
    def metadata(self) -> dict:
        return {
            "environment_id": "village_sim_v1",
            "motive_names": list(MOTIVE_NAMES),
            "motive_schema_version": "1.0.0",
            "payoff_schema_version": "1.0.0",
            "observation_schema_version": "1.0.0",
            "action_schema_version": "1.0.0",
        }

    def reset(self, seed=None, options=None) -> tuple[VillageState, dict]:
        if seed is not None:
            self.rng.seed(seed)
        self.state = VillageState()
        return self._snapshot(), {}

    def step(self, action: str):
        self._validate_action(action)
        s = self.state
        s.time_step += 1

        if not s.raid_active and self.rng.random() < 0.05 * (1.0 - s.infrastructure_pct):
            s.raid_active = True
            s.raid_intensity = self.rng.uniform(0.2, 0.6)

        if s.raid_active:
            dmg = s.raid_intensity * (1.0 - s.infrastructure_pct) * 3.0
            s.villager_hp = max(0.0, s.villager_hp - dmg)
            if self.rng.random() < 0.3:
                s.raid_active = False

        self._apply_action(action)

        motives = phi(s)
        done = s.is_terminal()
        task_reward = self._compute_task_reward(done)
        info = {
            "task_reward": task_reward,
            "task_payoff": float(task_reward),
            "raid_active": s.raid_active,
        }
        return self._snapshot(), motives, done, False, info

    def close(self) -> None:
        """Close and clean up environment resources."""
        pass

    def _snapshot(self) -> VillageState:
        """Return a copy of the live state so policies cannot mutate it (4.6)."""
        s = copy.copy(self.state)
        s.facts = set(self.state.facts)
        return s

    def _validate_action(self, action: str):
        if action not in VALID_ACTIONS:
            raise ValueError(f"Unknown village action: {action!r}")
        s = self.state
        if action == "torch_corridor" and s.fuel < 1:
            raise ValueError("Cannot torch corridor: not enough fuel (need 1)")
        if action == "iron_golem_spawn" and s.fuel < 3:
            raise ValueError("Cannot spawn iron golem: not enough fuel (need 3)")
        if action == "archer_kite" and not s.raid_active:
            raise ValueError("Cannot use archer kite: no raid is active")

    def _compute_task_reward(self, done: bool) -> float:
        s = self.state
        if s.task_completed:
            return 0.0
        if s.inventory_value >= DELIVERY_TARGET and s.time_step < s.total_steps:
            s.task_completed = True
            return 1.0
        if done:
            return -1.0
        return 0.0

    def _apply_action(self, action: str):
        s = self.state
        if action == "idle":
            pass
        elif action == "torch_corridor":
            s.fuel -= 1
            s.infrastructure_pct = min(1.0, s.infrastructure_pct + 0.15)
        elif action == "iron_golem_spawn":
            s.fuel -= 3
            s.facts.add("GolemPresent")
            s.infrastructure_pct = min(1.0, s.infrastructure_pct + 0.2)
            # Simplification (4.2): "GolemPresent" does not affect raid damage.
        elif action == "discount_chain":
            # Aligned with the paper (4.1): discount_chain has a positive
            # reputation effect (the paper's example), unlike the older
            # behavior which decreased reputation.
            s.emerald_price = max(1.0, s.emerald_price - 1.5)
            s.reputation = min(1.0, s.reputation + 0.05)
        elif action == "reputation_first":
            s.reputation = min(1.0, s.reputation + 0.1)
        elif action == "trade":
            # Simplification (4.3): trade creates value without consuming
            # resources or villager stock.
            value_gained = 24.0 / s.emerald_price
            s.inventory_value += value_gained
        elif action == "archer_kite":
            # Simplification (4.4): archer_kite can end a raid directly with
            # no modeled combat cost.
            if self.rng.random() < 0.8:
                s.raid_active = False
        else:
            raise ValueError(f"Unknown village action: {action!r}")
import random
import numpy as np
from village_sim.state import VillageState
from village_sim.motives import phi

class VillageEnv:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.state = None

    def reset(self) -> VillageState:
        self.state = VillageState()
        return self.state

    def step(self, action: str):
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
        info = {"raid_active": s.raid_active}
        return s, motives, done, info

    def _apply_action(self, action: str):
        s = self.state
        if action == "idle":
            pass
        elif action == "torch_corridor":
            if s.fuel >= 1:
                s.fuel -= 1
                s.infrastructure_pct = min(1.0, s.infrastructure_pct + 0.15)
        elif action == "iron_golem_spawn":
            if s.fuel >= 3:
                s.fuel -= 3
                s.facts.add("GolemPresent")
                s.infrastructure_pct = min(1.0, s.infrastructure_pct + 0.2)
        elif action == "discount_chain":
            s.emerald_price = max(1.0, s.emerald_price - 1.5)
            s.reputation = max(0.0, s.reputation - 0.05)
        elif action == "reputation_first":
            s.reputation = min(1.0, s.reputation + 0.1)
        elif action == "trade":
            value_gained = 24.0 / s.emerald_price
            s.inventory_value += value_gained
        elif action == "archer_kite":
            if s.raid_active and self.rng.random() < 0.8:
                s.raid_active = False
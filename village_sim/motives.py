import numpy as np
from village_sim.state import VillageState

MOTIVE_NAMES = ["Safety", "Reputation", "DeadlineSlack",
                "InventoryValue", "Sustainability", "Infrastructure"]

def phi(state: VillageState) -> np.ndarray:
    safety = (state.villager_hp / 20.0 + state.player_hp / 20.0) / 2.0
    reputation = state.reputation
    deadline_slack = 1.0 - (state.time_step / state.total_steps)
    inventory_value = min(state.inventory_value / 24.0, 1.0)
    sustainability = state.fuel / 10.0
    infrastructure = state.infrastructure_pct

    return np.array([safety, reputation, deadline_slack,
                      inventory_value, sustainability, infrastructure], dtype=np.float32)
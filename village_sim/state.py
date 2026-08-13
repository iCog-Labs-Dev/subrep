from dataclasses import dataclass, field

@dataclass
class VillageState:
    # reals(x) - continuous attributes
    time_step: int = 0
    total_steps: int = 100          # dusk -> sunrise
    villager_hp: float = 20.0
    player_hp: float = 20.0
    emerald_price: float = 8.0       # lower is better for the agent
    inventory_value: float = 0.0     # emerald-equivalent value delivered
    fuel: float = 10.0               # torches/resources remaining
    infrastructure_pct: float = 0.2  # fraction of perimeter defended
    reputation: float = 0.5          # 0-1
    raid_active: bool = False
    raid_intensity: float = 0.0

    # facts(x) - discrete world facts
    facts: set = field(default_factory=set)  # e.g. {"GateOpen", "GolemPresent"}

    def is_terminal(self) -> bool:
        return self.time_step >= self.total_steps or self.villager_hp <= 0
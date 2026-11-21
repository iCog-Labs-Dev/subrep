from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from env.minecraft import areas
from village_sim.mission import VillageMission
from village_sim.state import VillageState, dusk_state


ITEM_TO_FIELD = {
    "emerald": "emeralds",
    "torch": "torches",
    "charcoal": "charcoal",
    "coal": "coal",
    "iron_block": "iron_blocks",
    "carved_pumpkin": "pumpkins",
    "enchanted_book": "books",
    "book": "books",
    "cooked_chicken": "food_items",
    "cooked_beef": "food_items",
}
_COUNT_FIELDS = (
    "emeralds",
    "torches",
    "charcoal",
    "coal",
    "iron_blocks",
    "pumpkins",
    "books",
    "food_items",
    "cure_kits",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("{} must be a dict".format(name))
    return value


def _position(value: Any, name: str) -> Tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError("{} must be a 3-int position".format(name))
    return tuple(value)


def _number(value: Any, name: str, integer: bool = False) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be numeric".format(name))
    if integer and not isinstance(value, int):
        raise ValueError("{} must be an integer".format(name))
    return value


def inventory_counts(inventory: Any) -> Dict[str, int]:
    if not isinstance(inventory, list):
        raise ValueError("inventory must be a list")
    counts = {field: 0 for field in _COUNT_FIELDS}
    splash_potions = 0
    golden_apples = 0
    for index, entry in enumerate(inventory):
        item = _mapping(entry, "inventory[{}]".format(index))
        if "id" not in item or "count" not in item:
            raise ValueError("inventory[{}] requires id and count".format(index))
        if not isinstance(item["id"], str) or not item["id"]:
            raise ValueError("inventory[{}].id must be a string".format(index))
        count = _number(item["count"], "inventory[{}].count".format(index), True)
        if count < 0:
            raise ValueError("inventory[{}].count must not be negative".format(index))
        item_name = item["id"].split(":", 1)[-1]
        field = ITEM_TO_FIELD.get(item_name)
        if field is not None:
            counts[field] += count
        elif item_name == "splash_potion":
            splash_potions += count
        elif item_name == "golden_apple":
            golden_apples += count
    counts["cure_kits"] = min(splash_potions, golden_apples)
    return counts


def _digest_villagers(payload: Any) -> Dict[str, float]:
    villagers = _mapping(payload, "digest_payload")
    if not isinstance(villagers.get("villagers"), list):
        raise ValueError("digest_payload.villagers must be a list")
    by_uuid = {}
    for index, villager in enumerate(villagers["villagers"]):
        entry = _mapping(villager, "digest_payload.villagers[{}]".format(index))
        if "uuid" not in entry or "health" not in entry:
            raise ValueError(
                "digest_payload.villagers[{}] requires uuid and health".format(index)
            )
        if not isinstance(entry["uuid"], str):
            raise ValueError("villager uuid must be a string")
        by_uuid[entry["uuid"]] = float(
            _number(entry["health"], "villager health")
        )
    return by_uuid


def _bot_values(observation: Any) -> Tuple[float, float, Tuple[int, int, int], Dict[str, int]]:
    data = _mapping(observation, "bot_observation")
    for key in ("health", "food", "position", "inventory"):
        if key not in data:
            raise ValueError("bot_observation is missing {}".format(key))
    health = float(_number(data["health"], "bot_observation.health"))
    food = float(_number(data["food"], "bot_observation.food"))
    position = _position(data["position"], "bot_observation.position")
    return health, food, position, inventory_counts(data["inventory"])


def _route_progress(position: Tuple[int, int, int]) -> float:
    waypoints = areas.route_waypoints()
    first_x = waypoints[0][0]
    last_x = waypoints[-1][0]
    return max(0.0, min(1.0, (position[0] - first_x) / float(last_x - first_x)))


def build_start_state(
    digest_payload: Any, bot_observation: Any, mission: VillageMission
) -> VillageState:
    digest = _mapping(digest_payload, "digest_payload")
    for key in ("tick", "villagers", "zombie", "golem_count", "hostile_count", "raider_count", "wall_gaps"):
        if key not in digest:
            raise ValueError("digest_payload is missing {}".format(key))
    tick = _number(digest["tick"], "digest_payload.tick", True)
    health, food, position, counts = _bot_values(bot_observation)
    villager_by_uuid = _digest_villagers(digest)
    protected_uuids = areas.protected_villager_uuids()
    villager_health = tuple(
        villager_by_uuid.get(uuid, 0.0) for uuid in protected_uuids
    )
    zombie = _mapping(digest["zombie"], "digest_payload.zombie")
    if not isinstance(zombie.get("present"), bool):
        raise ValueError("digest_payload.zombie.present must be boolean")
    wall_gaps = digest["wall_gaps"]
    if not isinstance(wall_gaps, dict):
        raise ValueError("digest_payload.wall_gaps must be a dict")
    gap_total = len(areas.gap_positions())
    gate_count = sum(value == "minecraft:oak_fence_gate" for value in wall_gaps.values())
    state = dusk_state(mission)
    state.world_tick = tick
    state.location = areas.location_of(position)
    state.route_progress = _route_progress(position)
    state.player_health = health
    state.player_food = food
    state.villager_health = villager_health
    state.zombie_villagers = 1 if zombie["present"] else 0
    state.golems = _number(digest["golem_count"], "digest_payload.golem_count", True)
    state.hostiles = _number(digest["hostile_count"], "digest_payload.hostile_count", True)
    state.raiders = _number(digest["raider_count"], "digest_payload.raider_count", True)
    state.wall_fraction = gate_count / float(gap_total)
    for field, value in counts.items():
        setattr(state, field, value)
    return state


class StateTracker:
    def __init__(
        self,
        start_state: VillageState,
        mission: VillageMission,
        golem_build_steps: int = 2,
        cure_steps: int = 4,
        trade_steps: int = 2,
        fast_trade_steps: int = 1,
    ) -> None:
        if not isinstance(start_state, VillageState):
            raise ValueError("start_state must be a VillageState")
        self._state = start_state.copy()
        self._mission = mission
        # These are simulator dynamics fitted against real-server rollouts.
        self._golem_build_steps = golem_build_steps
        self._cure_steps = cure_steps
        self._trade_steps = trade_steps
        self._fast_trade_steps = fast_trade_steps

    @property
    def state(self) -> VillageState:
        return self._state.copy()

    def set_task_status(self, status: str) -> None:
        self._state.task_status = status

    def update_from_observation(self, bot_observation: Any, world_tick: int) -> VillageState:
        health, food, position, counts = _bot_values(bot_observation)
        self._state.world_tick = _number(world_tick, "world_tick", True)
        self._state.location = areas.location_of(position)
        self._state.route_progress = _route_progress(position)
        self._state.player_health = health
        self._state.player_food = food
        for field, value in counts.items():
            setattr(self._state, field, value)
        return self.state

    def apply_action(self, action: str, accepted: bool, result: Any = None) -> None:
        if not isinstance(action, str) or not isinstance(accepted, bool):
            raise ValueError("action must be a string and accepted must be boolean")
        if not accepted:
            return
        if action == "deliver":
            self._state.books_delivered += self._state.books
        elif action == "place_torch":
            self._state.lit_fraction = min(1.0, self._state.lit_fraction + 0.06)
        elif action == "build_gate":
            self._state.wall_fraction = min(1.0, self._state.wall_fraction + 0.1)
        elif action == "cure_villager":
            self._state.cure_progress += 1
            if self._state.cure_progress >= self._cure_steps:
                self._state.cure_progress = 0
                self._state.cured_villagers += 1
                self._state.zombie_villagers = max(0, self._state.zombie_villagers - 1)
        elif action == "build_golem":
            self._state.golem_progress += 1
            if self._state.golem_progress >= self._golem_build_steps:
                self._state.golem_progress = 0
                self._state.golems += 1
        elif action == "buy_book":
            self._state.trade_progress += 1
            needed = (
                self._fast_trade_steps
                if self._state.fast_trading
                else self._trade_steps
            )
            if self._state.trade_progress >= needed:
                self._state.trade_progress = 0
        elif action == "cheap_trade":
            self._state.trade_progress = 0
        elif action == "set_trade_hotkeys":
            self._state.fast_trading = True
        if action in ("buy_book", "cheap_trade"):
            if result is not None:
                result_data = _mapping(result, "result")
                if "price" in result_data:
                    self._state.book_price = _number(
                        result_data["price"], "result.price", True
                    )
        if action != "buy_book":
            self._state.trade_progress = 0
        if action != "cure_villager":
            self._state.cure_progress = 0
        if action != "build_golem":
            self._state.golem_progress = 0
        # Reputation stays at the mission start value because live telemetry
        # does not expose the lead's formula.
        # The real bot wears no armor, but the mission reports armor 4.0;
        # this discrepancy remains visible in the reported state.

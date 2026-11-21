from __future__ import annotations

import pytest

from env.minecraft.observation import StateTracker, build_start_state, inventory_counts
from village_sim.mission import VillageMission


MISSION = VillageMission()


def _digest():
    return {
        "tick": 12000,
        "villagers": [
            {"uuid": "1001,0,0,{}".format(index), "health": 20.0, "profession": "minecraft:librarian"}
            for index in range(1, 5)
        ],
        "zombie": {"present": True},
        "golem_count": 0,
        "hostile_count": 0,
        "raider_count": 0,
        "wall_gaps": {
            str(index): "minecraft:oak_fence_gate" if index < 6 else "minecraft:air"
            for index in range(10)
        },
        "chest": {"id": "minecraft:chest", "items": []},
    }


def _bot(position=(40, 64, 0)):
    return {
        "health": 20,
        "food": 14,
        "position": list(position),
        "inventory": [
            {"id": "emerald", "count": 20},
            {"id": "torch", "count": 4},
            {"id": "charcoal", "count": 2},
            {"id": "coal", "count": 4},
            {"id": "iron_block", "count": 4},
            {"id": "carved_pumpkin", "count": 1},
            {"id": "cooked_chicken", "count": 3},
            {"id": "splash_potion", "count": 1},
            {"id": "golden_apple", "count": 1},
        ],
    }


def test_build_start_state_maps_real_dusk_values():
    state = build_start_state(_digest(), _bot(), MISSION)
    assert state.world_tick == 12000
    assert state.location == "hall"
    assert state.villager_health == (20.0,) * 4
    assert state.zombie_villagers == 1
    assert state.wall_fraction == 0.6
    assert state.lit_fraction == 0.3
    assert state.emeralds == 20
    assert state.torches == 4
    assert state.cure_kits == 1
    assert state.book_price == 7
    assert state.reputation == 0.5
    assert state.armor == 4.0


def test_tracker_updates_live_fields_and_route_progress():
    tracker = StateTracker(build_start_state(_digest(), _bot(), MISSION), MISSION)
    state = tracker.update_from_observation(_bot((68, 64, 1)), 12100)
    assert state.location == "perimeter"
    assert state.route_progress == pytest.approx(24 / 106)
    changed = _bot((68, 64, 1))
    changed["health"] = 17
    changed["food"] = 9
    changed["inventory"][0]["count"] = 12
    state = tracker.update_from_observation(changed, 12200)
    assert state.player_health == 17
    assert state.player_food == 9
    assert state.emeralds == 12


def test_inventory_counts_and_cure_kit():
    inventory = _bot()["inventory"] + [
        {"id": "bow", "count": 1},
        {"id": "arrow", "count": 4},
    ]
    counts = inventory_counts(inventory)
    assert counts["emeralds"] == 20
    assert counts["cure_kits"] == 1
    with pytest.raises(ValueError):
        inventory_counts([{"id": "emerald", "count": -1}])


def test_tracker_actions_and_refusal():
    tracker = StateTracker(build_start_state(_digest(), _bot(), MISSION), MISSION)
    with_books = _bot()
    with_books["inventory"].append({"id": "enchanted_book", "count": 3})
    tracker.update_from_observation(with_books, 12000)
    tracker.apply_action("deliver", True)
    tracker.apply_action("place_torch", True)
    tracker.apply_action("build_gate", True)
    tracker.apply_action("set_trade_hotkeys", True)
    for _ in range(4):
        tracker.apply_action("cure_villager", True)
    state = tracker.state
    assert state.books_delivered == 3
    assert state.lit_fraction == pytest.approx(0.36)
    assert state.wall_fraction == pytest.approx(0.7)
    assert state.cured_villagers == 1
    assert state.zombie_villagers == 0
    assert state.fast_trading is True
    assert state.reputation == 0.5
    before = tracker.state
    tracker.apply_action("build_gate", False)
    assert tracker.state == before


def test_interleaved_actions_reset_progress():
    tracker = StateTracker(build_start_state(_digest(), _bot(), MISSION), MISSION)
    tracker.apply_action("cure_villager", True)
    tracker.apply_action("wait", True)
    tracker.apply_action("cure_villager", True)
    assert tracker.state.cure_progress == 1


def test_fast_trading_uses_one_buy_step():
    tracker = StateTracker(build_start_state(_digest(), _bot(), MISSION), MISSION)
    tracker.apply_action("set_trade_hotkeys", True)
    tracker.apply_action("buy_book", True)
    assert tracker.state.trade_progress == 0


def test_price_result_is_action_specific():
    tracker = StateTracker(build_start_state(_digest(), _bot(), MISSION), MISSION)
    tracker.apply_action("buy_book", True, result={"price": 5})
    assert tracker.state.book_price == 5
    tracker.apply_action("wait", True, result="done")


def test_malformed_bot_observation_raises():
    state = build_start_state(_digest(), _bot(), MISSION)
    tracker = StateTracker(state, MISSION)
    with pytest.raises(ValueError):
        tracker.update_from_observation({"health": 20, "food": 14, "inventory": []}, 12000)
    with pytest.raises(ValueError):
        tracker.update_from_observation(
            {"health": 20, "food": 14, "position": [40, 64, 0], "inventory": {}},
            12000,
        )

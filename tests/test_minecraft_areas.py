from __future__ import annotations

import pytest

from env.minecraft import areas


def test_location_of_real_layout_positions() -> None:
    assert areas.location_of(areas.player_start()) == "hall"
    assert areas.location_of(areas.golem_spot()) == "perimeter"
    assert areas.location_of(areas.delivery_chest()) == "settlement"
    assert areas.location_of((110, 63, 0)) == "route"
    assert areas.location_of((0, 64, 0)) == "route"
    assert areas.location_of(areas.villager_positions()[0]) == "perimeter"
    assert areas.location_of(areas.zombie_position()) == "perimeter"


def test_real_layout_position_collections() -> None:
    assert len(areas.gap_positions()) == 10
    assert len(areas.closed_gate_positions()) == 6
    assert areas.closed_gate_positions() <= areas.gap_positions()
    assert len(areas.torch_positions()) == 6
    assert len(areas.light_sample_positions()) == 4
    assert areas.protected_villager_uuids() == (
        "1001,0,0,1",
        "1001,0,0,2",
        "1001,0,0,3",
        "1001,0,0,4",
    )


def test_validate_rejects_missing_key() -> None:
    data = {"world": "dusk_v1"}
    with pytest.raises(ValueError, match="missing required key"):
        areas._validate(data)


def test_validate_rejects_reversed_bounds() -> None:
    data = {
        key: {"bounds": {"min": [0, 0, 0], "max": [1, 1, 1]}}
        for key in ("hall", "perimeter", "settlement")
    }
    data["village_bounds"] = {"min": [2, 0, 0], "max": [1, 1, 1]}
    for key in areas.REQUIRED_KEYS:
        data.setdefault(key, {})
    with pytest.raises(ValueError, match="village_bounds min"):
        areas._validate(data)

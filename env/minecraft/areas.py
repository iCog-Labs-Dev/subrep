from __future__ import annotations

import json
from pathlib import Path


AREAS_PATH = (
    Path(__file__).resolve().parents[2] / "minecraft" / "world" / "areas.json"
)
REQUIRED_KEYS = (
    "world",
    "spawn",
    "hall",
    "perimeter",
    "settlement",
    "route",
    "protected_villagers",
    "zombie_villager",
    "golem_spot",
    "torch_positions",
    "light_sample_positions",
    "village_bounds",
)
BOUNDS_KEYS = ("hall", "perimeter", "settlement", "village_bounds")


def _position(value, name):
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise ValueError("{} must be a 3-int list".format(name))
    return tuple(value)


def _validate(data):
    if not isinstance(data, dict):
        raise ValueError("areas.json must contain an object")
    for key in REQUIRED_KEYS:
        if key not in data:
            raise ValueError("missing required key: {}".format(key))

    for key in BOUNDS_KEYS:
        container = data[key]
        if not isinstance(container, dict):
            raise ValueError("malformed key: {}".format(key))
        value = container.get("bounds") if key != "village_bounds" else container
        if not isinstance(value, dict):
            raise ValueError("malformed key: {}".format(key))
        minimum = _position(value.get("min"), "{}.min".format(key))
        maximum = _position(value.get("max"), "{}.max".format(key))
        if any(low > high for low, high in zip(minimum, maximum)):
            raise ValueError("{} min must not exceed max".format(key))

    spawn = data["spawn"]
    if not isinstance(spawn, dict):
        raise ValueError("malformed key: spawn")
    _position([spawn.get("x"), spawn.get("y"), spawn.get("z")], "spawn")
    hall = data["hall"]
    if not isinstance(hall, dict):
        raise ValueError("malformed key: hall")
    _position(hall.get("player_start"), "hall.player_start")
    settlement = data["settlement"]
    if not isinstance(settlement, dict):
        raise ValueError("malformed key: settlement")
    _position(settlement.get("delivery_chest"), "settlement.delivery_chest")
    _position(data["golem_spot"], "golem_spot")

    for key in ("torch_positions", "light_sample_positions"):
        values = data[key]
        if not isinstance(values, list):
            raise ValueError("malformed key: {}".format(key))
        for index, value in enumerate(values):
            _position(value, "{}[{}]".format(key, index))

    route = data["route"]
    if not isinstance(route, dict):
        raise ValueError("malformed key: route")
    waypoints = route.get("waypoints")
    if not isinstance(waypoints, list):
        raise ValueError("malformed key: route.waypoints")
    for index, value in enumerate(waypoints):
        _position(value, "route.waypoints[{}]".format(index))

    for key in ("gap_positions", "closed_gate_positions"):
        perimeter = data["perimeter"]
        if not isinstance(perimeter, dict):
            raise ValueError("malformed key: perimeter")
        values = perimeter.get(key)
        if not isinstance(values, list):
            raise ValueError("malformed key: perimeter.{}".format(key))
        for index, value in enumerate(values):
            _position(value, "perimeter.{}[{}]".format(key, index))

    villagers = data["protected_villagers"]
    if not isinstance(villagers, list):
        raise ValueError("malformed key: protected_villagers")
    orders = []
    for index, villager in enumerate(villagers):
        if not isinstance(villager, dict):
            raise ValueError("malformed key: protected_villagers[{}]".format(index))
        orders.append(villager.get("order"))
        _position(
            villager.get("position"),
            "protected_villagers[{}].position".format(index),
        )
    if orders != [1, 2, 3, 4]:
        raise ValueError("protected_villagers order must be 1-4")
    zombie_villager = data["zombie_villager"]
    if not isinstance(zombie_villager, dict):
        raise ValueError("malformed key: zombie_villager")
    _position(zombie_villager.get("position"), "zombie_villager.position")
    return data


with AREAS_PATH.open(encoding="utf-8") as _areas_file:
    _DATA = _validate(json.load(_areas_file))


def _tuple_position(value):
    return tuple(value)


def _inside(position, bounds):
    minimum = bounds["min"]
    maximum = bounds["max"]
    return all(low <= value <= high for value, low, high in zip(position, minimum, maximum))


def location_of(position):
    position = _tuple_position(position)
    for name in ("hall", "perimeter", "settlement"):
        if _inside(position, _DATA[name]["bounds"]):
            return name
    return "route"


def player_start():
    return _tuple_position(_DATA["hall"]["player_start"])


def delivery_chest():
    return _tuple_position(_DATA["settlement"]["delivery_chest"])


def golem_spot():
    return _tuple_position(_DATA["golem_spot"])


def route_waypoints():
    return tuple(_tuple_position(item) for item in _DATA["route"]["waypoints"])


def gap_positions():
    return frozenset(_tuple_position(item) for item in _DATA["perimeter"]["gap_positions"])


def closed_gate_positions():
    return frozenset(
        _tuple_position(item) for item in _DATA["perimeter"]["closed_gate_positions"]
    )


def torch_positions():
    return tuple(_tuple_position(item) for item in _DATA["torch_positions"])


def light_sample_positions():
    return tuple(_tuple_position(item) for item in _DATA["light_sample_positions"])


def villager_positions():
    villagers = sorted(_DATA["protected_villagers"], key=lambda item: item["order"])
    return tuple(_tuple_position(item["position"]) for item in villagers)


def protected_villager_uuids():
    villagers = sorted(_DATA["protected_villagers"], key=lambda item: item["order"])
    return tuple(item["uuid"] for item in villagers)


def zombie_position():
    return _tuple_position(_DATA["zombie_villager"]["position"])


def village_bounds():
    bounds = _DATA["village_bounds"]
    return _tuple_position(bounds["min"]), _tuple_position(bounds["max"])


def is_gap(position):
    return _tuple_position(position) in gap_positions()


def is_closed_gate(position):
    return _tuple_position(position) in closed_gate_positions()

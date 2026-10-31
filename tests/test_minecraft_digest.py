from __future__ import annotations

import hashlib
import json

from env.minecraft.digest import AREAS_PATH, digest_for, _entity_from_line


LIBRARIAN = (
    '[12:00:00 INFO]: Librarian has the following entity data: '
    '{UUID: [I; 1, 2, 3, 4], Pos: [39.0d, 64.0d, 0.0d], Health: 20.0f, '
    'VillagerData: {profession: "minecraft:librarian"}, Offers: {}, '
    'Gossips: []}'
)
FARMER = (
    '[12:00:00 INFO]: Farmer has the following entity data: '
    '{UUID: [I; 5, 6, 7, 8], Pos: [41.0d, 64.0d, 2.0d], Health: 20.0f, '
    'VillagerData: {profession: "minecraft:farmer"}, Offers: {}, '
    'Gossips: []}'
)
ZOMBIE = (
    '[12:00:00 INFO]: Zombie Villager has the following entity data: '
    '{UUID: [I; 9, 10, 11, 12], Pos: [35.5d, 64.0d, -5.5d], '
    'Health: 20.0f, VillagerData: {profession: "minecraft:farmer"}}'
)
STICK = (
    '[12:00:00 INFO]: Stick has the following entity data: '
    '{Item: {id: "minecraft:stick", count: 1}, Health: 5s, '
    'UUID: [I; 13, 14, 15, 16], Pos: [9.5d, 89.0d, -15.0d]}'
)


def _observation():
    with AREAS_PATH.open(encoding="utf-8") as file:
        gaps = [tuple(item) for item in json.load(file)["perimeter"]["gap_positions"]]
    blocks = {position: {"id": "air", "items": []} for position in gaps}
    blocks.update(
        {
            (32, 64, 0): {"id": "torch", "items": []},
            (40, 64, 0): {"id": "torch", "items": []},
            (55, 64, 0): {"id": "chest", "items": []},
        }
    )
    return {
        "tick": 12000,
        "entities": [
            _entity_from_line(line) for line in [LIBRARIAN, FARMER, ZOMBIE, STICK]
        ],
        "blocks": blocks,
    }


def test_digest_is_deterministic() -> None:
    observation = _observation()
    assert digest_for(observation) == digest_for(observation)


def test_digest_uses_sorted_compact_canonical_json() -> None:
    observation = _observation()
    digest = digest_for(observation)
    assert digest == hashlib.sha256(
        json.dumps(
            {
                "tick": 12000,
                "player": None,
                "villagers": [
                    {
                        "uuid": "1,2,3,4",
                        "alive": True,
                        "health": 20.0,
                        "pos": [39, 64, 0],
                        "profession": "minecraft:librarian",
                    },
                    {
                        "uuid": "5,6,7,8",
                        "alive": True,
                        "health": 20.0,
                        "pos": [41, 64, 2],
                        "profession": "minecraft:farmer",
                    },
                ],
                "zombie": {
                    "present": True,
                    "health": 20.0,
                    "pos": [35, 64, -6],
                    "profession": "minecraft:farmer",
                },
                "golem_count": 0,
                "hostile_count": 0,
                "raider_count": 0,
                "other_entities": ["stick"],
                "raid_active": False,
                "light_samples": {
                    "40,64,0": "torch",
                    "36,64,-10": "air",
                    "44,64,-10": "air",
                    "36,64,10": "air",
                    "44,64,10": "air",
                    "32,64,0": "torch",
                    "48,64,0": "air",
                },
                "wall_gaps": {
                    ",".join(str(value) for value in position): "air"
                    for position in [
                        tuple(item)
                        for item in json.load(open(AREAS_PATH, encoding="utf-8"))[
                            "perimeter"
                        ]["gap_positions"]
                    ]
                },
                "chest": {"id": "chest", "items": []},
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_real_entity_lines_are_categorized() -> None:
    observation = _observation()
    digest = digest_for(observation)
    assert digest
    assert [entity["profession"] for entity in observation["entities"][:2]] == [
        "minecraft:librarian",
        "minecraft:farmer",
    ]
    assert observation["entities"][2]["health"] == 20.0
    assert observation["entities"][3]["health"] is None

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import re
import select
import time

from env.minecraft.reset_manager import (
    ALLOWED_CONSOLE_COMMANDS,
    BLOCK_PROBE_POSITIONS,
)


ROOT = Path(__file__).resolve().parents[2]
AREAS_PATH = ROOT / "minecraft" / "world" / "areas.json"
TORCH_POSITIONS = (
    (40, 64, 0),
    (36, 64, -10),
    (44, 64, -10),
    (36, 64, 10),
    (44, 64, 10),
    (32, 64, 0),
    (48, 64, 0),
)
GAP_COMMANDS = (
    "data get block 32 64 -8",
    "data get block 32 64 -7",
    "data get block 32 64 -6",
    "data get block 32 64 -5",
    "data get block 32 64 5",
    "data get block 32 64 6",
    "data get block 32 64 7",
    "data get block 32 64 8",
    "data get block 48 64 -8",
    "data get block 48 64 -7",
    "data get block 48 64 -6",
    "data get block 48 64 -5",
    "data get block 48 64 5",
    "data get block 48 64 6",
    "data get block 48 64 7",
    "data get block 48 64 8",
    "data get block 48 64 -1",
    "data get block 48 64 0",
    "data get block 48 64 1",
    "data get block 34 64 -10",
    "data get block 35 64 -10",
    "data get block 36 64 -10",
    "data get block 44 64 -10",
    "data get block 45 64 -10",
    "data get block 46 64 -10",
    "data get block 34 64 10",
    "data get block 35 64 10",
    "data get block 36 64 10",
    "data get block 44 64 10",
    "data get block 45 64 10",
    "data get block 46 64 10",
)
FIXED_BLOCK_COMMANDS = (
    "data get block 32 64 0",
    "data get block 40 64 0",
    "data get block 55 64 0",
)
HOSTILE_TYPES = frozenset(
    {
        "zombie",
        "skeleton",
        "spider",
        "creeper",
        "enderman",
        "witch",
        "slime",
        "cave_spider",
        "drowned",
        "husk",
        "stray",
        "phantom",
        "blaze",
        "ghast",
        "magma_cube",
        "silverfish",
        "zombie_pigman",
        "piglin_brute",
        "hoglin",
        "zoglin",
    }
)
RAIDER_TYPES = frozenset(
    {"pillager", "vindicator", "evoker", "ravager", "illusioner"}
)
UUID_PATTERN = re.compile(r"UUID:\s*\[I;\s*([^\]]+)\]")
NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
HEALTH_PATTERN = re.compile(r"Health:\s*(" + NUMBER + r")f")
POSITION_PATTERN = re.compile(
    r"Pos:\s*\[\s*("
    + NUMBER
    + r")d,\s*("
    + NUMBER
    + r")d,\s*("
    + NUMBER
    + r")d\s*\]"
)
PROFESSION_PATTERN = re.compile(r'profession:\s*"(minecraft:[^"]+)"')
BLOCK_PATTERN = re.compile(r"has the following block data:\s*(\{.*\})")
BLOCK_ID_PATTERN = re.compile(r'id:\s*"(minecraft:[^"]+)"')
PROBED_BLOCK_ID_PATTERN = re.compile(r"block_id=(minecraft:[a-z0-9_]+)")
ITEM_PATTERN = re.compile(
    r'\{\s*id:\s*"(minecraft:[^"]+)"\s*,\s*count:\s*(\d+)'
)
ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _read_until(manager, predicate, timeout):
    stdout = manager._process.stdout
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        ready, _, _ = select.select([stdout], [], [], remaining)
        if not ready:
            break
        line = stdout.readline()
        if not line:
            raise RuntimeError("server exited while reading observation")
        clean_line = ANSI_PATTERN.sub("", line)
        if predicate(clean_line):
            return clean_line
    raise TimeoutError("timed out waiting for observation output")


def _collect_batch(manager, silence_seconds):
    stdout = manager._process.stdout
    lines = []
    while True:
        ready, _, _ = select.select([stdout], [], [], silence_seconds)
        if not ready:
            return lines
        line = stdout.readline()
        if not line:
            raise RuntimeError("server exited while collecting entity data")
        lines.append(line)


def _entity_from_line(line):
    line = ANSI_PATTERN.sub("", line)
    line = " ".join(line.split())
    marker = " has the following entity data:"
    if marker not in line:
        raise ValueError("unparseable entity observation: {}".format(line.rstrip()))
    prefix = line.split(marker, 1)[0]
    entity_type = re.sub(r"^.*?\]:\s*", "", prefix).strip()
    if not entity_type:
        raise ValueError("entity display name is missing: {}".format(line.rstrip()))
    uuid_match = UUID_PATTERN.search(line)
    health_match = HEALTH_PATTERN.search(line)
    position_match = POSITION_PATTERN.search(line)
    if not uuid_match:
        raise ValueError("entity observation is missing UUID: {}".format(line.rstrip()))
    if not position_match:
        raise ValueError("entity observation is missing Pos: {}".format(line.rstrip()))
    uuid_parts = [part.strip() for part in uuid_match.group(1).split(",")]
    if len(uuid_parts) != 4 or any(
        re.fullmatch(r"-?\d+", part) is None for part in uuid_parts
    ):
        raise ValueError("entity observation has invalid UUID: {}".format(line.rstrip()))
    profession_match = PROFESSION_PATTERN.search(line)
    return {
        "type": entity_type,
        "uuid": ",".join(uuid_parts),
        "health": float(health_match.group(1)) if health_match else None,
        "pos": [float(value) for value in position_match.groups()],
        "profession": profession_match.group(1) if profession_match else None,
        "has_villager_data": "VillagerData:" in line,
        "has_offers": "Offers:" in line,
        "has_gossips": "Gossips:" in line,
    }


def _entity_lines(lines):
    records = []
    current = []
    for line in lines:
        if " has the following entity data:" in line:
            if current:
                records.append(" ".join(current))
            current = [line.rstrip()]
        elif current:
            current.append(line.rstrip())
    if current:
        records.append(" ".join(current))
    return records


def _block_from_line(line, position, chest=False):
    match = BLOCK_PATTERN.search(line)
    if not match:
        raise ValueError("unparseable block observation at {}".format(position))
    block_id = BLOCK_ID_PATTERN.search(match.group(1))
    if not block_id:
        raise ValueError("block observation is missing id at {}".format(position))
    items = []
    if chest:
        if "Items: []" not in match.group(1):
            items = [
                {"id": item.group(1), "count": int(item.group(2))}
                for item in ITEM_PATTERN.finditer(match.group(1))
            ]
            if not items:
                raise ValueError("chest Items cannot be parsed at {}".format(position))
    return {"id": block_id.group(1), "items": items}


def _probe_block(manager, position, block_id):
    x, y, z = position
    command = (
        "execute if block {} {} {} minecraft:{} run "
        "say block_id=minecraft:{}"
    ).format(x, y, z, block_id, block_id)
    manager.send_command(command)
    line = _read_until(
        manager, lambda value: "block_id=minecraft:" in value, 30
    )
    match = PROBED_BLOCK_ID_PATTERN.search(line)
    if not match:
        raise ValueError("block probe is missing an id at {}".format(position))
    return {"id": match.group(1), "items": []}


def observe(manager):
    manager.send_command("time query daytime")
    time_line = _read_until(manager, lambda line: "The time is " in line, 30)
    time_match = re.search(r"The time is (\d+)", time_line)
    if not time_match:
        raise ValueError("time observation is missing the tick")
    tick = int(time_match.group(1))
    if tick != 12000:
        raise ValueError("expected time 12000, got {}".format(tick))

    manager.send_command("execute as @e run data get entity @s")
    entity_records = _entity_lines(_collect_batch(manager, 3.0))
    if not entity_records:
        raise ValueError("entity observation returned no entities")
    entities = [_entity_from_line(line) for line in entity_records]

    with AREAS_PATH.open(encoding="utf-8") as areas_file:
        gap_positions = [tuple(position) for position in json.load(areas_file)["perimeter"]["gap_positions"]]
    block_observations = {}
    if len(gap_positions) != len(GAP_COMMANDS):
        raise ValueError("areas.json gap count does not match fixed observation commands")
    probe_ids = dict(BLOCK_PROBE_POSITIONS)
    for position in gap_positions + [(32, 64, 0), (40, 64, 0)]:
        block_observations[position] = _probe_block(
            manager, position, probe_ids[position]
        )
    manager.send_command("data get block 55 64 0")
    line = _read_until(manager, lambda value: "has the following block data:" in value, 30)
    position = (55, 64, 0)
    block_observations[position] = _block_from_line(
        line, position, chest=True
    )

    return {"tick": tick, "entities": entities, "blocks": block_observations}


def digest_for(observation):
    villagers = []
    zombie = None
    golem_count = 0
    hostile_count = 0
    raider_count = 0
    other_entities = []
    for entity in observation["entities"]:
        entity_type = entity["type"].lower().replace(" ", "_")
        is_villager = entity["has_villager_data"] and (
            entity["has_offers"] or entity["has_gossips"]
        )
        is_zombie_villager = entity_type == "zombie_villager"
        if is_villager or is_zombie_villager:
            if entity["health"] is None or entity["profession"] is None:
                raise ValueError(
                    "{} observation is missing health or profession".format(
                        entity["type"]
                    )
                )
        if is_villager:
            villagers.append(
                {
                    "uuid": entity["uuid"],
                    "alive": entity["health"] > 0,
                    "health": round(entity["health"], 2),
                    "pos": [math.floor(value) for value in entity["pos"]],
                    "profession": entity["profession"],
                }
            )
        elif is_zombie_villager:
            zombie = {
                "present": True,
                "health": round(entity["health"], 2),
                "pos": [math.floor(value) for value in entity["pos"]],
                "profession": entity["profession"],
            }
        elif entity_type == "iron_golem":
            golem_count += 1
        elif entity_type in HOSTILE_TYPES:
            hostile_count += 1
        elif entity_type in RAIDER_TYPES:
            raider_count += 1
        else:
            other_entities.append(entity_type)

    blocks = observation["blocks"]
    light_samples = {
        "{},{},{}".format(*position): blocks[position]["id"]
        for position in TORCH_POSITIONS
    }
    gap_positions = []
    with AREAS_PATH.open(encoding="utf-8") as areas_file:
        gap_positions = [tuple(position) for position in json.load(areas_file)["perimeter"]["gap_positions"]]
    wall_gaps = {
        "{},{},{}".format(*position): blocks[position]["id"]
        for position in gap_positions
    }
    chest = blocks[(55, 64, 0)]
    result = {
        "tick": int(observation["tick"]),
        "player": None,
        "villagers": sorted(villagers, key=lambda value: value["uuid"]),
        "zombie": zombie if zombie is not None else {"present": False},
        "golem_count": golem_count,
        "hostile_count": hostile_count,
        "raider_count": raider_count,
        "other_entities": sorted(other_entities),
        "raid_active": False,
        "light_samples": light_samples,
        "wall_gaps": wall_gaps,
        "chest": {"id": chest["id"], "items": chest["items"]},
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

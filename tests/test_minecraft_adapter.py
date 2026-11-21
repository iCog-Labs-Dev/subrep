from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from env.minecraft import areas
from env.minecraft.adapter import MinecraftVillageEnv
from env.minecraft.sidecar_client import SidecarClient


class FakeSidecar:
    def __init__(self):
        self.actions = []
        self.emergency_stopped = False
        self.fail_actions = set()
        self.observation = {
            "status": "ok",
            "position": [40, 64, 0],
            "health": 20,
            "food": 14,
            "inventory": [
                {"id": "minecraft:emerald", "count": 20},
                {"id": "minecraft:torch", "count": 4},
                {"id": "minecraft:charcoal", "count": 2},
                {"id": "minecraft:coal", "count": 4},
                {"id": "minecraft:iron_block", "count": 4},
                {"id": "minecraft:pumpkin", "count": 1},
                {"id": "minecraft:cooked_beef", "count": 3},
                {"id": "minecraft:golden_apple", "count": 1},
            ],
        }

    def status(self):
        return {"spawned": True, "token_expires_at": "2099-01-01T00:00:00Z"}

    def observe(self, **kwargs):
        return dict(self.observation)

    def act(self, action, **kwargs):
        self.actions.append(action)
        if action in self.fail_actions:
            return {"status": "error", "reason": "refused"}
        return dict(self.observation)

    def emergency_stop(self):
        self.emergency_stopped = True
        return {"status": "ok"}


class RaisingSidecar(FakeSidecar):
    def act(self, action, **kwargs):
        self.actions.append(action)
        raise RuntimeError("connection failed")


class FakeResetManager:
    def reset(self):
        return 0

    def stop(self):
        pass


def _raw_digest():
    entities = [
        {
            "type": "player",
            "uuid": "0,0,0,0",
            "health": 20.0,
            "food": 14,
            "pos": [40.0, 64.0, 0.0],
            "profession": None,
            "has_villager_data": False,
            "has_offers": False,
            "has_gossips": False,
        }
    ]
    for uuid, position in zip(
        areas.protected_villager_uuids(), areas.villager_positions()
    ):
        entities.append({
            "type": "villager",
            "uuid": uuid,
            "health": 20.0,
            "food": None,
            "pos": list(position),
            "profession": "minecraft:librarian",
            "has_villager_data": True,
            "has_offers": True,
            "has_gossips": False,
        })
    entities.append({
        "type": "zombie_villager",
        "uuid": "1001,0,0,5",
        "health": 20.0,
        "food": None,
        "pos": [67.0, 64.0, 8.0],
        "profession": "minecraft:farmer",
        "has_villager_data": False,
        "has_offers": False,
        "has_gossips": False,
    })
    blocks = {}
    for index, position in enumerate(areas.gap_positions()):
        blocks[position] = {
            "id": "minecraft:oak_fence_gate" if index < 6 else "minecraft:air",
            "items": [],
        }
    for position in areas.torch_positions():
        blocks[position] = {"id": "minecraft:torch", "items": []}
    for position in areas.light_sample_positions():
        blocks.setdefault(position, {"id": "minecraft:air", "items": []})
    blocks[areas.delivery_chest()] = {"id": "minecraft:chest", "items": []}
    return {"tick": 12000, "entities": entities, "blocks": blocks}


def make_env(sidecar=None, run_id="adapter-test"):
    sidecar = sidecar or FakeSidecar()
    digest_patch = patch(
        "env.minecraft.adapter.digest.observe",
        return_value=_raw_digest(),
    )
    digest_patch.start()
    env = MinecraftVillageEnv(
        sidecar=sidecar,
        reset_manager=FakeResetManager(),
        run_id=run_id,
    )
    env._digest_patch = digest_patch
    return env, sidecar


def close_env(env):
    env.close()
    env._digest_patch.stop()


def test_reset_returns_dusk_state_and_missing_items():
    env, _ = make_env(run_id="reset-state")
    try:
        state, info = env.reset()
        assert state.world_tick == 12000
        assert state.location == "hall"
        assert state.emeralds == 20
        assert state.books == 0
        assert state.reputation == 0.5
        assert state.task_status == "open"
        assert info["missing_start_items"] == [
            "splash_potion", "bow", "arrow", "stick",
            "oak_fence_gate", "carved_pumpkin",
        ]
    finally:
        close_env(env)


def test_reset_rejects_state_injection():
    env, _ = make_env(run_id="reject-state")
    try:
        with pytest.raises(ValueError):
            env.reset(options={"state": object()})
    finally:
        close_env(env)


def test_wait_returns_contract_tuple():
    env, sidecar = make_env(run_id="wait")
    try:
        env.reset()
        result = env.step("wait")
        assert len(result) == 5
        _, motives, terminated, truncated, info = result
        assert motives.dtype == np.float32
        assert motives.shape == (6,)
        assert not terminated
        assert not truncated
        assert "task_payoff" in info
        assert "task_reward" in info
        assert "raid_active" in info
        assert "events" in info
        assert info["tick_before"] == 12000
        assert info["tick_after"] == 12100
        assert info["action_ticks"] == 100
        assert info["run_id"] == "wait"
        assert sidecar.actions == ["wait"]
    finally:
        close_env(env)


def test_step_rejects_unknown_action_without_sidecar_call():
    env, sidecar = make_env(run_id="unknown")
    try:
        env.reset()
        with pytest.raises(ValueError):
            env.step("unknown")
        assert sidecar.actions == []
    finally:
        close_env(env)


def test_step_rejects_hall_preconditions():
    env, sidecar = make_env(run_id="preconditions")
    try:
        env.reset()
        with pytest.raises(ValueError):
            env.step("deliver")
        with pytest.raises(ValueError):
            env.step("buy_book")
        assert sidecar.actions == []
    finally:
        close_env(env)


def test_step_before_reset_raises():
    env, _ = make_env(run_id="before-reset")
    try:
        with pytest.raises(ValueError):
            env.step("wait")
    finally:
        close_env(env)


def test_shoot_is_allowed_without_threats():
    env, sidecar = make_env(run_id="shoot")
    try:
        env.reset()
        env.step("shoot")
        assert sidecar.actions == ["shoot"]
    finally:
        close_env(env)


def test_cure_second_call_does_not_require_cure_kit():
    env, sidecar = make_env(run_id="cure")
    try:
        env.reset()
        env._tracker._state.location = "perimeter"
        env._tracker._state.cure_progress = 1
        env._tracker._state.cure_kits = 0
        env.step("cure_villager")
        assert sidecar.actions == ["cure_villager"]
    finally:
        close_env(env)


def test_build_golem_moves_to_perimeter_first():
    env, sidecar = make_env(run_id="golem")
    try:
        env.reset()
        env.step("build_golem")
        assert sidecar.actions[:2] == ["move_perimeter", "build_golem"]
    finally:
        close_env(env)


def test_deliver_sets_payoff_after_total_books():
    env, _ = make_env(run_id="deliver")
    try:
        env.reset()
        env._tracker._state.location = "settlement"
        env._tracker._state.books = 3
        state, _, _, _, info = env.step("deliver")
        assert state.task_status == "delivered"
        assert info["task_payoff"] == 1.0
    finally:
        close_env(env)


def test_link_failure_stops_sidecar():
    sidecar = RaisingSidecar()
    env, _ = make_env(sidecar=sidecar, run_id="link-failure")
    try:
        env.reset()
        with pytest.raises(RuntimeError):
            env.step("wait")
        assert sidecar.emergency_stopped
    finally:
        close_env(env)


def test_refused_step_is_logged():
    run_id = "refused-{}".format(os.getpid())
    path = Path(__file__).resolve().parents[1] / "minecraft" / "logs" / (run_id + ".jsonl")
    env, sidecar = make_env(run_id=run_id)
    sidecar.fail_actions.add("wait")
    try:
        env.reset()
        env.step("wait")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records[-1]["accepted"] is False
    finally:
        close_env(env)
        if path.exists():
            path.unlink()


@pytest.mark.real_server
def test_adapter_real_server_smoke():
    if not os.environ.get("SUBREP_REAL_SERVER") or not os.environ.get("SUBREP_SIDECAR_TOKEN"):
        pytest.skip("real Minecraft server is not configured")
    env = MinecraftVillageEnv(
        sidecar=SidecarClient(token=os.environ["SUBREP_SIDECAR_TOKEN"]),
    )
    try:
        env.reset()
        for action in ("wait", "move_perimeter", "move_hall"):
            result = env.step(action)
            assert len(result) == 5
            assert result[1].dtype == np.float32
            assert result[1].shape == (6,)
            assert "task_payoff" in result[4]
    finally:
        env.close()

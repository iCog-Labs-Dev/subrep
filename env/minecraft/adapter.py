from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional, Tuple

from village_sim.env import ACTIONS
from village_sim.mission import VillageMission, VillageModelParams
from village_sim.motives import MOTIVE_NAMES, phi
from village_sim.state import TASK_DELIVERED, TASK_FAILED, VillageState

from env.minecraft import areas, digest
from env.minecraft.observation import StateTracker, build_start_state
from env.minecraft.reset_manager import ResetManager
from env.minecraft.sidecar_client import SidecarClient
from env.minecraft.step_log import StepLogger
from env.minecraft.telemetry import validate_action_result, validate_observation


class MinecraftVillageEnv:
    def __init__(
        self,
        seed=None,
        mission=None,
        params=None,
        sidecar=None,
        reset_manager=None,
        run_id=None,
        snapshot_id=None,
    ):
        self.mission = mission or VillageMission()
        self.params = params or VillageModelParams()
        self.sidecar = sidecar or SidecarClient(token=os.environ.get("SUBREP_BOT_TOKEN", ""))
        self.reset_manager = reset_manager or ResetManager()
        self.run_id = run_id or str(uuid.uuid4())
        self._msg_counter = 0
        self.snapshot_id = snapshot_id or "dusk_v1"
        self._step_logger = StepLogger(self.run_id)
        self.state = None
        self._tracker = None
        self._step = 0
        self._terminated = False
        self._world_tick = self.mission.dusk_tick
        self._reset_done = False

    @property
    def metadata(self):
        return {
            "environment_id": "minecraft_village_v1",
            "mission_id": self.mission.mission_id,
            "motive_names": list(MOTIVE_NAMES),
            "motive_schema_version": "2.0",
            "payoff_schema_version": "2.0",
            "observation_schema_version": "2.0",
            "action_schema_version": "2.0",
        }

    def _link_failure(self, error):
        try:
            self.sidecar.emergency_stop()
        except RuntimeError:
            pass
        self._terminated = True
        raise RuntimeError("Minecraft sidecar link failed: {}".format(error)) from error

    def _next_message_id(self, kind):
        self._msg_counter += 1
        return "{}-{}-{}".format(self.run_id, kind, self._msg_counter)


    def _observe(self):
        try:
            observation = self.sidecar.observe(
                message_id=self._next_message_id("observe"),
                snapshot_id=self.snapshot_id,
                world_tick=self._world_tick,
            )
            return validate_observation(observation)
        except (RuntimeError, ValueError, TypeError) as error:
            self._link_failure(error)

    def reset(self, seed=None, options=None):
        if isinstance(options, dict) and "state" in options:
            raise ValueError("Minecraft adapter does not accept state injection")
        options = options or {}
        self.snapshot_id = options.get("snapshot_id", "dusk_v1")
        try:
            self.reset_manager.reset()
            status = self.sidecar.status()
            if not status.get("spawned") or status.get("token_expires_at") is None:
                raise RuntimeError("Minecraft bot is not spawned or token is expired")
            # observe() returns raw telemetry; _digest_payload builds the dict
            # required by build_start_state.
            raw_observation = digest.observe(self.reset_manager)
            digest_payload = digest._digest_payload(raw_observation)
            bot_observation = self._observe()
            start = build_start_state(digest_payload, bot_observation, self.mission)
        except (RuntimeError, ValueError, TypeError) as error:
            self._link_failure(error)
        self._tracker = StateTracker(
            start,
            self.mission,
            golem_build_steps=self.params.golem_build_steps,
            cure_steps=2,
            trade_steps=self.params.trade_steps,
            fast_trade_steps=self.params.fast_trade_steps,
        )
        self.state = start.copy()
        self._world_tick = self.mission.dusk_tick
        self._step = 0
        self._terminated = False
        self._reset_done = True
        inventory = {}
        for entry in bot_observation["inventory"]:
            item = entry["id"].split(":", 1)[-1]
            inventory[item] = inventory.get(item, 0) + entry["count"]
        missing = [
            item for item in (
                "splash_potion", "bow", "arrow", "stick",
                "oak_fence_gate", "carved_pumpkin",
            ) if inventory.get(item, 0) == 0
        ]
        return self.state.copy(), {
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "missing_start_items": missing,
        }

    def _validate(self, action):
        if self.state is None or self._tracker is None:
            raise ValueError("reset() must be called before step()")
        if not isinstance(action, str) or action not in ACTIONS:
            raise ValueError("Unknown village action: {!r}".format(action))
        if self._terminated:
            raise ValueError("The night is over; reset() before stepping again")
        s = self._tracker.state
        if s.night_over(self.mission):
            raise ValueError("The night is over; reset() before stepping again")
        in_village = s.location in ("hall", "perimeter")
        requirements = {
            "move_hall": s.location == "perimeter",
            "move_perimeter": s.location == "hall",
            "walk_route": s.location in ("hall", "route"),
            "sprint_route": s.location in ("hall", "route") and s.player_food > 6.0,
            "deliver": s.location == "settlement" and s.books > 0,
            "craft_torches": s.charcoal + s.coal > 0,
            "place_torch": s.location == "perimeter" and s.torches > 0,
            "build_gate": s.location == "perimeter" and s.wall_fraction < 1.0,
            "build_golem": (
                in_village
                and (s.iron_blocks >= 4 if s.golem_progress == 0 else s.pumpkins > 0)
            ),
            "buy_book": s.location == "perimeter" and s.trading_access and s.emeralds >= s.book_price,
            "cheap_trade": s.location == "perimeter" and s.trading_access and s.emeralds >= 1,
            "cure_villager": (
                s.location == "perimeter"
                and s.zombie_villagers > 0
                and (s.cure_progress > 0 or s.cure_kits > 0)
            ),
            "set_trade_hotkeys": not s.fast_trading,
            "shoot": in_village,
            "eat": s.food_items > 0 and s.player_food < 20.0,
        }
        if action == "buy_book" and s.trade_progress > 0:
            requirements[action] = True
        if not requirements.get(action, True):
            raise ValueError("Cannot {}: precondition is unmet".format(action))

    def step(self, action):
        self._validate(action)
        tick_before = self._world_tick
        state_before = self._tracker.state
        internal_result = None
        if action == "build_golem" and self._tracker.state.location != "perimeter":
            observation = self._observe()
            position = observation["position"]
            perimeter = areas._DATA["perimeter"]["bounds"]
            minimum, maximum = tuple(perimeter["min"]), tuple(perimeter["max"])
            if not all(low <= value <= high for value, low, high in zip(position, minimum, maximum)):
                internal_result = self._act("move_perimeter")
        if internal_result is not None and internal_result.get("status") == "error":
            result = internal_result
        else:
            try:
                result = self.sidecar.act(
                    action,
                    message_id=self._next_message_id("act"),
                    snapshot_id=self.snapshot_id,
                    world_tick=tick_before,
                )
            except RuntimeError as error:
                self._link_failure(error)
        if not isinstance(result, dict):
            self._link_failure(ValueError("sidecar action result is not a dict"))
        try:
            result = validate_action_result(result)
        except ValueError as error:
            self._link_failure(error)
        if result.get("status") == "error":
            fresh = self._observe()
            accepted = False
            refusal = result.get("reason")
        else:
            fresh = self._observe()
            tracker_result = result
            if "book_price" in result and "price" not in result:
                tracker_result = dict(result, price=result["book_price"])
            self._tracker.apply_action(action, True, tracker_result)
            accepted = True
            refusal = None
        self._world_tick += self.mission.ticks_per_step
        self._tracker.update_from_observation(fresh, self._world_tick)
        self._step += 1
        self.state = self._tracker.state
        payoff = 0.0
        if (
            self.state.task_status == "open"
            and self.state.books_delivered * self.mission.book_value >= self.mission.contract_value
            and self.state.contract_valid
            and self._world_tick < self.mission.sunrise_tick
        ):
            self._tracker.set_task_status(TASK_DELIVERED)
            self.state = self._tracker.state
            payoff = 1.0
        elif (
            self.state.task_status == "open"
            and (
                self.state.player_health <= 0
                or not self.state.contract_valid
                or self._world_tick >= self.mission.sunrise_tick
            )
        ):
            self._tracker.set_task_status(TASK_FAILED)
            self.state = self._tracker.state
            payoff = -1.0
        terminated = self.state.night_over(self.mission)
        self._terminated = terminated
        info = {
            "tick_before": tick_before,
            "tick_after": self._world_tick,
            "action_ticks": self.mission.ticks_per_step,
            "run_id": self.run_id,
            "accepted": accepted,
            "refusal": refusal,
            "events": [action] if accepted else ["refused:{}".format(refusal)],
            "task_payoff": payoff,
            "task_reward": payoff,
            "raid_active": self.state.raid_active,
        }
        self._step_logger.log_step(
            run_id=self.run_id,
            snapshot_id=self.snapshot_id,
            step=self._step - 1,
            tick_before=tick_before,
            tick_after=self._world_tick,
            action=action,
            accepted=accepted,
            refusal=refusal,
            action_ticks=self.mission.ticks_per_step,
            finished=terminated,
            state_before=state_before,
            state_after=self.state,
            task_payoff=payoff,
            events=info["events"],
        )
        return self.state.copy(), phi(self.state, self.mission, self.params), terminated, False, info

    def _act(self, action):
        try:
            return self.sidecar.act(
                action,
                message_id="{}-internal-{}".format(self.run_id, self._step),
                snapshot_id=self.snapshot_id,
                world_tick=self._world_tick,
            )
        except RuntimeError as error:
            self._link_failure(error)

    def close(self):
        if self._reset_done:
            try:
                self.reset_manager.stop()
            finally:
                self.state = None
                self._reset_done = False
        else:
            self.state = None

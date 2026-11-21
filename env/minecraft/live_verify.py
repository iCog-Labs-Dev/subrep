from __future__ import annotations

import os
import time
from pathlib import Path

from env.minecraft.adapter import MinecraftVillageEnv
from env.minecraft.sidecar_client import SidecarClient


GIVE_COMMANDS = (
    '/give subrep_bot minecraft:splash_potion[minecraft:potion_contents={potion:"minecraft:weakness"}] 1',
    "/give subrep_bot minecraft:bow 1",
    "/give subrep_bot minecraft:arrow 8",
    "/give subrep_bot minecraft:stick 16",
    "/give subrep_bot minecraft:oak_fence_gate 4",
    "/give subrep_bot minecraft:carved_pumpkin 1",
)
SWEEP_ACTIONS = (
    "wait",
    "move_perimeter",
    "buy_book",
    "buy_book",
    "set_trade_hotkeys",
    "cheap_trade",
    "cure_villager",
    "cure_villager",
    "craft_torches",
    "place_torch",
    "build_gate",
    "build_golem",
    "build_golem",
    "shoot",
    "eat",
    "move_hall",
    "walk_route",
    "sprint_route",
    "sprint_route",
    "sprint_route",
    "deliver",
)


def _environment() -> MinecraftVillageEnv:
    token = os.environ.get("SUBREP_SIDECAR_TOKEN")
    if not token:
        raise RuntimeError("SUBREP_SIDECAR_TOKEN is required")
    return MinecraftVillageEnv(sidecar=SidecarClient(token=token))


def _sweep():
    env = _environment()
    rows = []
    run_id = None
    try:
        _, info = env.reset()
        for command in GIVE_COMMANDS:
            env.reset_manager.send_command(command)
        run_id = info["run_id"]
        print("run 1 missing_start_items:", info["missing_start_items"])
        print("run 1:", run_id)
        for action in SWEEP_ACTIONS:
            started = time.monotonic()
            try:
                _, _, _, _, step_info = env.step(action)
                rows.append({
                    "action": action,
                    "accepted": step_info.get("accepted", True),
                    "refusal": step_info.get("refusal"),
                    "duration_seconds": time.monotonic() - started,
                    "tick_after": step_info["tick_after"],
                    "task_payoff": step_info["task_payoff"],
                })
            except ValueError as error:
                rows.append({
                    "action": action,
                    "accepted": False,
                    "refusal": str(error),
                    "duration_seconds": time.monotonic() - started,
                    "tick_after": env._world_tick,
                    "task_payoff": 0.0,
                })
            except RuntimeError as error:
                print("run 1 link failure:", error)
                break
    finally:
        env.close()
    return rows, run_id



def _night():
    env = _environment()
    started = time.monotonic()
    steps = 0
    try:
        _, info = env.reset()
        run_id = info["run_id"]
        terminated = False
        final_state = env.state
        while not terminated and steps < 120:
            final_state, _, terminated, _, _ = env.step("wait")
            steps += 1
        duration = time.monotonic() - started
        log_path = Path(__file__).resolve().parents[2] / "minecraft" / "logs" / (run_id + ".jsonl")
        log_lines = len(log_path.read_text(encoding="utf-8").splitlines()) if log_path.exists() else 0
        return run_id, steps, final_state.world_tick, final_state.task_status, duration, log_lines
    finally:
        env.close()


def main() -> int:
    rows, sweep_run_id = _sweep()
    print("\nAction sweep")
    print("action | accepted | refusal | duration_seconds | tick_after | task_payoff")
    for row in rows:
        print(
            "{} | {} | {} | {:.3f} | {} | {}".format(
                row["action"],
                row["accepted"],
                row["refusal"] or "",
                row["duration_seconds"],
                row["tick_after"],
                row["task_payoff"],
            )
        )
    print("run_id: sweep={}".format(sweep_run_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

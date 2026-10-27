"""Measure how reliably the LunarLander sub-skills fly complete landings.

Flies two fixed sub-skill sequences from the same seeds and reports how often
each sub-skill completes, how each episode ends, and the fuel used.

    python -m demo.run_lunar_subskill_reliability
    python -m demo.run_lunar_subskill_reliability --seeds 100
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from env.lunar_facts import OUTCOMES, landing_outcome
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import SUBSKILLS, run_subskill

SEQUENCES: Dict[str, List[str]] = {
    "careful_landing": ["stabilise", "move_over_pad", "slow_descent", "touch_down"],
    "fuel_saving_landing": ["stabilise", "move_over_pad", "free_fall", "brake", "touch_down"],
}
# Once the steps are done the lander is settling; this sub-skill holds it until
# the episode ends so every flight has an outcome.
SETTLING_SKILL = "touch_down"


@dataclass(frozen=True)
class Flight:
    """One flight of one sub-skill sequence."""

    seed: int
    outcome: str
    completed_steps: List[str]
    failed_step: Optional[str]
    fuel: float
    frames: int


def fly(env: SubRepEnv, steps: Sequence[str], seed: int) -> Flight:
    """Fly ``steps`` in order from ``seed``, then let the lander settle."""
    obs, _ = env.reset(seed=seed)
    completed: List[str] = []
    failed: Optional[str] = None
    fuel = 0.0
    frames = 0
    info: Optional[dict] = None

    for name in steps:
        run = run_subskill(env, SUBSKILLS[name], obs)
        obs, frames, fuel = run.observation, frames + run.frames, fuel + float(run.rewards[:, 1].sum())
        if run.terminated or run.truncated:
            info = run.info
        if run.completed:
            completed.append(name)
        else:
            failed = name
        if info is not None or failed is not None:
            break

    settle = SUBSKILLS[SETTLING_SKILL]
    while info is None:
        obs, reward, terminated, truncated, step_info = env.step(int(settle.policy(obs)))
        fuel += float(reward[1])
        frames += 1
        if terminated or truncated:
            info = step_info

    return Flight(
        seed=seed,
        outcome=landing_outcome(info, obs),
        completed_steps=completed,
        failed_step=failed,
        fuel=fuel,
        frames=frames,
    )


def summarise(flights: Sequence[Flight], steps: Sequence[str]) -> dict:
    """Outcome rates, mean fuel, and the completion rate of each step."""
    count = len(flights)
    attempts = {name: 0 for name in steps}
    successes = {name: 0 for name in steps}
    for flight in flights:
        for name in flight.completed_steps:
            attempts[name] += 1
            successes[name] += 1
        if flight.failed_step is not None:
            attempts[flight.failed_step] += 1
    return {
        "flights": count,
        **{f"{o}_rate": sum(f.outcome == o for f in flights) / count for o in OUTCOMES},
        "mean_fuel": float(np.mean([f.fuel for f in flights])),
        "step_completion": {
            name: {"completed": successes[name], "attempted": attempts[name]} for name in steps
        },
    }


def run(seeds: Sequence[int]) -> Dict[str, dict]:
    report = {}
    env = SubRepEnv(seed=int(seeds[0]))
    for name, steps in SEQUENCES.items():
        flights = [fly(env, steps, int(seed)) for seed in seeds]
        report[name] = {"summary": summarise(flights, steps), "flights": [asdict(f) for f in flights]}
    env.close()
    return report


def to_markdown(report: Dict[str, dict], seeds: Sequence[int]) -> str:
    lines = [
        "# LunarLander sub-skill reliability",
        "",
        f"Seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} flights per sequence).",
        "",
        "| Sequence | On pad | Off pad | Crashed | Timeout | Mean Fuel | Step completion |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, data in report.items():
        s = data["summary"]
        steps = ", ".join(
            f"{step} {c['completed']}/{c['attempted']}" for step, c in s["step_completion"].items()
        )
        lines.append(
            f"| {name} | {s['landed_on_pad_rate']:.2f} | {s['landed_off_pad_rate']:.2f} | "
            f"{s['crashed_rate']:.2f} | {s['timeout_rate']:.2f} | {s['mean_fuel']:.1f} | {steps} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LunarLander sub-skill reliability.")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--output", default="demo/artifacts/lunar_subskill_reliability.json")
    parser.add_argument("--markdown-output", default="demo/artifacts/lunar_subskill_reliability.md")
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    report = run(seeds)
    markdown = to_markdown(report, seeds)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"seeds": seeds, "sequences": report}, indent=2), encoding="utf-8")
    Path(args.markdown_output).write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {output} and {args.markdown_output}")


if __name__ == "__main__":
    main()

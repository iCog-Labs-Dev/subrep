"""Measure the LunarLander controllers that goal decomposition is compared against.

Runs the idle policy, Gymnasium's built-in landing heuristic and the PPO pilot
on the same seeds, and reports how often each lands on the pad, lands beside it,
crashes or runs out of time, together with its Safety and Fuel totals.

    python -m demo.run_lunar_baseline
    python -m demo.run_lunar_baseline --seeds 100 --start-seed 0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

from env.lunar_lander_wrapper import SubRepEnv

Policy = Callable[[np.ndarray], int]

OUTCOMES = ("landed_on_pad", "landed_off_pad", "crashed", "timeout")
# The landing pad spans x in [-0.2, 0.2] in observation coordinates.
PAD_HALF_WIDTH = 0.2
DEFAULT_PPO_CHECKPOINT = "models/pilot_ppo.pt"


@dataclass(frozen=True)
class EpisodeResult:
    """One episode of one controller."""

    seed: int
    outcome: str
    safety: float
    fuel: float
    frames: int
    final_x: float


def run_episode(env: SubRepEnv, policy: Policy, seed: int) -> EpisodeResult:
    """Run one episode from `seed` and classify how it ended."""
    obs, _ = env.reset(seed=seed)
    totals = np.zeros(2, dtype=np.float64)
    frames = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        totals += reward
        frames += 1
        if terminated or truncated:
            break

    if info["landing_success"]:
        outcome = "landed_on_pad" if abs(float(obs[0])) <= PAD_HALF_WIDTH else "landed_off_pad"
    elif info["crashed"]:
        outcome = "crashed"
    else:
        outcome = "timeout"
    return EpisodeResult(
        seed=seed,
        outcome=outcome,
        safety=float(totals[0]),
        fuel=float(totals[1]),
        frames=frames,
        final_x=float(obs[0]),
    )


def summarise(results: Sequence[EpisodeResult]) -> Dict[str, float]:
    """Outcome rates and mean totals over a set of episodes."""
    if not results:
        raise ValueError("cannot summarise zero episodes")
    count = len(results)
    summary: Dict[str, float] = {"episodes": count}
    for outcome in OUTCOMES:
        summary[f"{outcome}_rate"] = sum(r.outcome == outcome for r in results) / count
    summary["mean_safety"] = float(np.mean([r.safety for r in results]))
    summary["mean_fuel"] = float(np.mean([r.fuel for r in results]))
    summary["mean_frames"] = float(np.mean([r.frames for r in results]))
    return summary


def idle_policy(env: SubRepEnv) -> Policy:
    """Always do nothing (action 0), as `baseline/idle_policy.py` does."""
    return lambda obs: 0


def heuristic_policy(env: SubRepEnv) -> Policy:
    """Gymnasium's built-in landing controller."""
    return lambda obs: int(heuristic(env.env, obs))


def ppo_policy(env: SubRepEnv, checkpoint: str = DEFAULT_PPO_CHECKPOINT) -> Policy:
    """The committed PPO pilot, acting deterministically."""
    from pilot.rl_pilot import RLPilot

    pilot = RLPilot.load(checkpoint, map_location="cpu")
    return lambda obs: int(pilot.predict(obs, deterministic=True, return_probability=False))


CONTROLLERS: Dict[str, Callable[[SubRepEnv], Policy]] = {
    "idle": idle_policy,
    "heuristic": heuristic_policy,
    "ppo_pilot": ppo_policy,
}


def run_baseline(seeds: Sequence[int], controllers: Sequence[str]) -> Dict[str, dict]:
    """Run every controller on every seed and return per-controller results."""
    report: Dict[str, dict] = {}
    for name in controllers:
        env = SubRepEnv(seed=int(seeds[0]))
        policy = CONTROLLERS[name](env)
        episodes: List[EpisodeResult] = [run_episode(env, policy, int(seed)) for seed in seeds]
        env.close()
        report[name] = {
            "summary": summarise(episodes),
            "episodes": [asdict(e) for e in episodes],
        }
    return report


def to_markdown(report: Dict[str, dict], seeds: Sequence[int]) -> str:
    lines = [
        "# LunarLander baseline",
        "",
        f"Seeds {seeds[0]}-{seeds[-1]} ({len(seeds)} episodes per controller).",
        "",
        "| Controller | On pad | Off pad | Crashed | Timeout | Mean Safety | Mean Fuel | Mean frames |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, data in report.items():
        s = data["summary"]
        lines.append(
            f"| {name} | {s['landed_on_pad_rate']:.2f} | {s['landed_off_pad_rate']:.2f} | "
            f"{s['crashed_rate']:.2f} | {s['timeout_rate']:.2f} | {s['mean_safety']:.1f} | "
            f"{s['mean_fuel']:.1f} | {s['mean_frames']:.0f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure LunarLander baseline controllers.")
    parser.add_argument("--seeds", type=int, default=100, help="number of seeds")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--controllers", nargs="+", default=list(CONTROLLERS), choices=list(CONTROLLERS))
    parser.add_argument("--output", default="demo/artifacts/lunar_baseline.json")
    parser.add_argument("--markdown-output", default="demo/artifacts/lunar_baseline.md")
    args = parser.parse_args()

    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    report = run_baseline(seeds, args.controllers)
    markdown = to_markdown(report, seeds)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"seeds": seeds, "controllers": report}, indent=2), encoding="utf-8")
    Path(args.markdown_output).write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {output} and {args.markdown_output}")


if __name__ == "__main__":
    main()

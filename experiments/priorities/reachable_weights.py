"""The Safety weights MetaMo can reach and visits, against the switch points.

Three questions, on development seeds:

1. Reachable: over every Securing, Threshold and Approach in [0, 1] (the only
   modulators the LunarLander mapping reads, env/lunar_lander_metamo.py), what
   Safety weight can `lunar_lander_weights` produce? At the initial goals, and
   at the goals the live governor reached at the end of each run below.
2. Visited: the Safety weights the demo's governor actually emitted on seeds
   1-16, under the default (motive) risk and under `--risk task`. The demo is
   run as is (demo/run_lunar_lander_metamo.py); its governor is captured only to
   read the goals at the end.
3. Did MetaMo's movement change a pick? Per live step, against the weights
   MetaMo starts from (the fixed arm's (0.632, 0.368)):
     records  with admission held at eps 0.1 (switch_weights.py), does a switch
              point lie between the starting Safety weight and this step's?
     live     on the step's own admitted set (from the demo JSON), does the
              pick under this step's weights differ from the pick under the
              starting weights?

    python experiments/priorities/switch_weights.py
    python experiments/priorities/reachable_weights.py
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import demo.run_lunar_lander_metamo as lunar_demo  # noqa: E402
from env.lunar_lander_metamo import (  # noqa: E402
    LUNAR_LANDER_INITIAL_GOALS,
    lunar_lander_weights,
)
from experiments.priorities.switch_weights import DEFAULT_OUT as SWITCH_POINTS  # noqa: E402

OUT_DIR = REPO / "runs" / "priorities"
SEED = 1
STEPS = 16
SAFETY = 0
# Modulator indices in MetaMo's order: Valence, Arousal, Approach, Resolution,
# Threshold, Securing.
APPROACH, THRESHOLD, SECURING = 2, 4, 5
GRID = np.linspace(0.0, 1.0, 21)

# name -> (switch-point pool, demo arguments)
RUNS: Dict[str, Tuple[Optional[str], List[str]]] = {
    "ppo_motive": ("ppo", ["--candidates", "ppo"]),
    "ppo_task": ("ppo", ["--candidates", "ppo", "--risk", "task"]),
    "tradeoff_motive": ("tradeoff", ["--candidates", "tradeoff", "--epsilon-scale", "on"]),
    "tradeoff_task": ("tradeoff", ["--candidates", "tradeoff", "--epsilon-scale", "on", "--risk", "task"]),
    "ci_task": (None, ["--candidates", "ci", "--risk", "task"]),
}


def starting_weights() -> np.ndarray:
    """MetaMo's weights before any outcome: initial goals, every modulator at 0.5."""
    return lunar_lander_weights(LUNAR_LANDER_INITIAL_GOALS, np.full(6, 0.5))


def reachable_range(goals: np.ndarray) -> Tuple[float, float]:
    """Min and max Safety weight over the modulator grid, at these goals."""
    values = []
    for securing, threshold, approach in itertools.product(GRID, GRID, GRID):
        modulators = np.full(6, 0.5)
        modulators[[SECURING, THRESHOLD, APPROACH]] = securing, threshold, approach
        values.append(float(lunar_lander_weights(goals, modulators)[SAFETY]))
    return min(values), max(values)


def run_demo(name: str, args: List[str]) -> Tuple[Dict[str, Any], np.ndarray]:
    """Run the demo on seeds SEED..SEED+STEPS-1; return its JSON and the final goals."""
    captured: Dict[str, Any] = {}
    real = lunar_demo.make_lunar_lander_governor

    def capture(**kwargs: Any) -> Any:
        captured["governor"] = real(**kwargs)
        return captured["governor"]

    path = OUT_DIR / f"live_{name}.json"
    lunar_demo.make_lunar_lander_governor = capture
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = lunar_demo.main(
                ["--steps", str(STEPS), "--seed", str(SEED), *args, "--json-out", str(path)]
            )
    finally:
        lunar_demo.make_lunar_lander_governor = real
    if code != 0:
        raise RuntimeError(f"demo run {name} exited with {code}")
    return json.loads(path.read_text(encoding="utf-8")), np.asarray(captured["governor"].state.G)


def live_pick(candidates: Sequence[Dict[str, Any]], weights: Sequence[float]) -> Optional[str]:
    """The selector's choice among the step's admitted candidates."""
    admitted = [c for c in candidates if c["admitted"]]
    if not admitted:
        return None
    w = np.asarray(weights, dtype=np.float64)
    return min(
        admitted,
        key=lambda c: (-(c["delta_r"] + float(w @ np.asarray(c["delta_n"], dtype=np.float64))), c["skill_id"]),
    )["skill_id"]


def main() -> int:
    if not SWITCH_POINTS.is_file():
        print(f"error: run experiments/priorities/switch_weights.py first ({SWITCH_POINTS})",
              file=sys.stderr)
        return 2
    switches = json.loads(SWITCH_POINTS.read_text(encoding="utf-8"))
    start = starting_weights()
    start_safety = float(start[SAFETY])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    low, high = reachable_range(LUNAR_LANDER_INITIAL_GOALS)
    print(f"starting weights: Safety {start[0]:.3f}, Fuel {start[1]:.3f}")
    print(f"reachable Safety weight at the initial goals: {low:.3f}-{high:.3f}")

    report: Dict[str, Any] = {
        "starting_weights": start.tolist(),
        "reachable_initial_goals": [low, high],
        "runs": {},
    }
    print()
    print(f"{'run':<16} {'visited Safety':>15} {'reachable at end goals':>23} "
          f"{'crossed (records)':>18} {'pick differs (live)':>20} {'selected = live pick':>21}")
    for name, (pool, args) in RUNS.items():
        data, goals = run_demo(name, args)
        safety = [s["signal"]["weights"][SAFETY] for s in data["steps"]]
        end_low, end_high = reachable_range(goals)

        crossed: List[int] = []
        differs: List[int] = []
        consistent = 0
        for step in data["steps"]:
            seed, w_safety = step["seed"], step["signal"]["weights"][SAFETY]
            if pool is not None:
                points = [w["s"] for w in switches["pools"][pool][str(seed)]["switches"]]
                lo, hi = sorted((start_safety, w_safety))
                if any(lo < s < hi for s in points):
                    crossed.append(seed)
            now = live_pick(step["candidates"], step["signal"]["weights"])
            if now != live_pick(step["candidates"], start):
                differs.append(seed)
            consistent += now == step["selected"]

        report["runs"][name] = {
            "visited_safety": [min(safety), max(safety)],
            "reachable_at_end_goals": [end_low, end_high],
            "end_goals": goals.tolist(),
            "seeds_crossing_a_switch_at_eps_0.1": crossed,
            "seeds_where_the_live_pick_differs_from_the_start_weights": differs,
            "steps": len(data["steps"]),
        }
        crossed_text = "-" if pool is None else f"{len(crossed)}/{len(safety)}"
        print(f"{name:<16} {min(safety):>7.3f}-{max(safety):.3f} {end_low:>15.3f}-{end_high:.3f} "
              f"{crossed_text:>18} {len(differs):>14}/{len(safety)} {consistent:>17}/{len(safety)}")
        if crossed or differs:
            print(f"{'':<16} crossed at seeds {crossed}; live pick differs at seeds {differs}")

    path = OUT_DIR / "reachable.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nsaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

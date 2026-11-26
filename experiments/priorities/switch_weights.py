"""The Safety weight at which the LunarLander pick switches.

SubRep selects the admitted skill with the highest delta_r + w . delta_n
(library/skill_selector.py), with ties going to the lower skill_id. With two
objectives and w = (s, 1 - s), every admitted record is a straight line in s:

    score(s) = (delta_r + dFuel) + s * (dSafety - dFuel)

so the pick can only change where two lines cross. This script finds, per seed,
every Safety weight s* in (0, 1) at which the top-scoring admitted skill
changes, exactly: it takes every pairwise crossing in (0, 1) and keeps those
where the pick just below and just above differ. A 10,001-point grid of s is a
cross-check.

Admission is held fixed while s moves: the guarded fixed-unit PDS rule at
eps 0.1 (bridge/epsilon_scale.py with the LunarLander unit), which does not
depend on w. The pools are the demo's candidate sets:

    ppo        ppo_deterministic, ppo_stochastic, the three engines, random
    tradeoff   the extended set plus the fuel-wasting PPO candidates

Reads the records experiments/epsilon_scale/collect.py writes (seeds 1-40),
and writes runs/priorities/switch_points.json for reachable_weights.py.

    python experiments/epsilon_scale/collect.py
    python experiments/priorities/switch_weights.py [records.json]
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.epsilon_scale.priorities import admitted  # noqa: E402

DEFAULT_IN = REPO / "runs" / "epsilon_scale" / "records.json"
DEFAULT_OUT = REPO / "runs" / "priorities" / "switch_points.json"

# Development seeds only; anything else is refused.
ALLOWED_SEEDS = range(1, 41)
EPSILON = 0.1
GRID_POINTS = 10_001

# The demo's `ppo` candidates (data_collector/collect_candidate_sets.py,
# build_default_candidate_policies), without the noop baseline.
PPO_SET = frozenset(
    {"ppo_deterministic", "ppo_stochastic", "left_engine", "main_engine", "right_engine", "random"}
)
POOLS = {
    "ppo": lambda r: r["set"] == "task" and r["skill_id"] in PPO_SET,
    "tradeoff": lambda r: r["set"] in ("task", "tradeoff"),
}
BINS = np.round(np.arange(0.0, 1.01, 0.1), 1)

Switch = Tuple[float, str, str]


def score(record: Dict[str, Any], safety_weight: float) -> float:
    """delta_r + w . delta_n with w = (safety_weight, 1 - safety_weight)."""
    safety, fuel = record["delta_n"]
    return record["delta_r"] + safety_weight * safety + (1.0 - safety_weight) * fuel


def pick(records: Sequence[Dict[str, Any]], safety_weight: float) -> Optional[str]:
    """The selector's choice at this Safety weight: highest score, ties to the lower skill_id."""
    if not records:
        return None
    return min(records, key=lambda r: (-score(r, safety_weight), r["skill_id"]))["skill_id"]


def crossing(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[float]:
    """The Safety weight where the two score lines cross, or None if they are parallel."""
    (sa, fa), (sb, fb) = a["delta_n"], b["delta_n"]
    slope = (sa - fa) - (sb - fb)
    if slope == 0.0:
        return None
    return -((a["delta_r"] + fa) - (b["delta_r"] + fb)) / slope


def switch_points(records: Sequence[Dict[str, Any]]) -> List[Switch]:
    """Every s* in (0, 1) where the pick changes, with the picks below and above it."""
    breaks = sorted(
        {
            s
            for a, b in itertools.combinations(records, 2)
            for s in [crossing(a, b)]
            if s is not None and 0.0 < s < 1.0
        }
    )
    if not breaks:
        return []
    # The pick is constant between consecutive crossings, so one midpoint per
    # interval decides it.
    edges = [0.0] + breaks + [1.0]
    picks = [pick(records, (lo + hi) / 2.0) for lo, hi in zip(edges[:-1], edges[1:])]
    return [
        (float(s), below, above)
        for s, below, above in zip(breaks, picks[:-1], picks[1:])
        if below != above
    ]


def grid_switches(records: Sequence[Dict[str, Any]]) -> List[float]:
    """Where the pick changes on a fine grid of s: the cross-check."""
    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    picks = [pick(records, float(s)) for s in grid]
    return [float(grid[i]) for i in range(1, len(grid)) if picks[i] != picks[i - 1]]


def analyse(rows: Iterable[Dict[str, Any]], seeds: Sequence[int]) -> Dict[str, Any]:
    """Per pool and seed: the admitted set at EPSILON and its switch points."""
    by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        by_seed.setdefault(int(r["seed"]), []).append(r)
    result: Dict[str, Any] = {"epsilon": EPSILON, "seeds": list(seeds), "pools": {}}
    for name, member in POOLS.items():
        per_seed = {}
        for seed in seeds:
            pool = [r for r in by_seed.get(seed, []) if member(r) and admitted(r, EPSILON)]
            exact = switch_points(pool)
            grid = grid_switches(pool)
            if len(grid) != len(exact) or any(
                abs(g - s) > 1.0 / (GRID_POINTS - 1) for g, (s, _, _) in zip(grid, exact)
            ):
                raise RuntimeError(f"grid and exact switch points disagree: pool {name}, seed {seed}")
            per_seed[seed] = {
                "admitted": sorted(r["skill_id"] for r in pool),
                "switches": [{"s": s, "below": b, "above": a} for s, b, a in exact],
            }
        result["pools"][name] = per_seed
    return result


def summary_lines(result: Dict[str, Any]) -> List[str]:
    lines = [f"admission: guarded fixed-unit PDS at eps {result['epsilon']}; "
             f"seeds {result['seeds'][0]}-{result['seeds'][-1]}"]
    for name, per_seed in result["pools"].items():
        points = [w["s"] for seed in per_seed.values() for w in seed["switches"]]
        with_switch = sum(1 for seed in per_seed.values() if seed["switches"])
        lines.append("")
        lines.append(f"pool {name}: {with_switch}/{len(per_seed)} seeds have a switch in (0, 1); "
                     f"{len(points)} switch points")
        counts = Counter(min(int(s * 10), 9) for s in points)
        lines.append("  Safety weight   switches")
        for k in range(10):
            lines.append(f"  {BINS[k]:.1f}-{BINS[k + 1]:.1f}        {counts.get(k, 0):>3}")
        lines.append("  per seed (s*: pick below -> pick above):")
        for seed, data in per_seed.items():
            if data["switches"]:
                text = "; ".join(f"{w['s']:.3f}: {w['below']} -> {w['above']}" for w in data["switches"])
                lines.append(f"    seed {seed:>2}: {text}")
    return lines


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_IN
    rows = json.loads(path.read_text(encoding="utf-8"))
    seeds = sorted({int(r["seed"]) for r in rows})
    outside = [s for s in seeds if s not in ALLOWED_SEEDS]
    if outside:
        print(f"error: records hold seeds outside {ALLOWED_SEEDS.start}-{ALLOWED_SEEDS.stop - 1}: "
              f"{outside[:5]}", file=sys.stderr)
        return 2
    result = analyse(rows, seeds)
    print("\n".join(summary_lines(result)))
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nsaved: {DEFAULT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

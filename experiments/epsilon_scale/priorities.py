"""Does changing the Safety/Fuel weights change the selected skill? (held-out seeds 21-40)

The admitted set is the guarded fixed-unit rule (bridge/epsilon_scale.py) with
the LunarLander unit, at eps 0 and eps 0.1. The pick maximises
delta_r + w . delta_n. Reported for the task set alone and with the trade-off
and borderline candidates added, at extreme weights (Safety 0.8 vs 0.2) and at
the range the live governor produces (Safety 0.72 vs 0.60).

    python experiments/epsilon_scale/priorities.py [records.json]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from env.lunar_lander_metamo import (  # noqa: E402
    LUNAR_LANDER_EPSILON_UNIT,
    LUNAR_LANDER_PAYOFF_FLOOR,
)

DEFAULT_IN = REPO / "runs" / "epsilon_scale" / "records.json"


def admitted(r, eps: float) -> bool:
    if r["baseline_payoff"] + r["delta_r"] < LUNAR_LANDER_PAYOFF_FLOOR:
        return False
    return r["delta_r"] + min(r["delta_n"]) >= -eps * LUNAR_LANDER_EPSILON_UNIT


def pick(rows, safety_weight: float):
    if not rows:
        return None
    w = np.array([safety_weight, 1.0 - safety_weight])
    return sorted(
        rows, key=lambda r: (-(r["delta_r"] + float(w @ np.array(r["delta_n"]))), r["skill_id"])
    )[0]["skill_id"]


def main(path: Path) -> None:
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_seed = defaultdict(list)
    for r in rows:
        if r["seed"] > 20:
            by_seed[r["seed"]].append(r)

    for sets in (("task",), ("task", "tradeoff"), ("task", "tradeoff", "borderline")):
        for eps in (0.0, 0.1):
            for low, high in ((0.2, 0.8), (0.60, 0.72)):
                differ = []
                for seed, rs in sorted(by_seed.items()):
                    pool = [r for r in rs if r["set"] in sets and admitted(r, eps)]
                    p_low, p_high = pick(pool, low), pick(pool, high)
                    if p_low != p_high:
                        differ.append((seed, p_high, p_low))
                print(f"{'+'.join(sets):<26} eps {eps:<4} Safety {high} vs {low}: "
                      f"pick differs on {len(differ):>2}/{len(by_seed)} seeds {differ[:3]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN)

"""Compare epsilon-scale rules offline, on the records `collect.py` wrote.

Calibration seeds 1-20 measure the unit U; evaluation seeds 21-40 are held out.
For each rule it reports, on the held-out seeds: how many candidates change
admission between eps 0 and eps 0.5, how many crashes any eps admits, whether
the PPO pilot's landings are admitted at eps 0.1, and whether admission is
monotonic in eps.

Rules (the PDS margin is delta_r + min(delta_n)):

  raw               margin >= -eps                       (no scale)
  normalised        delta_r / s_r + min(delta_n) / s_n >= -eps
  unit              margin >= -eps * U
  unit_100          margin >= -eps * 100
  unit_raw_failures unit, but task failures keep the raw eps
  guarded_unit      task failures rejected; others margin >= -eps * U
                    (bridge/epsilon_scale.py)

    python experiments/epsilon_scale/analyse.py [records.json]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO / "runs" / "epsilon_scale" / "records.json"
GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
SCALE_FLOOR = 10.0
PAYOFF_FLOOR = 0.0
RULES = ["raw", "normalised", "unit", "unit_100", "unit_raw_failures", "guarded_unit"]


def load(path: Path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    for r in rows:
        r["dn"] = np.asarray(r["delta_n"], dtype=float)
        r["margin"] = r["delta_r"] + float(r["dn"].min())
        r["failing"] = r["baseline_payoff"] + r["delta_r"] < PAYOFF_FLOOR
        r["outcome"] = "landed" if r["landed"] else ("crashed" if r["crashed"] else "timeout")
    return rows


def scales(rows):
    """The `appraisal_scales` formula (env/lunar_lander_rollout.py), floor 10."""
    s_r = max(float(np.mean([abs(r["delta_r"]) for r in rows])), SCALE_FLOOR)
    s_n = max(float(np.mean([np.mean(np.abs(r["dn"])) for r in rows])), SCALE_FLOOR)
    return s_r, s_n


def make_admit(s_r: float, s_n: float):
    unit = s_r + s_n

    def admit(rule: str, r, eps: float) -> bool:
        m = r["margin"]
        if rule == "raw":
            return m >= -eps
        if rule == "normalised":
            return r["delta_r"] / s_r + float(r["dn"].min()) / s_n >= -eps
        if rule == "unit":
            return m >= -eps * unit
        if rule == "unit_100":
            return m >= -eps * 100.0
        if rule == "unit_raw_failures":
            return m >= (-eps if r["failing"] else -eps * unit)
        if rule == "guarded_unit":
            return (not r["failing"]) and m >= -eps * unit
        raise ValueError(rule)

    return admit


def report(admit, rule: str, data, label: str) -> str:
    flips = [r for r in data if admit(rule, r, 0.0) != admit(rule, r, 0.5)]
    crashes = [r for r in data if r["crashed"] and any(admit(rule, r, e) for e in GRID)]
    ppo = [r for r in data if r["skill_id"] == "ppo_deterministic" and r["landed"]]
    ppo_ok = sum(admit(rule, r, 0.1) for r in ppo)
    sweep = np.linspace(0.0, 0.5, 51)
    violations = sum(
        1 for r in data for a, b in zip(sweep, sweep[1:]) if admit(rule, r, a) and not admit(rule, r, b)
    )
    kinds = dict(Counter(r["outcome"] for r in flips))
    return (
        f"{rule:<18} {label:<10} flips 0->0.5: {len(flips):>3}/{len(data):<3} {kinds} | "
        f"crashes admitted at any eps: {len(crashes):>3} | "
        f"ppo_deterministic landings at 0.1: {ppo_ok}/{len(ppo)} | monotonic violations: {violations}"
    )


def main(path: Path) -> None:
    rows = load(path)
    calibration = [r for r in rows if r["seed"] <= 20 and r["set"] == "task"]
    held_out = [r for r in rows if r["seed"] > 20]

    s_r, s_n = scales(calibration)
    per_seed = np.array([sum(scales([r for r in calibration if r["seed"] == s])) for s in range(1, 21)])
    print(f"scales (seeds 1-20, task set): s_r = {s_r:.1f}, s_n = {s_n:.1f}, U = {s_r + s_n:.1f}; "
          f"per-seed CV of U = {per_seed.std() / per_seed.mean():.3f}")
    fuel_binds = sum(1 for r in rows if r["dn"][1] < r["dn"][0])
    print(f"Fuel is the objective that binds in {fuel_binds}/{len(rows)} pairs")

    admit = make_admit(s_r, s_n)
    print("held-out seeds 21-40:")
    for rule in RULES:
        for label in ("all", "task", "borderline", "tradeoff"):
            data = held_out if label == "all" else [r for r in held_out if r["set"] == label]
            print(report(admit, rule, data, label))
        print()


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN)

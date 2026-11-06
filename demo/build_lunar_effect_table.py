"""Build the measured effect table of the LunarLander skills and plans.

Flies every plan the planner proposes, at every state where the runtime would
decide, from a fixed range of seeds. Measures each plan and each of its steps
against doing nothing from the same state, and writes the summaries as versioned
JSON. The seeds are shared out between processes, and the table does not depend
on how many there are.

    python -m demo.build_lunar_effect_table
    python -m demo.build_lunar_effect_table --mission
    python -m demo.build_lunar_effect_table --seeds 50 --output demo/artifacts/lunar_effects.json
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Sequence, Tuple

import torch

from decomposition.domain import Domain
from decomposition.effects import (
    DEFAULT_TABLE_PATH,
    LUNAR_BEACON_TABLE,
    MAX_STATES_PER_SEED,
    EffectTable,
    SeedMeasurement,
    measure_seeds,
    table_from_measurements,
)
from decomposition.planner import plans_for
from decomposition.recipe_book import LUNAR_BEACON_BOOK, LUNAR_LANDER_BOOK, RecipeBook, load_recipe_book
from env.lunar_beacon import BEACON_FACT_NAMES, BEACON_MISSION, DEFAULT_BEACON, beacon_domain
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import FACT_NAMES
from env.lunar_pilots import mission_pilots

DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)
_CHUNK = 20  # seeds handed to a process at a time


def landing_book() -> RecipeBook:
    return load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, LUNAR_LANDER.skills)


def mission_book() -> RecipeBook:
    return load_recipe_book(LUNAR_BEACON_BOOK, BEACON_FACT_NAMES, BEACON_MISSION.skills)


def _setup(beacon: Optional[Tuple[float, float]]) -> Tuple[Domain, RecipeBook]:
    if beacon is None:
        return LUNAR_LANDER, landing_book()
    return beacon_domain(beacon, mission_pilots()), mission_book()


def _measure(job: Tuple[Sequence[int], Optional[Tuple[float, float]]]) -> List[SeedMeasurement]:
    seeds, beacon = job
    torch.set_num_threads(1)  # each process runs one small network at a time
    domain, book = _setup(beacon)
    return measure_seeds(domain, seeds, plans_for(book), MAX_STATES_PER_SEED)


def build(
    seeds: range, beacon: Optional[Tuple[float, float]] = None, workers: int = DEFAULT_WORKERS
) -> EffectTable:
    """Build the table for ``seeds``, in order whatever the number of processes.

    With a ``beacon``, the table is for the scan-then-land mission with the beacon there.
    """
    jobs = [(list(seeds[start : start + _CHUNK]), beacon) for start in range(0, len(seeds), _CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        measured = [measurement for chunk in pool.map(_measure, jobs) for measurement in chunk]
    domain, _ = _setup(beacon)
    return table_from_measurements(domain, seeds, measured)


def build_mission_table(seeds: range, beacon: Tuple[float, float] = DEFAULT_BEACON) -> EffectTable:
    return build(seeds, beacon)


def to_text(table: EffectTable) -> str:
    """A plain-text overview of the table: one row per plan or step and group."""
    header = (
        f"{'name':<62} {'group':<60} {'n':>4} {'done':>5} {'on pad':>6} "
        f"{'Δr':>8} {'Δr low':>8} {'ΔSafety':>8} {'ΔFuel':>8}"
    )
    lines = []
    for kind in ("plans", "steps"):
        lines += ["", kind, header]
        for entry in table.data[kind]:
            on_pad = f"{entry['outcomes']['landed_on_pad']:.2f}" if "outcomes" in entry else "-"
            lines.append(
                f"{entry['name']:<62} {entry['group']:<60} {entry['samples']:>4} "
                f"{entry['completion_rate']:>5.2f} {on_pad:>6} {entry['delta_r']['mean']:>8.2f} "
                f"{entry['delta_r']['lower']:>8.2f} {entry['delta_safety']['mean']:>8.2f} "
                f"{entry['delta_fuel']['mean']:>8.2f}"
            )
    return "\n".join(lines).lstrip("\n")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the reports hold symbols a Windows console encoding cannot write
    parser = argparse.ArgumentParser(description="Build the LunarLander effect table.")
    parser.add_argument("--seeds", type=int, default=2000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--mission", action="store_true", help="build the table for the scan-then-land mission")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    seeds = range(args.start_seed, args.start_seed + args.seeds)
    table = build(seeds, DEFAULT_BEACON if args.mission else None, args.workers)
    table.save(args.output or (LUNAR_BEACON_TABLE if args.mission else DEFAULT_TABLE_PATH))
    print(to_text(table))
    print("wrote the table")


if __name__ == "__main__":
    main()

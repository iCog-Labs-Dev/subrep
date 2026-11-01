"""Build the measured effect table of the LunarLander skills and plans.

Flies every plan the planner proposes, at every state where the runtime would
decide, from a fixed range of seeds. Measures each plan and each of its steps
against doing nothing from the same state, and writes the summaries as versioned
JSON. The seeds are shared out between processes, and the table does not depend
on how many there are.

    python -m demo.build_lunar_effect_table
    python -m demo.build_lunar_effect_table --seeds 50 --output demo/artifacts/lunar_effects.json
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Sequence

from decomposition.effects import (
    DEFAULT_TABLE_PATH,
    MAX_STATES_PER_SEED,
    EffectTable,
    SeedMeasurement,
    measure_seeds,
    table_from_measurements,
)
from decomposition.planner import plans_for
from decomposition.recipe_book import LUNAR_LANDER_BOOK, load_recipe_book
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import FACT_NAMES
from env.lunar_subskills import SUBSKILLS

DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) // 2)
_CHUNK = 20  # seeds handed to a process at a time


def _measure(seeds: Sequence[int]) -> List[SeedMeasurement]:
    book = load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, SUBSKILLS)
    return measure_seeds(LUNAR_LANDER, seeds, plans_for(book), MAX_STATES_PER_SEED)


def build(seeds: range, workers: int = DEFAULT_WORKERS) -> EffectTable:
    """Build the table for ``seeds``, in order whatever the number of processes."""
    chunks = [list(seeds[start : start + _CHUNK]) for start in range(0, len(seeds), _CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        measured = [measurement for chunk in pool.map(_measure, chunks) for measurement in chunk]
    return table_from_measurements(LUNAR_LANDER, seeds, measured)


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
    parser = argparse.ArgumentParser(description="Build the LunarLander effect table.")
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output", default=str(DEFAULT_TABLE_PATH))
    args = parser.parse_args()

    table = build(range(args.start_seed, args.start_seed + args.seeds), args.workers)
    table.save(args.output)
    print(to_text(table))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

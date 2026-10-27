"""Build the measured effect table of the LunarLander sub-skills and plans.

Flies every plan from a fixed range of seeds, measures each plan and each of
its steps against doing nothing from the same state, and writes the summaries
as versioned JSON.

    python -m demo.build_lunar_effect_table
    python -m demo.build_lunar_effect_table --seeds 50 --output demo/artifacts/lunar_effects.json
"""

from __future__ import annotations

import argparse
from typing import List

from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable, build_table
from demo.run_lunar_subskill_reliability import SEQUENCES

PLANS: List[List[str]] = [*SEQUENCES.values(), ["stabilise"]]


def to_text(table: EffectTable) -> str:
    """A plain-text overview of the table: one row per plan or step and group."""
    header = (
        f"{'name':<52} {'group':<35} {'n':>4} {'done':>5} {'on pad':>6} "
        f"{'Δr':>8} {'Δr low':>8} {'ΔSafety':>8} {'ΔFuel':>8}"
    )
    lines = []
    for kind in ("plans", "steps"):
        lines += ["", kind, header]
        for entry in table.data[kind]:
            on_pad = f"{entry['outcomes']['landed_on_pad']:.2f}" if "outcomes" in entry else "-"
            lines.append(
                f"{entry['name']:<52} {entry['group']:<35} {entry['samples']:>4} "
                f"{entry['completion_rate']:>5.2f} {on_pad:>6} {entry['delta_r']['mean']:>8.2f} "
                f"{entry['delta_r']['lower']:>8.2f} {entry['delta_safety']['mean']:>8.2f} "
                f"{entry['delta_fuel']['mean']:>8.2f}"
            )
    return "\n".join(lines).lstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the LunarLander effect table.")
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--output", default=str(DEFAULT_TABLE_PATH))
    args = parser.parse_args()

    table = build_table(range(args.start_seed, args.start_seed + args.seeds), PLANS)
    table.save(args.output)
    print(to_text(table))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

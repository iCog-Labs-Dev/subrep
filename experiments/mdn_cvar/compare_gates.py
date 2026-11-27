"""PDS and CVaR on LunarLander, apart and together, with the untrained and the trained MDN.

Four gate configurations, always after the task-failure guard
(bridge/epsilon_scale.py), so a crash is rejected by every one:

    PDS          delta_r + min(delta_n) >= -eps * U
    CVaR         lower-tail mean of delta_r + w . delta_n >= 0, w ~ Dirichlet(MDN alpha)
    PDS or CVaR  the pipeline's `use_cvar` rule, which the demo's `--cvar` runs
    PDS and CVaR computed from the two verdicts, for reporting only

1. Offline, on the stored records (experiments/epsilon_scale/collect.py, the
   demo's `tradeoff` candidates, seeds 1-40): admitted counts per configuration
   at fixed (eps, alpha) points, for the untrained and the trained MDN; which
   PDS rejections CVaR overturns under OR; and whether CVaR alone tells the
   fuel-wasting landings apart. The MDN reads each seed's start observation.
2. The alpha direction: one landing's CVaR as alpha grows, on the same samples.
3. Live, through the demo: seeds 1-16, `tradeoff` with the epsilon scale and
   the trained MDN, each `--gates` value under both risks. Per run, the
   selections, the alpha range, and how often MetaMo's weights lie inside the
   MDN's support region; and, for the CVaR-gated runs, what would change on the
   outside steps if PDS decided them instead, holding the rest of the run as it
   was.

    python experiments/epsilon_scale/collect.py
    python experiments/mdn_cvar/compare_gates.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import demo.run_lunar_lander_metamo as lunar_demo  # noqa: E402
from env.lunar_lander_metamo import LUNAR_LANDER_PAYOFF_FLOOR  # noqa: E402
from env.lunar_lander_wrapper import SubRepEnv  # noqa: E402
from experiments.epsilon_scale.priorities import admitted as pds_admits  # noqa: E402

RECORDS = REPO / "runs" / "epsilon_scale" / "records.json"
OUT_DIR = REPO / "runs" / "mdn_cvar"
TRAINED = REPO / "models" / "lunar_mdn.pth"

# Development seeds only; anything else is refused.
ALLOWED_SEEDS = range(1, 41)
POOL = ("task", "tradeoff")  # the demo's `tradeoff` candidates
CVAR_SAMPLES = 1000
UNTRAINED_SEED = 1  # the demo's default seed, so the same untrained network
# (eps, alpha): the defaults, a cautious MetaMo, a middle point, and where the
# task risk takes a relaxed MetaMo.
GRID = ((0.1, 0.1), (0.0, 0.05), (0.3, 0.3), (0.37, 0.43))
ALPHAS = (0.05, 0.1, 0.2, 0.43, 0.5)
LIVE_SEED, LIVE_STEPS = 1, 16
GATES = ("pds", "cvar", "pds-or-cvar")
RISKS = ("motive", "task")


def kind(row: Dict[str, Any]) -> str:
    """landing / timeout / crash, with the fuel-wasting landings apart."""
    if row["landed"]:
        return "trade-off landing" if row["set"] == "tradeoff" else "landing"
    return "crash" if row["crashed"] else "timeout"


def models() -> Dict[str, Any]:
    """The demo's untrained network (seeded as the demo seeds it) and the trained one."""
    return {
        "untrained": lunar_demo.build_pipeline(
            seed=UNTRAINED_SEED, gates="pds", cvar_samples=CVAR_SAMPLES
        ).model,
        "trained": lunar_demo.load_mdn(TRAINED),
    }


def offline(rows: List[Dict[str, Any]], starts: Dict[int, np.ndarray]) -> Dict[str, Any]:
    """Admitted counts per configuration, per MDN and (eps, alpha) point."""
    result: Dict[str, Any] = {}
    for name, model in models().items():
        views = {seed: lunar_demo.mdn_view(model, obs) for seed, obs in starts.items()}
        for epsilon, alpha in GRID:
            counts: Dict[str, Counter] = {g: Counter() for g in ("pds", "cvar", "or", "and")}
            overturned: Counter = Counter()
            trade_off_cvar: List[float] = []
            for row in rows:
                failing = row["baseline_payoff"] + row["delta_r"] < LUNAR_LANDER_PAYOFF_FLOOR
                (value,) = lunar_demo.cvar_values(
                    [row], views[row["seed"]]["concentration"], alpha, CVAR_SAMPLES, row["seed"]
                )
                pds = pds_admits(row, epsilon)
                cvar = (not failing) and value >= 0.0
                for gate, verdict in (("pds", pds), ("cvar", cvar), ("or", pds or cvar), ("and", pds and cvar)):
                    if verdict:
                        counts[gate][kind(row)] += 1
                if cvar and not pds:
                    overturned[kind(row)] += 1
                if kind(row) == "trade-off landing":
                    trade_off_cvar.append(value)
            result[f"{name} eps={epsilon} alpha={alpha}"] = {
                "admitted": {g: dict(c) for g, c in counts.items()},
                "pds_rejections_overturned_by_cvar": dict(overturned),
                "trade_off_landing_cvar_min": float(min(trade_off_cvar)),
            }
    return result


def alpha_direction(rows: List[Dict[str, Any]], starts: Dict[int, np.ndarray]) -> Dict[str, Any]:
    """CVaR of two landings as alpha grows, on the same samples: it must not fall."""
    first_seed = min(starts)
    landing = next(r for r in rows if r["seed"] == first_seed and r["skill_id"] == "ppo_deterministic")
    trade_offs = [r for r in rows if kind(r) == "trade-off landing"]
    nets = models()
    trained_view = {r["seed"]: lunar_demo.mdn_view(nets["trained"], starts[r["seed"]]) for r in trade_offs}
    # The fuel-wasting landing whose CVaR is lowest at alpha 0.1 with the trained MDN.
    hardest = min(
        trade_offs,
        key=lambda r: lunar_demo.cvar_values([r], trained_view[r["seed"]]["concentration"], 0.1,
                                             CVAR_SAMPLES, r["seed"])[0],
    )
    result = {}
    for label, row in (("ppo_deterministic landing", landing), ("hardest trade-off landing", hardest)):
        for name, model in nets.items():
            view = lunar_demo.mdn_view(model, starts[row["seed"]])
            values = [
                lunar_demo.cvar_values([row], view["concentration"], a, CVAR_SAMPLES, row["seed"])[0]
                for a in ALPHAS
            ]
            result[f"{label} (seed {row['seed']}, {row['skill_id']}), {name}"] = {
                "cvar": dict(zip(map(str, ALPHAS), values)),
                "non_decreasing": all(b >= a for a, b in zip(values, values[1:])),
            }
    return result


def pick(candidates: Sequence[Dict[str, Any]], key: str, weights: Sequence[float]) -> Optional[str]:
    """The selector's choice among the candidates whose `key` verdict is true."""
    chosen = [c for c in candidates if c[key]]
    if not chosen:
        return None
    w = np.asarray(weights, dtype=np.float64)
    return min(
        chosen,
        key=lambda c: (-(c["delta_r"] + float(w @ np.asarray(c["delta_n"], dtype=np.float64))), c["skill_id"]),
    )["skill_id"]


def live() -> Dict[str, Any]:
    """The demo with the trained MDN, every --gates value, both risks."""
    result = {}
    for risk in RISKS:
        for gates in GATES:
            name = f"{risk}_{gates}"
            path = OUT_DIR / f"live_{name}.json"
            argv = ["--steps", str(LIVE_STEPS), "--seed", str(LIVE_SEED), "--candidates", "tradeoff",
                    "--epsilon-scale", "on", "--mdn", str(TRAINED), "--gates", gates,
                    "--json-out", str(path)]
            if risk == "task":
                argv += ["--risk", "task"]
            with contextlib.redirect_stdout(io.StringIO()):
                if lunar_demo.main(argv) != 0:
                    raise RuntimeError(f"demo run {name} failed")
            data = json.loads(path.read_text(encoding="utf-8"))
            steps = data["steps"]
            alpha = [s["signal"]["alpha"] for s in steps]
            safety = [s["signal"]["weights"][0] for s in steps]
            outside = [s for s in steps if not s["w_t_inside_mdn_region"]]
            entry = {
                "selections": data["summary"]["selections"],
                "results": data["summary"]["results"],
                "alpha_range": [min(alpha), max(alpha)],
                "safety_weight_range": [min(safety), max(safety)],
                "inside_mdn_region_steps": len(steps) - len(outside),
                "admitted_by_the_decision": data["summary"]["admitted_total"],
                "gate_totals": data["summary"]["gate_admitted_totals"],
                "verdict_disagreements": data["summary"]["verdict_disagreements"],
            }
            if gates != "pds":
                # If PDS decided the outside steps instead: does the admitted
                # set or the pick change on that step?
                set_changes = sum(
                    1 for s in outside
                    if [c["admitted"] for c in s["candidates"]] != [c["pds_admitted"] for c in s["candidates"]]
                )
                pick_changes = sum(
                    1 for s in outside
                    if pick(s["candidates"], "pds_admitted", s["signal"]["weights"]) != s["selected"]
                )
                entry["if_pds_decided_the_outside_steps"] = {
                    "steps": len(outside),
                    "admitted_set_changes": set_changes,
                    "selection_changes": pick_changes,
                }
            result[name] = entry
    return result


def main() -> int:
    if not RECORDS.is_file():
        print(f"error: run experiments/epsilon_scale/collect.py first ({RECORDS})", file=sys.stderr)
        return 2
    rows = [r for r in json.loads(RECORDS.read_text(encoding="utf-8")) if r["set"] in POOL]
    seeds = sorted({int(r["seed"]) for r in rows})
    if any(s not in ALLOWED_SEEDS for s in seeds):
        print(f"error: records hold seeds outside {ALLOWED_SEEDS.start}-{ALLOWED_SEEDS.stop - 1}",
              file=sys.stderr)
        return 2
    env = SubRepEnv(seed=seeds[0])
    try:
        starts = {seed: np.asarray(env.reset(seed=seed)[0], dtype=np.float32) for seed in seeds}
    finally:
        env.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "offline": offline(rows, starts),
        "alpha_direction": alpha_direction(rows, starts),
        "live": live(),
    }
    print(json.dumps(report, indent=1))
    path = OUT_DIR / "compare_gates.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

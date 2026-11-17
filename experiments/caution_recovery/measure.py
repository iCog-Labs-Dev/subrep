"""Measure what MetaMo is told after a LunarLander episode, and where that drives caution.

Three measurements under the bridge's default risk (the worst objective change
against the noop baseline), all from one set of rollouts on the development
seeds:

1. Stimulus by outcome type. The risk, novelty, conduciveness and effort the
   bridge builds for a landing, a crash, a timeout and an abstention (flying
   the noop baseline, or flying nothing), with the spread across seeds.
2. Fixed points. A fresh governor is fed 30 identical outcomes of one type;
   Securing, Threshold, eps and the Safety weight are recorded per step.
3. Sequences. 8 crashes then 16 landings, and the reverse.

Everything is computed under two Safety conventions:

    kept    Safety = landing reward + shaping   (the wrapper; the one in use)
    once    Safety = shaping only               (the landing counts once, in the payoff)

Each episode is flown once and its summed landing reward, shaping and fuel are
stored separately, so both conventions come from the same runs. Nothing in
bridge/ or env/ is changed: the sums are taken by a pass-through around the
environment, and the existing `run_episode` does the flying.

The candidates are the task, borderline and trade-off sets of
experiments/epsilon_scale/collect.py. The outcome fed to the governor for a
type is the median outcome of the task set, and the appraisal scales come from
the task set on the first seed, using the demo's formula.

The first section of the report repeats, as a check, the seeds 1-20 comparison
of the two Safety conventions, made with the same method when the landing
reward's double count was settled. Its landing risk, novelty and
Securing-after-30-landings values are published in
docs/subrep_metamo_integration.md, "Payoff and Safety"; the crash and timeout
values were not published.

Flight results depend on the platform: run this on Linux to reproduce the
committed numbers.

Run:
    python experiments/caution_recovery/measure.py
    python experiments/caution_recovery/measure.py --refresh   # fly again
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bridge._loader import find_metamo_root  # noqa: E402
from bridge.protocol import SkillOutcome  # noqa: E402
from bridge.stimulus import build_stimulus_values  # noqa: E402
from env.lunar_lander_metamo import make_lunar_lander_governor  # noqa: E402
from env.lunar_lander_rollout import (  # noqa: E402
    NOOP_BASELINE,
    appraisal_scales,
    episode_outcome,
    run_episode,
)
from env.lunar_lander_wrapper import SubRepEnv  # noqa: E402

HERE = Path(__file__).resolve().parent
PILOT_CHECKPOINT = REPO_ROOT / "models" / "pilot_ppo.pt"
CANDIDATE_SETS_SCRIPT = REPO_ROOT / "experiments" / "epsilon_scale" / "collect.py"

# Development seeds. Anything outside this range is refused, so a mistyped
# argument cannot touch seeds held back for evaluation.
ALLOWED_SEEDS = range(1, 41)
# The seeds of the seeds 1-20 comparison the report checks itself against.
REFERENCE_SEEDS = range(1, 21)

TASK_SET = "task"
BASELINE_SET = "baseline"

SAFETY = 0
FIXED_POINT_STEPS = 30
SEQUENCE_SHORT = 8
SEQUENCE_LONG = 16
# A policy needs this many episodes of a type to get its own row of spread.
MIN_EPISODES_PER_POLICY = 5

LANDING = "landing"
CRASH = "crash"
TIMEOUT = "timeout"
ABSTAIN_NOOP = "abstention (noop)"
ABSTAIN_STOP = "abstention (stop)"
FLOWN_TYPES = (LANDING, CRASH, TIMEOUT)
ALL_TYPES = FLOWN_TYPES + (ABSTAIN_NOOP, ABSTAIN_STOP)

KEPT = "kept"
ONCE = "once"
# Safety of one stored episode, per convention.
CONVENTIONS: Dict[str, Callable[[Dict[str, Any]], float]] = {
    KEPT: lambda e: e["terminal"] + e["shaping"],
    ONCE: lambda e: e["shaping"],
}
CONVENTION_TITLES = {
    KEPT: "kept: Safety = landing reward + shaping (the convention in use)",
    ONCE: "once: Safety = shaping only (the alternative convention)",
}

STIMULUS_FIELDS = ("risk", "novelty", "conduciveness", "effort")
TRACE_FIELDS = ("securing", "threshold", "epsilon", "safety_weight")
PERCENTILES = (0, 10, 25, 50, 75, 90, 100)

# The seeds 1-20 comparison: medians, with the same MetaMo commit.
# Stimulus: (risk, novelty, conduciveness). Runs: (Securing, eps) at the end;
# None where the comparison did not record the value.
REFERENCE_STIMULUS = {
    KEPT: {
        LANDING: (0.393, 0.760, 0.991),
        CRASH: (0.885, 0.769, 0.268),
        TIMEOUT: (0.995, 0.980, 0.834),
        ABSTAIN_STOP: (0.250, 0.000, 0.500),
    },
    ONCE: {
        LANDING: (0.435, 0.461, 0.985),
        CRASH: (0.917, 0.815, 0.244),
        TIMEOUT: (0.998, 0.986, 0.779),
        ABSTAIN_STOP: (0.250, 0.000, 0.500),
    },
}
REFERENCE_RUNS = {
    KEPT: {
        "30 landings": (0.896, 0.033),
        "30 crashes": (0.998, None),
        "8 crashes then 16 landings": (0.907, 0.029),
    },
    ONCE: {
        "30 landings": (0.943, 0.005),
        "30 crashes": (0.999, None),
        "8 crashes then 16 landings": (0.952, 0.001),
    },
}


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


class SummingEnv:
    """Passes every call through and sums the per-step reward parts of an episode."""

    def __init__(self, inner: SubRepEnv) -> None:
        self._inner = inner
        self.env = inner.env
        self.sums = {"terminal": 0.0, "shaping": 0.0, "fuel": 0.0}

    def reset(self, seed: Optional[int] = None) -> Any:
        self.sums = {"terminal": 0.0, "shaping": 0.0, "fuel": 0.0}
        return self._inner.reset(seed=seed)

    def step(self, action: int) -> Any:
        out = self._inner.step(action)
        info = out[4]
        self.sums["terminal"] += float(info["terminal_reward"])
        self.sums["shaping"] += float(info["shaping_reward"])
        self.sums["fuel"] += float(info["fuel_usage"])
        return out


def episode_result(info: Dict[str, Any]) -> str:
    """How an episode ended, by the rule the demo uses."""
    if info.get("landing_success"):
        return LANDING
    if info.get("crashed"):
        return CRASH
    if info.get("stop_reason") in ("truncated", "max_steps"):
        return TIMEOUT
    return "other"


def candidate_sets(env: SubRepEnv) -> Dict[str, Tuple[Any, ...]]:
    """The task, borderline and trade-off candidates of the epsilon-scale study."""
    spec = importlib.util.spec_from_file_location(
        "epsilon_scale_collect", CANDIDATE_SETS_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_sets(env)


def fly_all(seeds: Sequence[int]) -> List[Dict[str, Any]]:
    """Fly the noop baseline and every candidate once per seed."""
    inner = SubRepEnv(seed=seeds[0])
    env = SummingEnv(inner)
    episodes: List[Dict[str, Any]] = []
    try:
        policies = [(BASELINE_SET, NOOP_BASELINE)]
        for set_name, members in candidate_sets(inner).items():
            policies += [
                (set_name, c) for c in members if c.skill_id != NOOP_BASELINE.skill_id
            ]
        for seed in seeds:
            for set_name, policy in policies:
                r_hat, n_hat, info = run_episode(env, policy, seed=seed)
                sums = dict(env.sums)
                # The stored parts must rebuild what the rollout adapter reports.
                if abs(sums["terminal"] - r_hat) > 1e-6:
                    raise RuntimeError(f"landing reward mismatch: {policy.skill_id} seed {seed}")
                if abs(sums["terminal"] + sums["shaping"] - float(n_hat[0])) > 1e-2:
                    raise RuntimeError(f"Safety mismatch: {policy.skill_id} seed {seed}")
                if abs(sums["fuel"] - float(n_hat[1])) > 1e-2:
                    raise RuntimeError(f"Fuel mismatch: {policy.skill_id} seed {seed}")
                episodes.append(
                    {
                        "seed": int(seed),
                        "policy": policy.skill_id,
                        "set": set_name,
                        "result": episode_result(info),
                        "steps": int(info["steps"]),
                        **sums,
                    }
                )
            print(f"flown seed {seed}", file=sys.stderr)
    finally:
        inner.close()
    return episodes


def load_or_fly(path: Path, seeds: Sequence[int], refresh: bool) -> List[Dict[str, Any]]:
    """The stored rollouts for exactly these seeds, flying them if needed."""
    if path.is_file() and not refresh:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("seeds") == list(seeds):
            return stored["episodes"]
    episodes = fly_all(seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"seeds": list(seeds), "episodes": episodes}, indent=1),
        encoding="utf-8",
    )
    return episodes


# ---------------------------------------------------------------------------
# Outcomes under one convention
# ---------------------------------------------------------------------------


def motives(episode: Dict[str, Any], convention: str) -> np.ndarray:
    """The [Safety, Fuel] sums of a stored episode under `convention`."""
    return np.array([CONVENTIONS[convention](episode), episode["fuel"]], dtype=np.float64)


def flown_outcomes(
    episodes: Sequence[Dict[str, Any]], convention: str
) -> List[Dict[str, Any]]:
    """Every non-baseline episode as improvements over noop from the same seed."""
    baseline = {e["seed"]: e for e in episodes if e["set"] == BASELINE_SET}
    rows = []
    for e in episodes:
        if e["set"] == BASELINE_SET:
            continue
        base = baseline[e["seed"]]
        delta_r = e["terminal"] - base["terminal"]
        delta_n = motives(e, convention) - motives(base, convention)
        outcome = episode_outcome(
            delta_r, delta_n, {"steps": e["steps"], "max_steps": None}, admitted=True
        )
        rows.append(
            {
                "seed": e["seed"],
                "policy": e["policy"],
                "set": e["set"],
                "type": e["result"],
                "outcome": outcome,
            }
        )
    return rows


def abstention_outcomes(episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the demo feeds back when nothing is admitted, in both modes.

    `noop` flies the baseline, so the improvement over the baseline is exactly
    zero and only the flight time differs by seed. `stop` flies nothing.
    """
    rows = []
    for e in episodes:
        if e["set"] != BASELINE_SET:
            continue
        outcome = episode_outcome(
            0.0, np.zeros(2), {"steps": e["steps"], "max_steps": None}, admitted=False
        )
        rows.append(
            {
                "seed": e["seed"],
                "policy": NOOP_BASELINE.skill_id,
                "set": BASELINE_SET,
                "type": ABSTAIN_NOOP,
                "outcome": outcome,
            }
        )
    rows.append(
        {
            "seed": None,
            "policy": "-",
            "set": BASELINE_SET,
            "type": ABSTAIN_STOP,
            "outcome": SkillOutcome(
                delta_r=0.0, delta_n=np.zeros(2), admitted=False, effort=0.0
            ),
        }
    )
    return rows


def scales_from(rows: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    """(payoff_scale, motive_scale) by the demo's formula over `rows`."""
    return appraisal_scales(
        [
            SimpleNamespace(delta_r=r["outcome"].delta_r, delta_n=r["outcome"].delta_n)
            for r in rows
        ]
    )


def median_outcome(rows: Sequence[Dict[str, Any]]) -> Optional[SkillOutcome]:
    """The componentwise median outcome of `rows`, or None if there are none."""
    outcomes = [r["outcome"] for r in rows]
    if not outcomes:
        return None
    return SkillOutcome(
        delta_r=float(np.median([o.delta_r for o in outcomes])),
        delta_n=np.median(np.stack([o.delta_n for o in outcomes]), axis=0),
        admitted=outcomes[0].admitted,
        effort=float(np.median([o.effort for o in outcomes])),
    )


def of_type(rows: Sequence[Dict[str, Any]], kind: str, pool: Optional[str] = TASK_SET):
    """Rows of one outcome type; flown types are limited to `pool` unless it is None."""
    if kind not in FLOWN_TYPES or pool is None:
        return [r for r in rows if r["type"] == kind]
    return [r for r in rows if r["type"] == kind and r["set"] == pool]


# ---------------------------------------------------------------------------
# Governor traces
# ---------------------------------------------------------------------------


def snapshot(governor: Any) -> Dict[str, float]:
    mods = governor.modulator_summary()
    signal = governor.signal()
    return {
        "securing": mods["securing"],
        "threshold": mods["threshold"],
        "approach": mods["approach"],
        "epsilon": float(signal.pds_epsilon),
        "safety_weight": float(signal.weights[SAFETY]),
    }


def trace(outcomes: Sequence[SkillOutcome], scales: Tuple[float, float]) -> List[Dict[str, float]]:
    """State of a fresh governor before any outcome (step 0) and after each one."""
    governor = make_lunar_lander_governor(payoff_scale=scales[0], motive_scale=scales[1])
    states = [snapshot(governor)]
    for outcome in outcomes:
        governor.step(outcome)
        states.append(snapshot(governor))
    return states


def measure(
    episodes: Sequence[Dict[str, Any]],
    convention: str,
    seeds: Sequence[int],
    scale_seed: int,
    *,
    spread: bool,
) -> Dict[str, Any]:
    """All three measurements under one convention, on `seeds`."""
    flown = flown_outcomes(episodes, convention)
    task_rows = [r for r in flown if r["set"] == TASK_SET]
    scales = scales_from([r for r in task_rows if r["seed"] == scale_seed])
    other_scales = {
        f"task set, pooled over seeds {REFERENCE_SEEDS[0]}–{REFERENCE_SEEDS[-1]}": scales_from(
            [r for r in task_rows if r["seed"] in REFERENCE_SEEDS]
        ),
        "task set, pooled over all seeds flown": scales_from(task_rows),
    }

    wanted = set(seeds)
    rows = [r for r in flown if r["seed"] in wanted]
    rows += [
        r for r in abstention_outcomes(episodes) if r["seed"] is None or r["seed"] in wanted
    ]

    initial_weights = make_lunar_lander_governor(
        payoff_scale=scales[0], motive_scale=scales[1]
    ).signal().weights
    for row in rows:
        row["stimulus"] = build_stimulus_values(
            row["outcome"],
            weights=initial_weights,
            payoff_scale=scales[0],
            motive_scale=scales[1],
        )

    reps = {kind: median_outcome(of_type(rows, kind)) for kind in ALL_TYPES}
    rep_stimulus = {
        kind: build_stimulus_values(
            rep, weights=initial_weights, payoff_scale=scales[0], motive_scale=scales[1]
        )
        for kind, rep in reps.items()
        if rep is not None
    }
    fixed = {
        kind: trace([rep] * FIXED_POINT_STEPS, scales)
        for kind, rep in reps.items()
        if rep is not None
    }
    sequences = {}
    if reps[CRASH] is not None and reps[LANDING] is not None:
        sequences = {
            "crashes_then_landings": trace(
                [reps[CRASH]] * SEQUENCE_SHORT + [reps[LANDING]] * SEQUENCE_LONG, scales
            ),
            "landings_then_crashes": trace(
                [reps[LANDING]] * SEQUENCE_LONG + [reps[CRASH]] * SEQUENCE_SHORT, scales
            ),
        }

    result = {
        "rows": rows,
        "scales": scales,
        "initial_weights": initial_weights,
        "representatives": reps,
        "representative_stimulus": rep_stimulus,
        "fixed": fixed,
        "sequences": sequences,
    }
    if not spread:
        return result

    # Where each single episode would take the governor if it repeated: the
    # spread across seeds that the median outcome hides.
    per_episode: Dict[Tuple[str, str, str], List[Dict[str, float]]] = {}
    for row in rows:
        if row["type"] not in FLOWN_TYPES and row["type"] != ABSTAIN_NOOP:
            continue
        key = (row["type"], row["set"], row["policy"])
        end = trace([row["outcome"]] * FIXED_POINT_STEPS, scales)[-1]
        per_episode.setdefault(key, []).append(end)
    result["per_episode_fixed"] = per_episode

    # The landing and crash fixed points under other ways of setting the scales.
    sensitivity = {}
    for label, alt in other_scales.items():
        sensitivity[label] = {
            "scales": alt,
            **{
                kind: trace([reps[kind]] * FIXED_POINT_STEPS, alt)[-1]
                for kind in (LANDING, CRASH)
                if reps[kind] is not None
            },
        }
    result["scale_sensitivity"] = sensitivity
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def table(header: Sequence[str], body: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return lines + [""]


def spread(values: Sequence[float], digits: int = 3) -> str:
    """Median, with the range when the values differ."""
    lo, hi, mid = float(np.min(values)), float(np.max(values)), float(np.median(values))
    if hi - lo < 0.5 * 10 ** -digits:
        return f"{mid:.{digits}f}"
    return f"{mid:.{digits}f} ({lo:.{digits}f}–{hi:.{digits}f})"


def number(field: str, value: float) -> str:
    return f"{value:.4f}" if field == "epsilon" else f"{value:.3f}"


STIMULUS_HEADER = (
    "n", "risk", "novelty", "conduciveness", "effort", "Δr", "ΔSafety", "ΔFuel",
)


def stimulus_cells(rows: Sequence[Dict[str, Any]], label: str) -> List[str]:
    cells = [label, str(len(rows))]
    cells += [spread([getattr(r["stimulus"], f) for r in rows]) for f in STIMULUS_FIELDS]
    cells.append(f"{np.median([r['outcome'].delta_r for r in rows]):+.0f}")
    cells.append(f"{np.median([r['outcome'].delta_n[0] for r in rows]):+.1f}")
    cells.append(f"{np.median([r['outcome'].delta_n[1] for r in rows]):+.1f}")
    return cells


def stimulus_section(result: Dict[str, Any]) -> List[str]:
    rows = result["rows"]
    sets = sorted({r["set"] for r in rows if r["type"] in FLOWN_TYPES})

    lines = ["Task set, and the two abstentions:", ""]
    body = []
    for kind in ALL_TYPES:
        picked = of_type(rows, kind)
        if picked:
            body.append(stimulus_cells(picked, kind))
        else:
            body.append([kind, "0", "not observed"] + [""] * (len(STIMULUS_HEADER) - 2))
    lines += table(("outcome",) + STIMULUS_HEADER, body)

    lines += ["Spread across episodes (task set; percentiles):", ""]
    body = []
    for kind in FLOWN_TYPES + (ABSTAIN_NOOP,):
        picked = of_type(rows, kind)
        for field in STIMULUS_FIELDS:
            values = [getattr(r["stimulus"], field) for r in picked]
            if not values or max(values) - min(values) < 5e-4:
                continue
            body.append(
                [kind, field] + [f"{np.percentile(values, p):.3f}" for p in PERCENTILES]
            )
    lines += table(
        ("outcome", "value", "min", "10%", "25%", "median", "75%", "90%", "max"), body
    )

    lines += ["By candidate set:", ""]
    body = []
    for kind in FLOWN_TYPES:
        for pool in sets + [None]:
            picked = of_type(rows, kind, pool)
            if picked:
                body.append(stimulus_cells(picked, f"{kind}: {pool or 'all sets'}"))
    lines += table(("outcome: set",) + STIMULUS_HEADER, body)

    lines += ["<details><summary>By policy</summary>", ""]
    body = []
    for kind in FLOWN_TYPES:
        for pool in sets:
            for policy in sorted({r["policy"] for r in of_type(rows, kind, pool)}):
                picked = [r for r in of_type(rows, kind, pool) if r["policy"] == policy]
                body.append(stimulus_cells(picked, f"{kind}: {pool}: {policy}"))
    lines += table(("outcome: set: policy",) + STIMULUS_HEADER, body)
    lines += ["</details>", ""]
    return lines


def risk_comparison(result: Dict[str, Any], pool: Optional[str]) -> Optional[Dict[str, Any]]:
    """How well risk tells a crash from a landing, or None if either is missing."""
    land = [r["stimulus"].risk for r in of_type(result["rows"], LANDING, pool)]
    crash = [r["stimulus"].risk for r in of_type(result["rows"], CRASH, pool)]
    if not land or not crash:
        return None
    return {
        "land": land,
        "crash": crash,
        # Chance that a crash drawn at random reads as riskier than a landing
        # drawn at random: 1.0 is perfect separation, 0.5 is none.
        "separation": float(np.mean([c > l for c in crash for l in land])),
        "crashes_below_landing_median": float(np.mean(np.array(crash) <= np.median(land))),
    }


def risk_verdict(result: Dict[str, Any]) -> List[str]:
    lines = []
    for pool in (TASK_SET, None):
        cmp = risk_comparison(result, pool)
        if cmp is None:
            continue
        land, crash = cmp["land"], cmp["crash"]
        lines.append(
            f"- {pool or 'all'} set{'' if pool else 's'}: landing risk median "
            f"{np.median(land):.3f} ({min(land):.3f}–{max(land):.3f}), crash risk median "
            f"{np.median(crash):.3f} ({min(crash):.3f}–{max(crash):.3f}). A random crash "
            f"has a higher risk than a random landing {cmp['separation']:.0%} of the time; "
            f"{cmp['crashes_below_landing_median']:.0%} of crashes have a risk at or below "
            "the median landing's."
        )
    return ["Crash against landing:", ""] + lines + [""]


def side_by_side(traces: Dict[str, List[Dict[str, float]]], fed: Optional[Sequence[str]]) -> List[str]:
    """One row per step, the two conventions next to each other."""
    names = list(traces)
    header = ["step"] + (["fed"] if fed else [])
    for name in names:
        header += [f"{name} {f}" for f in TRACE_FIELDS]
    body = []
    for step in range(len(traces[names[0]])):
        cells = [str(step)] + ([fed[step]] if fed else [])
        for name in names:
            cells += [number(f, traces[name][step][f]) for f in TRACE_FIELDS]
        body.append(cells)
    return table(header, body)


def metamo_commit() -> str:
    root = find_metamo_root()
    if root is None or not (root / ".git").exists():
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def reference_section(check: Dict[str, Dict[str, Any]]) -> List[str]:
    """This script's values on the reference seeds, beside the seeds 1-20 comparison."""

    def compare(earlier: Optional[float], here: float, digits: int = 3) -> List[str]:
        if earlier is None:
            return ["", f"{here:.{digits}f}", ""]
        gap = round(here - earlier, digits) + 0.0  # no "-0.000"
        return [f"{earlier:.{digits}f}", f"{here:.{digits}f}", f"{gap:+.{digits}f}"]

    lines = [
        f"## 1. Check against the seeds {REFERENCE_SEEDS[0]}–{REFERENCE_SEEDS[-1]} comparison",
        "",
        f"`earlier` is the seeds {REFERENCE_SEEDS[0]}–{REFERENCE_SEEDS[-1]} comparison of the "
        "two Safety conventions, made with the same method when the landing reward's double "
        "count was settled. Its landing risk, novelty and Securing-after-30-landings values "
        "are published in `docs/subrep_metamo_integration.md`, \"Payoff and Safety\"; the "
        "crash and timeout values were not published.",
        "",
        "`here` is the same method as below, limited to those seeds: the stimulus of the "
        "task set's median outcome, and the governor runs fed that outcome. `difference` is "
        "here minus earlier.",
        "",
    ]
    body, worst = [], 0.0
    for name, result in check.items():
        for kind, earlier in REFERENCE_STIMULUS[name].items():
            stimulus = result["representative_stimulus"].get(kind)
            if stimulus is None:
                body.append([name, kind, "", "", "not observed", ""])
                continue
            for field, value in zip(STIMULUS_FIELDS, earlier):
                here = getattr(stimulus, field)
                worst = max(worst, abs(here - value))
                body.append([name, kind, field] + compare(value, here))
        runs = {
            "30 landings": result["fixed"].get(LANDING),
            "30 crashes": result["fixed"].get(CRASH),
            "8 crashes then 16 landings": result["sequences"].get("crashes_then_landings"),
        }
        for label, (securing, epsilon) in REFERENCE_RUNS[name].items():
            states = runs[label]
            if states is None:
                body.append([name, label, "", "", "not observed", ""])
                continue
            worst = max(worst, abs(states[-1]["securing"] - securing))
            body.append([name, label, "securing at the end"] + compare(securing, states[-1]["securing"]))
            if epsilon is not None:
                worst = max(worst, abs(states[-1]["epsilon"] - epsilon))
            body.append([name, label, "eps at the end"] + compare(epsilon, states[-1]["epsilon"]))
    lines += table(("convention", "case", "value", "earlier", "here", "difference"), body)
    lines += [f"Largest absolute difference: {worst:.4f}.", ""]
    return lines


def summary_section(results: Dict[str, Dict[str, Any]]) -> List[str]:
    """The headline numbers, computed from the measurements below."""
    body = []
    for name, result in results.items():
        task, everything = risk_comparison(result, TASK_SET), risk_comparison(result, None)
        fixed, seq = result["fixed"], result["sequences"].get("crashes_then_landings")
        if task is None or seq is None:
            continue
        after_crashes, at_end = seq[SEQUENCE_SHORT], seq[-1]
        first_open = next(
            (i - SEQUENCE_SHORT for i in range(SEQUENCE_SHORT + 1, len(seq))
             if seq[i]["epsilon"] > 0.0),
            None,
        )
        landing_ends = [
            end["securing"]
            for key, ends in result["per_episode_fixed"].items()
            if key[0] == LANDING and key[1] == TASK_SET
            for end in ends
        ]
        landing_eps = [
            end["epsilon"]
            for key, ends in result["per_episode_fixed"].items()
            if key[0] == LANDING and key[1] == TASK_SET
            for end in ends
        ]
        body += [
            [name, "risk of the median landing / crash (task set)",
             f"{result['representative_stimulus'][LANDING].risk:.3f} / "
             f"{result['representative_stimulus'][CRASH].risk:.3f}"],
            [name, "landing risk across episodes, 10%–90% (task set)",
             f"{np.percentile(task['land'], 10):.3f}–{np.percentile(task['land'], 90):.3f}"],
            [name, "crash risk across episodes, 10%–90% (task set)",
             f"{np.percentile(task['crash'], 10):.3f}–{np.percentile(task['crash'], 90):.3f}"],
            [name, "a random crash reads riskier than a random landing: task set / all sets",
             f"{task['separation']:.0%} / {everything['separation']:.0%}"],
            [name, "securing fixed point, landing / crash",
             f"{fixed[LANDING][-1]['securing']:.3f} / {fixed[CRASH][-1]['securing']:.3f}"],
            [name, "eps fixed point, landing / crash",
             f"{fixed[LANDING][-1]['epsilon']:.4f} / {fixed[CRASH][-1]['epsilon']:.4f}"],
            [name, "securing fixed point of each single landing, min–max across episodes",
             f"{min(landing_ends):.3f}–{max(landing_ends):.3f}"],
            [name, "eps fixed point of each single landing, min–max across episodes",
             f"{min(landing_eps):.4f}–{max(landing_eps):.4f}"],
            [name, f"securing after {SEQUENCE_SHORT} crashes, then after "
             f"{SEQUENCE_LONG} landings",
             f"{after_crashes['securing']:.3f} → {at_end['securing']:.3f}"],
            [name, "eps over that recovery",
             f"{after_crashes['epsilon']:.4f} → {at_end['epsilon']:.4f}"],
            [name, "landings before eps leaves 0",
             "never" if first_open is None else str(first_open)],
            [name, "eps fixed point, abstention noop / stop",
             f"{fixed[ABSTAIN_NOOP][-1]['epsilon']:.4f} / {fixed[ABSTAIN_STOP][-1]['epsilon']:.4f}"],
        ]
    return ["## Summary", ""] + table(("convention", "quantity", "value"), body)


def fixed_point_section(results: Dict[str, Dict[str, Any]]) -> List[str]:
    names = list(results)
    lines = [
        "## 3. Fixed points",
        "",
        f"A fresh governor fed {FIXED_POINT_STEPS} identical outcomes: the median outcome "
        "of each type in the task set. The outcome fed, and the stimulus it gives a "
        "fresh governor:",
        "",
    ]
    body = []
    for kind in ALL_TYPES:
        for name in names:
            rep = results[name]["representatives"][kind]
            if rep is None:
                body.append([kind, name, "not observed"] + [""] * 7)
                continue
            stimulus = results[name]["representative_stimulus"][kind]
            body.append(
                [kind, name, f"{rep.delta_r:+.0f}", f"{rep.delta_n[0]:+.1f}",
                 f"{rep.delta_n[1]:+.1f}"]
                + [f"{getattr(stimulus, f):.3f}" for f in STIMULUS_FIELDS]
            )
    lines += table(
        ("outcome", "convention", "Δr", "ΔSafety", "ΔFuel") + STIMULUS_FIELDS, body
    )

    lines += [
        f"State after {FIXED_POINT_STEPS} steps. `last change` is the Securing change "
        "over the final step.",
        "",
    ]
    body = []
    for kind in ALL_TYPES:
        for name in names:
            states = results[name]["fixed"].get(kind)
            if states is None:
                continue
            last = states[-1]
            body.append(
                [kind, name]
                + [number(f, last[f]) for f in TRACE_FIELDS]
                + [f"{last['securing'] - states[-2]['securing']:+.1e}",
                   f"{max(s['securing'] for s in states):.3f}",
                   str(sum(1 for s in states[1:] if s["epsilon"] == 0.0))]
            )
    lines += table(
        ("outcome", "convention") + TRACE_FIELDS
        + ("last change", "max securing", "steps with eps = 0"),
        body,
    )

    lines += [
        "Spread across seeds: each single episode's own outcome fed "
        f"{FIXED_POINT_STEPS} times, and the state it ends in. Cells are the median "
        f"and the range across episodes; policies with at least {MIN_EPISODES_PER_POLICY} "
        "episodes of the type.",
        "",
    ]
    header = (
        ("outcome: set: policy", "n", "convention") + TRACE_FIELDS
        + ("episodes ending at eps = 0",)
    )
    shown, folded = [], []
    for key in sorted(results[names[0]]["per_episode_fixed"]):
        for name in names:
            ends = results[name]["per_episode_fixed"].get(key, [])
            if len(ends) < MIN_EPISODES_PER_POLICY:
                continue
            label = key[0] if key[0] == ABSTAIN_NOOP else f"{key[0]}: {key[1]}: {key[2]}"
            body = shown if key[1] in (TASK_SET, BASELINE_SET) else folded
            body.append(
                [label, str(len(ends)), name]
                + [spread([e[f] for e in ends], 4 if f == "epsilon" else 3) for f in TRACE_FIELDS]
                + [str(sum(1 for e in ends if e["epsilon"] == 0.0))]
            )
    lines += table(header, shown)
    lines += ["<details><summary>The borderline and trade-off sets</summary>", ""]
    lines += table(header, folded)
    lines += ["</details>", ""]

    lines += [
        "Sensitivity to the appraisal scales: the same landing and crash outcomes, "
        f"state after {FIXED_POINT_STEPS} steps.",
        "",
    ]
    body = []
    for name in names:
        result = results[name]
        choices = {"task set on the first seed (used throughout)": {
            "scales": result["scales"],
            LANDING: result["fixed"][LANDING][-1],
            CRASH: result["fixed"][CRASH][-1],
        }}
        choices.update(result["scale_sensitivity"])
        for label, entry in choices.items():
            body.append(
                [name, label, f"{entry['scales'][0]:.1f} / {entry['scales'][1]:.1f}",
                 f"{entry[LANDING]['securing']:.3f}", f"{entry[LANDING]['epsilon']:.4f}",
                 f"{entry[CRASH]['securing']:.3f}", f"{entry[CRASH]['epsilon']:.4f}"]
            )
    lines += table(
        ("convention", "scales from", "payoff / motive", "landing securing",
         "landing eps", "crash securing", "crash eps"),
        body,
    )

    lines += ["Per step (step 0 is the fresh governor):", ""]
    for kind in ALL_TYPES:
        traces = {n: results[n]["fixed"][kind] for n in names if kind in results[n]["fixed"]}
        if not traces:
            continue
        lines += [f"<details><summary>{kind}</summary>", ""]
        lines += side_by_side(traces, None)
        lines += ["</details>", ""]
    return lines


def sequence_section(results: Dict[str, Dict[str, Any]]) -> List[str]:
    names = list(results)
    lines = ["## 4. Sequences", ""]
    plans = {
        "crashes_then_landings": (
            f"{SEQUENCE_SHORT} crashes then {SEQUENCE_LONG} landings",
            [CRASH] * SEQUENCE_SHORT + [LANDING] * SEQUENCE_LONG,
        ),
        "landings_then_crashes": (
            f"{SEQUENCE_LONG} landings then {SEQUENCE_SHORT} crashes",
            [LANDING] * SEQUENCE_LONG + [CRASH] * SEQUENCE_SHORT,
        ),
    }
    for key, (title, fed) in plans.items():
        traces = {n: results[n]["sequences"][key] for n in names if key in results[n]["sequences"]}
        lines += [f"### {title}", ""]
        if not traces:
            lines += ["Not run: landing or crash not observed.", ""]
            continue
        switch = fed.index(fed[-1])
        body = []
        for name in names:
            states = traces[name]
            body.append(
                [name]
                + [f"{states[i]['securing']:.3f}" for i in (0, switch, len(fed))]
                + [f"{states[i]['epsilon']:.4f}" for i in (0, switch, len(fed))]
            )
        lines += table(
            ("convention", "securing start", f"securing after step {switch}",
             f"securing after step {len(fed)}", "eps start",
             f"eps after step {switch}", f"eps after step {len(fed)}"),
            body,
        )
        lines += side_by_side(traces, ["-"] + fed)
    return lines


def report(
    episodes: Sequence[Dict[str, Any]],
    results: Dict[str, Dict[str, Any]],
    check: Optional[Dict[str, Dict[str, Any]]],
    seeds: Sequence[int],
) -> str:
    names = list(results)
    counts: Dict[str, int] = {}
    for e in episodes:
        counts[e["result"]] = counts.get(e["result"], 0) + 1
    noop = [e["result"] for e in episodes if e["set"] == BASELINE_SET]
    per_set = {
        s: len({e["policy"] for e in episodes if e["set"] == s})
        for s in sorted({e["set"] for e in episodes} - {BASELINE_SET})
    }
    lines = [
        "# Caution stimulus measurement",
        "",
        "Generated by `experiments/caution_recovery/measure.py`. Not hand-edited.",
        "",
        "## Setup",
        "",
        f"- Seeds {seeds[0]}–{seeds[-1]}, {len(episodes)} episodes flown once "
        f"({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}). "
        f"The noop baseline: {', '.join(f'{k} {noop.count(k)}' for k in sorted(set(noop)))}.",
        "- Candidates: " + ", ".join(f"{s} set ({n})" for s, n in per_set.items())
        + ", from `experiments/epsilon_scale/collect.py`.",
        f"- MetaMo commit `{metamo_commit()}`; governor from `make_lunar_lander_governor`; "
        f"flown on {platform.system()}.",
        "- Risk is the bridge's default: the worst objective change against noop. "
        "No outcome carries a measured risk.",
        "- Two Safety conventions, both computed from the same episodes:",
    ]
    for name in names:
        payoff, motive = results[name]["scales"]
        lines.append(
            f"  - **{CONVENTION_TITLES[name]}**: payoff scale {payoff:.1f}, "
            f"motive scale {motive:.1f}"
        )
    lines += [
        f"- Scales use the demo's formula (`appraisal_scales`) on the task set on seed {seeds[0]}. "
        "The demo applies it to whichever candidate set it runs (66.7 / 266.7 with "
        "`--candidates ppo`), so eps and Securing here are not directly comparable with the demo's.",
        "- Improvements are against the noop baseline flown from the same seed. "
        "Flown outcomes are appraised as admitted; a rejected one adds 0.25 to risk.",
        "- Conduciveness in sections 1 and 2 uses a fresh governor's weights "
        f"(Safety {results[names[0]]['initial_weights'][SAFETY]:.3f}).",
        "- eps is clipped to [0, 0.5]; 0.0000 means it is at the lower bound.",
        "",
    ]
    lines += summary_section(results)
    if check is not None:
        lines += reference_section(check)
    lines += [
        f"## 2. Stimulus by outcome type (seeds {seeds[0]}–{seeds[-1]})",
        "",
        "Cells are the median, with the range in brackets when the episodes differ.",
        "",
    ]
    for name in names:
        lines += [f"### {CONVENTION_TITLES[name]}", ""]
        lines += stimulus_section(results[name])
        lines += risk_verdict(results[name])
    lines += fixed_point_section(results)
    lines += sequence_section(results)
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_seeds(text: str) -> List[int]:
    """`1-40` or `1,2,5`. Only development seeds are accepted."""
    if "-" in text:
        first, last = (int(part) for part in text.split("-", 1))
        seeds = list(range(first, last + 1))
    else:
        seeds = [int(part) for part in text.split(",")]
    outside = [s for s in seeds if s not in ALLOWED_SEEDS]
    if not seeds or outside:
        raise argparse.ArgumentTypeError(
            f"seeds must lie in {ALLOWED_SEEDS.start}-{ALLOWED_SEEDS.stop - 1}, got {text!r}"
        )
    return seeds


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=parse_seeds, default=list(ALLOWED_SEEDS))
    parser.add_argument("--rollouts", type=Path,
                        default=REPO_ROOT / "runs" / "caution_recovery" / "rollouts.json",
                        help="where the flown episodes are stored and reused from")
    parser.add_argument("--refresh", action="store_true", help="fly again even if stored")
    parser.add_argument("--out", type=Path, default=HERE / "results.md")
    args = parser.parse_args(argv)

    if not PILOT_CHECKPOINT.is_file():
        print(f"error: needs the PPO checkpoint at {PILOT_CHECKPOINT}", file=sys.stderr)
        return 2

    episodes = load_or_fly(args.rollouts, args.seeds, args.refresh)
    first = args.seeds[0]
    results = {
        name: measure(episodes, name, args.seeds, first, spread=True) for name in CONVENTIONS
    }
    # The check only applies when the reference seeds were all flown.
    check = None
    if set(REFERENCE_SEEDS) <= set(args.seeds):
        check = {
            name: measure(episodes, name, list(REFERENCE_SEEDS), first, spread=False)
            for name in CONVENTIONS
        }
    args.out.write_text(report(episodes, results, check, args.seeds), encoding="utf-8")
    print(f"saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

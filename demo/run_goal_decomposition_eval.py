"""Evaluate goal decomposition against the baseline controllers on unseen seeds.

Every controller flies one episode from the start of every seed:

- idle, Gymnasium's heuristic and the PPO pilot, as in the baseline script;
- the two fixed landing sequences, flown with the runtime checks but without
  certification or replanning;
- the goal runtime for ``land_on_pad`` under several priority profiles, which
  re-certifies the rest of the plan after every skill. When it ends without
  achieving the goal and the episode goes on, the lander does nothing for the
  rest of the episode, as SubRep does without a certificate.

The evaluation also checks the measurements behind the certificates:

- feedback: for every plan the runtime certified, at the start of an episode or
  in mid-flight, and then flew to the end, the gains it really achieved from that
  state are compared with its certificate;
- the effect table: the plans are measured again on the unseen seeds and
  compared with the committed table;
- ``recover_stability`` is pursued from every tilted start.

The seeds must not overlap those the effect table was built from, and every
episode's result is kept so that any failure can be traced to its seed. The
criteria at the top of this file were written before the evaluation seeds were
flown, and the report says which of them were met.

    python -m demo.run_goal_decomposition_eval
    python -m demo.run_goal_decomposition_eval --seeds 50 --start-seed 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from decomposition.domain import Domain
from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable, build_table
from decomposition.executor import execute
from decomposition.feedback import FeedbackLog, realised_gains
from decomposition.mdn_priorities import priorities_from_weights
from decomposition.planner import plans_for
from decomposition.recipe_book import LUNAR_LANDER_BOOK, RecipeBook, load_recipe_book
from decomposition.runtime import GoalRun, GoalStatus, pursue
from decomposition.verifier import Priorities
from demo.run_lunar_baseline import CONTROLLERS, EpisodeResult, run_episode, summarise
from demo.run_lunar_subskill_reliability import SEQUENCES
from demo.train_lunar_mdn import load_landing_mdn
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import FACT_NAMES, OUTCOMES, landing_outcome, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import NO_ENGINE

GOAL = "land_on_pad"
_GAINS = ("delta_r", "delta_safety", "delta_fuel")
# The table was built from seeds 0-999. Seeds 2000-2499 and 6000-6499 were looked at
# while reviewing the plans, and 3000-3199 while tuning skills, so these are fresh.
DEFAULT_START_SEED = 8000
DEFAULT_SEEDS = 500
TIMESTAMP = "2026-01-01T00:00:00+00:00"

# Written before the evaluation seeds were flown.
MIN_ON_PAD = 0.95
MAX_CRASHES = 0.02
MIN_RECERTIFIED = 0.90

PROFILES: Dict[str, Priorities] = {
    "safety_first": priorities_from_weights((0.7, 0.3), cvar_tail_level=0.1),
    "balanced": priorities_from_weights((0.5, 0.5)),
    "fuel_first": priorities_from_weights((0.3, 0.7), cvar_tail_level=0.1),
    "no_region": Priorities(weights=(0.5, 0.5)),
}


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


def finish_episode(
    env: SubRepEnv,
    seed: int,
    rewards: np.ndarray,
    observation: np.ndarray,
    info: Optional[Mapping[str, Any]],
    episode_over: bool,
) -> EpisodeResult:
    """Do nothing until the episode ends, then classify the whole episode."""
    totals = np.asarray(rewards, dtype=np.float64).reshape(-1, 2).sum(axis=0)
    frames = len(rewards)
    while not episode_over:
        observation, reward, terminated, truncated, info = env.step(NO_ENGINE)
        totals += reward
        frames += 1
        episode_over = terminated or truncated
    return EpisodeResult(
        seed=seed,
        outcome=landing_outcome(info or {}, observation),
        safety=float(totals[0]),
        fuel=float(totals[1]),
        frames=frames,
        final_x=float(observation[0]),
    )


def fixed_sequence_episode(env: SubRepEnv, book: RecipeBook, skills: Sequence[str], seed: int) -> EpisodeResult:
    """Fly a fixed sequence with the runtime checks but no certification or replanning."""
    observation, _ = env.reset(seed=seed)
    execution = execute(LUNAR_LANDER, env, book, skills, observation)
    return finish_episode(
        env, seed, execution.rewards, execution.observation, execution.info, execution.episode_over
    )


def evaluate_profile(
    env: SubRepEnv, book: RecipeBook, table: EffectTable, priorities: Any, seeds: Sequence[int]
) -> Dict[str, Any]:
    """Pursue the goal from every seed under one source of priorities."""
    episodes: List[EpisodeResult] = []
    statuses: Counter = Counter()
    first_plans: Counter = Counter()
    stops: Counter = Counter()
    replanned = rechecks = recertified = switches = predictions_checked = predictions_held = 0
    feedback = FeedbackLog(LUNAR_LANDER)

    records: List[Dict[str, Any]] = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        run = pursue(LUNAR_LANDER, env, observation, GOAL, book, table, priorities, timestamp=TIMESTAMP)
        episode = finish_episode(env, seed, run.rewards, run.observation, run.info, run.episode_over)
        episodes.append(episode)

        statuses[run.status.value] += 1
        replanned += run.replans > 0
        rechecks += run.rechecks
        recertified += run.recertified_rechecks
        switches += run.switches
        first = run.decisions[0].chosen if run.decisions else None
        first_plans[first.entry.skill_id if first else "none"] += 1
        records.append(
            {
                **asdict(episode),
                "status": run.status.value,
                "plan": first.entry.skill_id if first else None,
                "replans": run.replans,
            }
        )
        for decision in run.flown:
            stopped = decision.execution.stopped_at
            if stopped is not None:
                stops[f"{stopped.skill}: {stopped.status.value}"] += 1
            checked = [step.prediction_held for step in decision.execution.steps if step.prediction_held is not None]
            predictions_checked += len(checked)
            predictions_held += sum(checked)
        record_followed_plans(LUNAR_LANDER, feedback, env, seed, run)

    return {
        "summary": summarise(episodes),
        "statuses": dict(statuses),
        "first_plans": dict(first_plans),
        "episodes_with_replans": replanned,
        "rechecks": rechecks,
        "recertified_rechecks": recertified,
        "switches": switches,
        "plan_stops": dict(stops),
        "predictions": {"checked": predictions_checked, "held": predictions_held},
        "feedback": feedback.report(),
        "episodes": records,
    }


def record_followed_plans(domain: Domain, feedback: FeedbackLog, env: Any, seed: int, run: GoalRun) -> None:
    """Record what each certified plan really achieved, for those flown to the end from where they were certified."""
    frames = 0
    followed_from = _last_change(run)
    for index, decision in enumerate(run.flown):
        if decision.recertified and index >= followed_from:
            start = run.actions[:frames]
            feedback.record(
                decision.chosen, decision.facts, realised_gains(domain, env, seed, run.rewards[frames:], start)
            )
        frames += decision.execution.frames


def _last_change(run: GoalRun) -> int:
    """The index of the first flown decision from which the run followed one plan to its end."""
    flown = run.flown
    index = len(flown) - 1
    while index > 0 and flown[index - 1].execution.completed and flown[index].skills == flown[index - 1].skills[1:]:
        index -= 1
    return index


# ---------------------------------------------------------------------------
# Checks of the measurements
# ---------------------------------------------------------------------------


def check_table(committed: EffectTable, book: RecipeBook, seeds: range) -> List[Dict[str, Any]]:
    """Measure the plans again on ``seeds`` and compare each group with the committed table."""
    unseen = build_table(LUNAR_LANDER, seeds, plans_for(book), min_samples=1)
    fresh = {(entry["name"], entry["group"]): entry for entry in unseen.data["plans"]}
    rows = []
    for entry in committed.data["plans"]:
        if entry["samples"] < committed.min_samples or (entry["name"], entry["group"]) not in fresh:
            continue
        new = fresh[(entry["name"], entry["group"])]
        rows.append(
            {
                "plan": entry["name"],
                "group": entry["group"],
                "samples": new["samples"],
                "committed_mean_delta_r": entry["delta_r"]["mean"],
                "committed_lower_delta_r": entry["delta_r"]["lower"],
                "unseen_mean_delta_r": new["delta_r"]["mean"],
                "below_committed_lower": [name for name in _GAINS if new[name]["mean"] < entry[name]["lower"]],
            }
        )
    return rows


def check_recovery(
    env: SubRepEnv, book: RecipeBook, table: EffectTable, priorities: Priorities, seeds: Sequence[int]
) -> Dict[str, Any]:
    """Pursue ``recover_stability`` from every tilted start."""
    statuses: Counter = Counter()
    frames, fuel = [], []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        if lander_facts(observation)["upright"]:
            continue
        run = pursue(LUNAR_LANDER, env, observation, "recover_stability", book, table, priorities, timestamp=TIMESTAMP)
        statuses[run.status.value] += 1
        if run.status is GoalStatus.ACHIEVED:
            frames.append(run.frames)
            fuel.append(float(run.rewards[:, 1].sum()))
    return {
        "tilted_starts": sum(statuses.values()),
        "statuses": dict(statuses),
        "mean_frames": float(np.mean(frames)) if frames else None,
        "mean_fuel": float(np.mean(fuel)) if fuel else None,
    }


# ---------------------------------------------------------------------------
# The whole evaluation
# ---------------------------------------------------------------------------


def evaluate(
    seeds: range,
    profiles: Mapping[str, Priorities] = PROFILES,
    baselines: Sequence[str] = tuple(CONTROLLERS),
    use_mdn: bool = True,
) -> Dict[str, Any]:
    """Fly every controller from every seed and check the measurements behind the certificates."""
    table = EffectTable.load(DEFAULT_TABLE_PATH, LUNAR_LANDER)
    measured = range(table.data["seeds"]["start"], table.data["seeds"]["stop"])
    if len(seeds) == 0:
        raise ValueError("the evaluation needs at least one seed")
    if seeds.start < measured.stop and measured.start < seeds.stop:
        raise ValueError(
            f"evaluation seeds {seeds.start}-{seeds.stop - 1} overlap seeds {measured.start}-{measured.stop - 1}, "
            "which the effect table was built from"
        )
    book = load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, LUNAR_LANDER.skills)
    env = SubRepEnv(seed=int(seeds[0]))
    controllers: Dict[str, Dict[str, Any]] = {}
    goal_runs: Dict[str, Dict[str, Any]] = {}
    try:
        for name in baselines:
            policy = CONTROLLERS[name](env)
            controllers[name] = _results([run_episode(env, policy, int(seed)) for seed in seeds])
        for name, steps in SEQUENCES.items():
            episodes = [fixed_sequence_episode(env, book, steps, int(seed)) for seed in seeds]
            controllers[f"fixed_{name}"] = _results(episodes)
        for name, priorities in profiles.items():
            goal_runs[name] = evaluate_profile(env, book, table, priorities, [int(seed) for seed in seeds])
            controllers[f"runtime_{name}"] = {"summary": goal_runs[name]["summary"]}
        if use_mdn:
            goal_runs["mdn"] = evaluate_profile(env, book, table, load_landing_mdn(), [int(seed) for seed in seeds])
            controllers["runtime_mdn"] = {"summary": goal_runs["mdn"]["summary"]}
        recovery = check_recovery(env, book, table, PROFILES["safety_first"], [int(seed) for seed in seeds])
    finally:
        env.close()

    report = {
        "seeds": {"start": int(seeds[0]), "count": len(seeds)},
        "goal": GOAL,
        "profiles": {name: _profile_record(p) for name, p in profiles.items()},
        "controllers": controllers,
        "goal_runs": goal_runs,
        "table_check": check_table(table, book, seeds),
        "recover_stability": recovery,
    }
    report["criteria"] = check_criteria(report)
    return report


def check_criteria(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Each criterion written before the evaluation, with its result."""
    if "safety_first" not in report["goal_runs"]:
        return []
    run = report["goal_runs"]["safety_first"]
    summary = run["summary"]
    revoked = [e for each in report["goal_runs"].values() for e in each["feedback"] if e["status"] == "revoked"]
    below = [row for row in report["table_check"] if row["below_committed_lower"]]
    recertified = run["recertified_rechecks"] / max(1, run["rechecks"])
    predictions = run["predictions"]
    held = predictions["held"] / max(1, predictions["checked"])
    return [
        _criterion("On the pad under safety-first priorities", summary["landed_on_pad_rate"], f">= {MIN_ON_PAD}", summary["landed_on_pad_rate"] >= MIN_ON_PAD),
        _criterion("Crashes under safety-first priorities", summary["crashed_rate"], f"<= {MAX_CRASHES}", summary["crashed_rate"] <= MAX_CRASHES),
        _criterion("Mid-flight re-checks admitted from the real state", recertified, f">= {MIN_RECERTIFIED}", recertified >= MIN_RECERTIFIED),
        _criterion("Book predictions that held", held, ">= 0.99", held >= 0.99),
        _criterion("Certificates revoked by what really happened", len(revoked), "== 0", not revoked),
        _criterion("Table entries whose unseen mean fell below the committed bound", len(below), "report only", True),
    ]


def _criterion(name: str, value: float, needed: str, met: bool) -> Dict[str, Any]:
    return {"criterion": name, "value": float(value), "needed": needed, "met": bool(met)}


def _results(episodes: Sequence[EpisodeResult]) -> Dict[str, Any]:
    return {"summary": summarise(episodes), "episodes": [asdict(episode) for episode in episodes]}


def _profile_record(priorities: Priorities) -> Dict[str, Any]:
    region = priorities.region
    return {
        "weights": list(priorities.weights),
        "pds_epsilon": priorities.pds_epsilon,
        "cvar_tail_level": priorities.cvar_tail_level,
        "region_support_values": None if region is None else list(region.support_values),
        "mdn_alpha": None if region is None else list(region.alpha),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def to_markdown(report: Mapping[str, Any]) -> str:
    seeds = report["seeds"]
    lines = [
        "# Goal decomposition evaluation",
        "",
        f"Goal `{report['goal']}`, seeds {seeds['start']}-{seeds['start'] + seeds['count'] - 1} "
        f"({seeds['count']} episodes per controller).",
        "",
        "## Criteria written before the evaluation",
        "",
        "| Criterion | Value | Needed | Met |",
        "|---|---|---|---|",
        *(
            f"| {row['criterion']} | {row['value']:.3f} | {row['needed']} | {'yes' if row['met'] else 'NO'} |"
            for row in report["criteria"]
        ),
        "",
        "## Controllers",
        "",
        "| Controller | On pad | Off pad | Crashed | Timeout | Mean Safety | Mean Fuel | Mean frames |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, data in report["controllers"].items():
        s = data["summary"]
        rates = " | ".join(f"{s[f'{outcome}_rate']:.3f}" for outcome in OUTCOMES)
        lines.append(f"| {name} | {rates} | {s['mean_safety']:.1f} | {s['mean_fuel']:.1f} | {s['mean_frames']:.0f} |")

    lines += [
        "",
        "## Goal runtime",
        "",
        "| Profile | Statuses | First plan | Mid-flight re-checks | Re-certified | Switches | "
        "Episodes with replans | Predictions held | Plan stops |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in report["goal_runs"].items():
        predictions = run["predictions"]
        lines.append(
            f"| {name} | {_counts(run['statuses'])} | {_counts(run['first_plans'])} | "
            f"{run['rechecks']} | {run['recertified_rechecks']} | {run['switches']} | "
            f"{run['episodes_with_replans']} | {predictions['held']}/{predictions['checked']} | "
            f"{_counts(run['plan_stops']) or '-'} |"
        )

    lines += [
        "",
        "## Certificates against realised gains",
        "",
        "| Profile | Plan | Start group | Samples | Certified Δr | Realised mean Δr | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in report["goal_runs"].items():
        for entry in run["feedback"]:
            lines.append(
                f"| {name} | {entry['plan']} | {entry['group']} | {entry['samples']} | "
                f"{entry['delta_r']['certified']:.1f} | {entry['delta_r']['realised_mean']:.1f} | {entry['status']} |"
            )

    lines += [
        "",
        "## Effect table on unseen seeds",
        "",
        "| Plan | Start group | Samples | Committed mean Δr | Committed lower Δr | Unseen mean Δr "
        "| Unseen mean below lower |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["table_check"]:
        lines.append(
            f"| {row['plan']} | {row['group']} | {row['samples']} | {row['committed_mean_delta_r']:.1f} | "
            f"{row['committed_lower_delta_r']:.1f} | {row['unseen_mean_delta_r']:.1f} | "
            f"{', '.join(row['below_committed_lower']) or 'none'} |"
        )

    recovery = report["recover_stability"]
    lines += [
        "",
        "## Recover stability",
        "",
        f"{recovery['tilted_starts']} tilted starts: {_counts(recovery['statuses'])}; "
        f"mean frames {_number(recovery['mean_frames'])}, mean Fuel {_number(recovery['mean_fuel'])}.",
    ]
    return "\n".join(lines) + "\n"


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _number(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.1f}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the reports hold symbols a Windows console encoding cannot write
    parser = argparse.ArgumentParser(description="Evaluate goal decomposition against baseline controllers.")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--output", default="demo/artifacts/goal_decomposition_eval.json")
    parser.add_argument("--markdown-output", default="demo/artifacts/goal_decomposition_eval.md")
    args = parser.parse_args()

    seeds = range(args.start_seed, args.start_seed + args.seeds)
    report = evaluate(seeds)
    markdown = to_markdown(report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {output} and {args.markdown_output}")


if __name__ == "__main__":
    main()

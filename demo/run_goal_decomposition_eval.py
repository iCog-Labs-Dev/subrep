"""Evaluate goal decomposition against the baseline controllers on unseen seeds.

Every controller flies one episode from the start of every seed:

- idle, Gymnasium's heuristic and the PPO pilot, as in the baseline script;
- the two fixed landing sequences, flown with the runtime checks but without
  certification or replanning;
- the goal runtime for ``land_on_pad`` under several priority profiles. When it
  ends without achieving the goal and the episode goes on, the lander does
  nothing for the rest of the episode, as SubRep does without a certificate.

The evaluation also checks the measurements behind the certificates:

- feedback: for each plan the runtime certified at the start of an episode,
  the gains it really achieved are compared with its certificate;
- the effect table: the plans are measured again on the unseen seeds and
  compared with the committed table;
- ``recover_stability`` is pursued from every tilted start.

The seeds must not overlap those the effect table was built from, and every
episode's result is kept so that any failure can be traced to its seed.

    python -m demo.run_goal_decomposition_eval
    python -m demo.run_goal_decomposition_eval --seeds 50 --start-seed 5000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable, build_table
from decomposition.executor import execute
from decomposition.feedback import FeedbackLog, realised_gains
from decomposition.recipe_book import LUNAR_LANDER_BOOK, RecipeBook, load_recipe_book
from decomposition.runtime import GoalStatus, pursue
from decomposition.verifier import MdnRegion, Priorities
from demo.build_lunar_effect_table import PLANS
from demo.run_lunar_baseline import CONTROLLERS, EpisodeResult, run_episode, summarise
from demo.run_lunar_subskill_reliability import SEQUENCES
from env.lunar_facts import FACT_NAMES, OUTCOMES, landing_outcome, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import NO_ENGINE, SUBSKILLS

GOAL = "land_on_pad"
_GAINS = ("delta_r", "delta_safety", "delta_fuel")
# The effect table was built from seeds 0-999, so the evaluation starts after them.
DEFAULT_START_SEED = 2000
DEFAULT_SEEDS = 500
TIMESTAMP = "2026-01-01T00:00:00+00:00"


def _region(fuel_cap: float, alpha: Sequence[float]) -> MdnRegion:
    """An MDN region over (Safety, Fuel) that caps the Fuel weight."""
    return MdnRegion(
        context=(0.0,),
        alpha=tuple(alpha),
        support_directions=((1.0, 0.0), (0.0, 1.0)),
        support_values=(1.0, fuel_cap),
    )


PROFILES: Dict[str, Priorities] = {
    "safety_first": Priorities(weights=(0.7, 0.3), cvar_tail_level=0.1, region=_region(0.3, (4.0, 1.0))),
    "balanced": Priorities(weights=(0.5, 0.5), region=_region(0.5, (2.0, 2.0))),
    "fuel_first": Priorities(weights=(0.3, 0.7), cvar_tail_level=0.1, region=_region(0.7, (1.0, 4.0))),
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
    execution = execute(env, book, skills, observation)
    return finish_episode(
        env, seed, execution.rewards, execution.observation, execution.info, execution.episode_over
    )


def evaluate_profile(
    env: SubRepEnv, book: RecipeBook, table: EffectTable, priorities: Priorities, seeds: Sequence[int]
) -> Dict[str, Any]:
    """Pursue the goal from every seed under one priority profile."""
    episodes: List[EpisodeResult] = []
    statuses: Counter = Counter()
    first_plans: Counter = Counter()
    stops: Counter = Counter()
    replanned = predictions_checked = predictions_held = 0
    feedback = FeedbackLog()

    records: List[Dict[str, Any]] = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        facts = lander_facts(observation)
        run = pursue(env, observation, GOAL, book, table, priorities, timestamp=TIMESTAMP)
        episode = finish_episode(env, seed, run.rewards, run.observation, run.info, run.episode_over)
        episodes.append(episode)

        statuses[run.status.value] += 1
        replanned += run.replans > 0
        first = run.attempts[0].chosen if run.attempts else None
        first_plans[first.entry.skill_id if first else "none"] += 1
        records.append(
            {
                **asdict(episode),
                "status": run.status.value,
                "plan": first.entry.skill_id if first else None,
                "replans": run.replans,
            }
        )
        for attempt in run.attempts:
            if attempt.execution is None:
                continue
            stopped = attempt.execution.stopped_at
            if stopped is not None:
                stops[f"{stopped.skill}: {stopped.status.value}"] += 1
            checked = [step.prediction_held for step in attempt.execution.steps if step.prediction_held is not None]
            predictions_checked += len(checked)
            predictions_held += sum(checked)
        if first is not None:
            feedback.record(first, facts, realised_gains(env, seed, run.rewards))

    return {
        "summary": summarise(episodes),
        "statuses": dict(statuses),
        "first_plans": dict(first_plans),
        "episodes_with_replans": replanned,
        "plan_stops": dict(stops),
        "predictions": {"checked": predictions_checked, "held": predictions_held},
        "feedback": feedback.report(),
        "episodes": records,
    }


# ---------------------------------------------------------------------------
# Checks of the measurements
# ---------------------------------------------------------------------------


def check_table(committed: EffectTable, seeds: range) -> List[Dict[str, Any]]:
    """Measure the plans again on ``seeds`` and compare each group with the committed table."""
    unseen = build_table(seeds, PLANS, min_samples=1)
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
        run = pursue(env, observation, "recover_stability", book, table, priorities, timestamp=TIMESTAMP)
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
) -> Dict[str, Any]:
    """Fly every controller from every seed and check the measurements behind the certificates."""
    table = EffectTable.load(DEFAULT_TABLE_PATH)
    measured = range(table.data["seeds"]["start"], table.data["seeds"]["stop"])
    if len(seeds) == 0:
        raise ValueError("the evaluation needs at least one seed")
    if seeds.start < measured.stop and measured.start < seeds.stop:
        raise ValueError(
            f"evaluation seeds {seeds.start}-{seeds.stop - 1} overlap seeds {measured.start}-{measured.stop - 1}, "
            "which the effect table was built from"
        )
    book = load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, SUBSKILLS)
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
        recovery = check_recovery(env, book, table, PROFILES["safety_first"], [int(seed) for seed in seeds])
    finally:
        env.close()

    return {
        "seeds": {"start": int(seeds[0]), "count": len(seeds)},
        "goal": GOAL,
        "profiles": {name: _profile_record(p) for name, p in profiles.items()},
        "controllers": controllers,
        "goal_runs": goal_runs,
        "table_check": check_table(table, seeds),
        "recover_stability": recovery,
    }


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
        "| Profile | Statuses | First plan | Episodes with replans | Predictions held | Plan stops |",
        "|---|---|---|---|---|---|",
    ]
    for name, run in report["goal_runs"].items():
        predictions = run["predictions"]
        lines.append(
            f"| {name} | {_counts(run['statuses'])} | {_counts(run['first_plans'])} | "
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

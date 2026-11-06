"""Evaluate goal decomposition on the scan-then-land mission, on seeds nothing was tuned on.

Every controller flies one episode from the start of every seed:

- idle, the heuristic and the landing PPO pilot, which never saw the beacon;
- the PPO pilot trained on the mission;
- the mission's skills in a fixed order, with the runtime checks but no certification;
- the goal runtime under several priorities, with ``scan_then_land`` and
  ``land_on_pad`` both on offer, so that priorities can decide between them.
  Mission-first priorities are also flown under the continuation policy,
  reported apart from the criteria.

It also moves the beacon and repeats the mission with the same recipe book, and
flies with a limit on how many plans are measured at each decision. Certified
plans are checked against what they really achieved after every episode has been
flown, so a revocation found there was not applied during the runs. The criteria
at the top of this file were written before the evaluation seeds were flown, and
the report says which of them were met, which were missed and which had too
little evidence to judge.

    python -m demo.run_beacon_mission_eval
    python -m demo.run_beacon_mission_eval --seeds 50 --start-seed 3000
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from decomposition.domain import Domain
from decomposition.effects import LUNAR_BEACON_TABLE, EffectTable
from decomposition.executor import execute
from decomposition.feedback import CertificateStatus, FeedbackLog
from decomposition.mdn_priorities import priorities_from_weights
from decomposition.recipe_book import RecipeBook
from decomposition.runtime import Continuation, GoalRun, pursue
from decomposition.verifier import Priorities
from demo.build_lunar_effect_table import build_mission_table, mission_book
from demo.run_goal_decomposition_eval import (
    CERTIFICATES_NOTE,
    continuation_lines,
    criteria_table,
    criterion,
    record_followed_plans,
    revocation_criterion,
    share_criterion,
    shortfalls,
)
from demo.train_lunar_mdn import load_mission_mdn
from env.lunar_beacon import BEACON_MISSION, MISSION_COMPLETE, beacon_domain
from env.lunar_pilots import mission_pilots

GOALS = ("scan_then_land", "land_on_pad")
# The mission table was built from seeds 0-1999 and the moved-beacon tables from 0-999.
# The skills were tuned on seeds 3000-3199, and an earlier evaluation flew 8000-8499.
DEFAULT_START_SEED = 10000
DEFAULT_SEEDS = 500
TIMESTAMP = "2026-01-01T00:00:00+00:00"
FIXED_MISSION = ("go_to_beacon", "cruise_to_pad", "slow_descent", "touch_down")
MOVED_BEACONS = ((0.5, 0.7), (0.3, 0.5))
MOVED_TABLE_SEEDS = range(0, 1000)
MEASURING_LIMIT = 5

# Written before the evaluation seeds were flown.
MIN_COMPLETION = 0.90
MAX_CRASHES = 0.03
MIN_RECERTIFIED = 0.90
MIN_GOAL_SPLIT = 0.80  # episodes in which mission-first and fuel-first go for different goals
MIN_MOVED_COMPLETION = 0.85
MIN_LIMIT_AGREEMENT = 0.95
SINGLE_CONTROLLER_TOLERANCE = 0.02  # the runtime can pick a single controller itself, so it should not fall behind one

PROFILES: Dict[str, Priorities] = {
    "mission_first": priorities_from_weights((0.2, 0.1, 0.7), cvar_tail_level=0.1),
    "safety_first": priorities_from_weights((0.7, 0.2, 0.1), cvar_tail_level=0.1),
    "fuel_first": priorities_from_weights((0.2, 0.7, 0.1), cvar_tail_level=0.1),
    "no_region": Priorities(weights=(0.4, 0.3, 0.3)),
}


@dataclass(frozen=True)
class MissionEpisode:
    seed: int
    outcome: str
    safety: float
    fuel: float
    mission: float
    frames: int


def _episode(domain: Domain, env: Any, seed: int, rewards: np.ndarray, observation, info, over: bool) -> MissionEpisode:
    """Do nothing until the episode ends, then classify the whole episode."""
    totals = np.asarray(rewards, dtype=np.float64).reshape(-1, domain.num_objectives).sum(axis=0)
    frames = len(rewards)
    while not over:
        observation, reward, terminated, truncated, info = env.step(domain.idle_action)
        totals += reward
        frames += 1
        over = terminated or truncated
    return MissionEpisode(seed, domain.outcome(info or {}, observation), *map(float, totals), frames)


def fly_controller(domain: Domain, env: Any, policy: Callable[[np.ndarray], int], seed: int) -> MissionEpisode:
    observation, _ = env.reset(seed=seed)
    rewards: List[np.ndarray] = []
    while True:
        observation, reward, terminated, truncated, info = env.step(int(policy(observation)))
        rewards.append(reward)
        if terminated or truncated:
            return _episode(domain, env, seed, np.asarray(rewards), observation, info, True)


def fly_fixed(domain: Domain, env: Any, book: RecipeBook, seed: int) -> MissionEpisode:
    observation, _ = env.reset(seed=seed)
    execution = execute(domain, env, book, FIXED_MISSION, observation)
    return _episode(domain, env, seed, execution.rewards, execution.observation, execution.info, execution.episode_over)


def rates(episodes: Sequence[MissionEpisode], outcomes: Sequence[str]) -> Dict[str, float]:
    count = len(episodes)
    return {
        "episodes": count,
        **{f"{outcome}_rate": sum(e.outcome == outcome for e in episodes) / count for outcome in outcomes},
        "mean_safety": float(np.mean([e.safety for e in episodes])),
        "mean_fuel": float(np.mean([e.fuel for e in episodes])),
        "mean_mission": float(np.mean([e.mission for e in episodes])),
        "mean_frames": float(np.mean([e.frames for e in episodes])),
    }


def wilson_interval(successes: int, count: int, z: float = 1.96) -> Tuple[float, float]:
    """Two-sided 95% confidence interval of a rate."""
    rate = successes / count
    centre = rate + z * z / (2 * count)
    margin = z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
    return float((centre - margin) / (1 + z * z / count)), float((centre + margin) / (1 + z * z / count))


def evaluate_runtime(
    domain: Domain,
    env: Any,
    book: RecipeBook,
    table: EffectTable,
    priorities: Any,
    seeds: Sequence[int],
    limit: Optional[int] = None,
    check_certificates: bool = True,
    continuation: Continuation = Continuation.STOP,
) -> Dict[str, Any]:
    """Pursue the mission's goals from every seed under one source of priorities."""
    episodes: List[MissionEpisode] = []
    statuses: Counter = Counter()
    first_goals: Counter = Counter()
    first_plans: Counter = Counter()
    stops: Counter = Counter()
    rechecks = recertified = continued = switches = replans = checked = held = 0
    feedback = FeedbackLog(domain)
    records: List[Dict[str, Any]] = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        run: GoalRun = pursue(
            domain,
            env,
            observation,
            GOALS,
            book,
            table,
            priorities,
            timestamp=TIMESTAMP,
            limit=limit,
            continuation=continuation,
        )
        episode = _episode(domain, env, seed, run.rewards, run.observation, run.info, run.episode_over)
        episodes.append(episode)
        statuses[run.status.value] += 1
        first = run.decisions[0].chosen if run.decisions else None
        first_goals[first.candidate.goal if first else "none"] += 1
        first_plans[">".join(first.candidate.skills) if first else "none"] += 1
        rechecks += run.rechecks
        recertified += run.recertified_rechecks
        continued += run.continued
        switches += run.switches
        replans += run.replans
        for decision in run.flown:
            stopped = decision.execution.stopped_at
            if stopped is not None:
                stops[f"{stopped.skill}: {stopped.status.value}"] += 1
            predictions = [s.prediction_held for s in decision.execution.steps if s.prediction_held is not None]
            checked += len(predictions)
            held += sum(predictions)
        records.append(
            {
                **asdict(episode),
                "status": run.status.value,
                "first_plan": ">".join(first.candidate.skills) if first else None,
                "goal": run.goal if run.flown else None,
            }
        )
        if check_certificates:
            record_followed_plans(domain, feedback, env, seed, run)
    return {
        "summary": rates(episodes, domain.outcomes),
        "statuses": dict(statuses),
        "first_goals": dict(first_goals),
        "first_plans": dict(first_plans),
        "rechecks": rechecks,
        "recertified_rechecks": recertified,
        "continued": continued,
        "switches": switches,
        "replans": replans,
        "plan_stops": dict(stops),
        "predictions": {"checked": checked, "held": held},
        "feedback": feedback.report(),
        "episodes": records,
    }


def evaluate_moved_beacon(
    beacon: Tuple[float, float], book: RecipeBook, seeds: Sequence[int], priorities: Priorities
) -> Dict[str, Any]:
    """The same recipe book and skills with the beacon somewhere else, and only the measurements taken again."""
    domain = beacon_domain(beacon, mission_pilots())
    table = build_mission_table(MOVED_TABLE_SEEDS, beacon)
    env = domain.make_env(int(seeds[0]))
    try:
        run = evaluate_runtime(domain, env, book, table, priorities, seeds, check_certificates=False)
        pilot = domain.skills["mission_pilot"].policy
        trained = [fly_controller(domain, env, pilot, int(seed)) for seed in seeds]
    finally:
        env.close()
    return {
        "beacon": list(beacon),
        "table_seeds": len(MOVED_TABLE_SEEDS),
        "runtime": {key: run[key] for key in ("summary", "statuses", "first_goals")},
        "ppo_trained_on_the_first_beacon": rates(trained, domain.outcomes),
    }


def compare_limit(full: Mapping[str, Any], limited: Mapping[str, Any]) -> Dict[str, Any]:
    """How often flying with a measuring limit chose the same first plan and ended the same way."""
    pairs = list(zip(full["episodes"], limited["episodes"]))
    return {
        "limit": MEASURING_LIMIT,
        "same_first_plan_rate": float(np.mean([a["first_plan"] == b["first_plan"] for a, b in pairs])),
        "same_outcome_rate": float(np.mean([a["outcome"] == b["outcome"] for a, b in pairs])),
        "limited_completion_rate": limited["summary"][f"{MISSION_COMPLETE}_rate"],
    }


def goal_split(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """The share of episodes in which two sets of priorities went for different goals."""
    return float(np.mean([a["goal"] != b["goal"] for a, b in zip(first["episodes"], second["episodes"])]))


def check_criteria(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Each criterion written before the evaluation, with its result."""
    runs, controllers = report["goal_runs"], report["controllers"]
    mission = runs["mission_first"]
    summary = mission["summary"]
    completion = summary[f"{MISSION_COMPLETE}_rate"]
    episodes = f"{summary['episodes']} episodes"
    single = {
        name: data["summary"][f"{MISSION_COMPLETE}_rate"]
        for name, data in controllers.items()
        if name.startswith(("heuristic", "ppo", "mission_ppo"))
    }
    best = max(single, key=single.get)
    moved = {str(m["beacon"]): m["runtime"]["summary"][f"{MISSION_COMPLETE}_rate"] for m in report["moved_beacons"]}
    return [
        criterion("Mission completion under mission-first priorities", completion, ">=", MIN_COMPLETION, episodes),
        criterion("Crashes under mission-first priorities", summary["crashed_rate"], "<=", MAX_CRASHES, episodes),
        criterion(
            "Best single controller's completion (runtime must be within 0.02)",
            single[best],
            "<=",
            round(completion + SINGLE_CONTROLLER_TOLERANCE, 3),
            best,
        ),
        criterion(
            "Priorities change the goal pursued (mission-first against fuel-first)",
            report["goal_split"],
            ">=",
            MIN_GOAL_SPLIT,
            episodes,
        ),
        share_criterion(
            "Mid-flight re-checks admitted from the real state",
            mission["recertified_rechecks"],
            mission["rechecks"],
            MIN_RECERTIFIED,
            "mid-flight re-checks",
        ),
        revocation_criterion(runs.values()),
        criterion(
            "Completion with the beacon moved and the same recipe book (lowest)",
            min(moved.values()),
            ">=",
            MIN_MOVED_COMPLETION,
            "beacons " + ", ".join(moved),
        ),
        criterion(
            "Same first plan with a measuring limit",
            report["limit"]["same_first_plan_rate"],
            ">=",
            MIN_LIMIT_AGREEMENT,
            episodes,
        ),
    ]


def evaluate(
    seeds: range,
    profiles: Mapping[str, Any] = PROFILES,
    moved: Sequence[Tuple[float, float]] = MOVED_BEACONS,
    continuing: Sequence[str] = ("mission_first",),
) -> Dict[str, Any]:
    """Fly every controller from every seed; the profiles in ``continuing`` also fly under the continuation policy."""
    table = EffectTable.load(LUNAR_BEACON_TABLE, BEACON_MISSION)
    measured = range(table.data["seeds"]["start"], table.data["seeds"]["stop"])
    if len(seeds) == 0:
        raise ValueError("the evaluation needs at least one seed")
    if seeds.start < measured.stop and measured.start < seeds.stop:
        raise ValueError(f"evaluation seeds overlap seeds {measured.start}-{measured.stop - 1}, which the table was built from")
    domain = BEACON_MISSION
    book = mission_book()
    env = domain.make_env(int(seeds[0]))
    int_seeds = [int(seed) for seed in seeds]
    try:
        controllers: Dict[str, Dict[str, Any]] = {}
        policies = {
            "idle": lambda obs: domain.idle_action,
            "heuristic": domain.skills["heuristic_pilot"].policy,
            "ppo_landing": domain.skills["ppo_pilot"].policy,
            "mission_ppo": domain.skills["mission_pilot"].policy,
        }
        for name, policy in policies.items():
            episodes = [fly_controller(domain, env, policy, seed) for seed in int_seeds]
            controllers[name] = {"summary": rates(episodes, domain.outcomes), "episodes": [asdict(e) for e in episodes]}
        fixed = [fly_fixed(domain, env, book, seed) for seed in int_seeds]
        controllers["fixed_mission_skills"] = {"summary": rates(fixed, domain.outcomes), "episodes": [asdict(e) for e in fixed]}
        goal_runs = {name: evaluate_runtime(domain, env, book, table, source, int_seeds) for name, source in profiles.items()}
        if "mission_first" in goal_runs and "fuel_first" in goal_runs:
            split = goal_split(goal_runs["mission_first"], goal_runs["fuel_first"])
        else:
            split = 0.0
        goal_runs["mdn"] = evaluate_runtime(domain, env, book, table, load_mission_mdn(), int_seeds)
        limited = evaluate_runtime(domain, env, book, table, profiles["mission_first"], int_seeds, MEASURING_LIMIT, False)
        continued_runs = {
            name: evaluate_runtime(
                domain, env, book, table, profiles[name], int_seeds, continuation=Continuation.KEEP_UNREJECTED_PLAN
            )
            for name in continuing
            if name in profiles
        }
    finally:
        env.close()
    moved_runs = [evaluate_moved_beacon(beacon, book, int_seeds, profiles["mission_first"]) for beacon in moved]
    report = {
        "seeds": {"start": int(seeds[0]), "count": len(seeds)},
        "goals": list(GOALS),
        "controllers": controllers,
        "goal_runs": goal_runs,
        "continuation": continued_runs,
        "goal_split": split,
        "limit": compare_limit(goal_runs["mission_first"], limited),
        "moved_beacons": moved_runs,
    }
    report["criteria"] = check_criteria(report)
    return report


def to_markdown(report: Mapping[str, Any]) -> str:
    seeds = report["seeds"]
    lines = [
        "# Scan-then-land mission evaluation",
        "",
        f"Seeds {seeds['start']}-{seeds['start'] + seeds['count'] - 1} ({seeds['count']} episodes per controller).",
        "",
        "## Criteria written before the evaluation",
        "",
        *criteria_table(report["criteria"]),
    ]
    lines += [
        "",
        "## Controllers",
        "",
        "| Controller | Mission done (95% CI) | Pad, no scan | Off pad | Crashed | Timeout | Safety | Fuel | Mission | Frames |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = [(name, data["summary"]) for name, data in report["controllers"].items()]
    rows += [(f"runtime_{name}", run["summary"]) for name, run in report["goal_runs"].items()]
    for name, s in rows:
        low, high = wilson_interval(round(s["mission_complete_rate"] * s["episodes"]), s["episodes"])
        lines.append(
            f"| {name} | {s['mission_complete_rate']:.3f} ({low:.3f}-{high:.3f}) | {s['landed_on_pad_rate']:.3f} | {s['landed_off_pad_rate']:.3f} | "
            f"{s['crashed_rate']:.3f} | {s['timeout_rate']:.3f} | {s['mean_safety']:.1f} | {s['mean_fuel']:.1f} | {s['mean_mission']:.1f} | {s['mean_frames']:.0f} |"
        )
    lines += ["", "## Goal runtime", "", "| Priorities | Statuses | First goal | Re-checks | Re-certified | Switches | Replans | Predictions held | Plan stops |", "|---|---|---|---|---|---|---|---|---|"]
    for name, run in report["goal_runs"].items():
        p = run["predictions"]
        lines.append(
            f"| {name} | {_counts(run['statuses'])} | {_counts(run['first_goals'])} | {run['rechecks']} | {run['recertified_rechecks']} | "
            f"{run['switches']} | {run['replans']} | {p['held']}/{p['checked']} | {_counts(run['plan_stops']) or '-'} |"
        )
    lines += continuation_lines(report, f"{MISSION_COMPLETE}_rate", "Mission done")
    lines += ["", "## First plans", ""]
    for name, run in report["goal_runs"].items():
        lines.append(f"- {name}: {_counts(run['first_plans'])}")
    lines += [
        "",
        "## Certificates against realised gains",
        "",
        CERTIFICATES_NOTE,
        "",
        "| Priorities | Plan | Start group | Samples | Certified Δr | Realised mean Δr | Status | Fell short on |",
        "|---|---|---|---|---|---|---|---|",
    ]
    unjudged = 0
    for name, run in report["goal_runs"].items():
        for entry in run["feedback"]:
            if entry["status"] == CertificateStatus.TOO_FEW_SAMPLES.value:
                unjudged += 1
                continue
            lines.append(
                f"| {name} | {entry['plan']} | {entry['group']} | {entry['samples']} | "
                f"{entry['delta_r']['certified']:.1f} | {entry['delta_r']['realised_mean']:.1f} | {entry['status']} | "
                f"{shortfalls(entry)} |"
            )
    lines += ["", f"{unjudged} more plan and start groups had too few samples and are not listed."]
    lines += ["", "## The beacon moved, the recipe book unchanged", "", "| Beacon | Runtime: mission done | Runtime: crashed | PPO trained on the first beacon: mission done |", "|---|---|---|---|"]
    for moved in report["moved_beacons"]:
        s = moved["runtime"]["summary"]
        lines.append(f"| {moved['beacon']} | {s['mission_complete_rate']:.3f} | {s['crashed_rate']:.3f} | {moved['ppo_trained_on_the_first_beacon']['mission_complete_rate']:.3f} |")
    limit = report["limit"]
    lines += ["", "## Measuring limit", "", f"With at most {limit['limit']} plans measured at each decision: same first plan {limit['same_first_plan_rate']:.3f}, same outcome {limit['same_outcome_rate']:.3f}, mission done {limit['limited_completion_rate']:.3f}.", "", f"Mission-first and fuel-first priorities went for different goals in {report['goal_split']:.3f} of episodes."]
    return "\n".join(lines) + "\n"


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the reports hold symbols a Windows console encoding cannot write
    parser = argparse.ArgumentParser(description="Evaluate goal decomposition on the scan-then-land mission.")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument("--output", default="demo/artifacts/beacon_mission_eval.json")
    parser.add_argument("--markdown-output", default="demo/artifacts/beacon_mission_eval.md")
    args = parser.parse_args()

    report = evaluate(range(args.start_seed, args.start_seed + args.seeds))
    markdown = to_markdown(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown_output).write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {output} and {args.markdown_output}")


if __name__ == "__main__":
    main()

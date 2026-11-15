"""Evaluate goal decomposition against the baseline controllers on unseen seeds.

Every controller flies one episode from the start of every seed:

- idle, Gymnasium's heuristic and the PPO pilot, as in the baseline script;
- the two fixed landing sequences, flown with the runtime checks but without
  certification or replanning;
- the goal runtime for ``land_on_pad`` under several priority profiles, which
  re-certifies the rest of the plan after every skill and, when no plan is
  admitted, may fly a fallback admitted from the current state. When it ends
  without achieving the goal and the episode goes on, the lander does nothing
  for the rest of the episode, as SubRep does without a certificate. Safety-first
  priorities are also flown under the strict policy, which never falls back,
  and under the continuation policy, and once more with revocations applied
  during the runs. These are reported apart from the criteria, which judge the
  runtime as it runs by default.

The evaluation also checks the measurements behind the certificates:

- feedback: for every plan the runtime certified, at the start of an episode or
  in mid-flight, and then flew to the end, the gains it really achieved from that
  state are compared with its certificate. A run the runtime stopped in the
  middle of a plan did not fly that plan as certified, so it is counted but not
  compared. This happens after every episode has been flown, so a revocation
  found here was not applied during the runs. A plan and start region with fewer
  than ``MIN_SAMPLES`` flights is not judged, and a criterion with nothing to
  judge reports insufficient evidence, never a pass;
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
import operator
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from decomposition.domain import Domain
from decomposition.effects import DEFAULT_TABLE_PATH, MAX_STATES_PER_SEED, EffectTable, gain_names
from decomposition.executor import execute
from decomposition.feedback import MIN_SAMPLES, REVOKED_REASON, CertificateStatus, FeedbackLog, realised_gains
from decomposition.mdn_priorities import priorities_from_weights
from decomposition.recipe_book import LUNAR_LANDER_BOOK, RecipeBook, load_recipe_book
from decomposition.runtime import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MIN_GOAL_RATE,
    MAX_DECISIONS,
    SWITCH_MARGIN,
    Continuation,
    GoalRun,
    GoalStatus,
    pursue,
)
from decomposition.verifier import Priorities
from demo.build_lunar_effect_table import build
from demo.run_lunar_baseline import CONTROLLERS, EpisodeResult, run_episode, summarise
from demo.run_lunar_subskill_reliability import SEQUENCES
from demo.train_lunar_mdn import load_landing_mdn
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import FACT_NAMES, OUTCOMES, landing_outcome, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import NO_ENGINE

GOAL = "land_on_pad"
# The table was built from seeds 0-1999. Seeds 2000-2499 and 6000-6499 were looked at
# while reviewing the plans, 3000-3199 while tuning skills, and 8000-8499, 9000-9499 and
# 10000-10499 by earlier evaluations, so these are fresh.
DEFAULT_START_SEED = 11000
DEFAULT_SEEDS = 500
TIMESTAMP = "2026-01-01T00:00:00+00:00"
# The policies for when no plan is admitted, as the reports name them; the first is the default.
POLICY_NAMES = {
    Continuation.FALLBACK: "fallback",
    Continuation.STOP: "stop",
    Continuation.KEEP_UNREJECTED_PLAN: "continuation",
}
OTHER_POLICIES = (Continuation.STOP, Continuation.KEEP_UNREJECTED_PLAN)
ABSTAINING = (GoalStatus.NO_CERTIFIED_PLAN, GoalStatus.PRIORITIES_REFUSED)

# Written before the evaluation seeds were flown.
MIN_ON_PAD = 0.95
MAX_CRASHES = 0.02
MIN_RECERTIFIED = 0.90

# How a criterion came out. With nothing to judge it on, it is neither met nor missed.
MET = "met"
MISSED = "missed"
INSUFFICIENT_EVIDENCE = "insufficient evidence"
REPORT_ONLY = "report only"
_COMPARISONS = {">=": operator.ge, "<=": operator.le, "==": operator.eq}
CERTIFICATES_NOTE = (
    "Checked after every episode was flown: the runtime did not apply these revocations during the runs. "
    f"A plan and start region with fewer than {MIN_SAMPLES} flights has too few samples and is not judged."
)

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


NOTHING_PROPOSED = "no plan proposed"


@dataclass
class Safeguards:
    """What the runtime did over many runs when no plan was admitted, and how often revocations stopped a plan.

    Every run that flew a fallback or abstained is kept whole, with each decision's
    evidence, certificate and outcome, so that it can be audited.
    """

    fallbacks: int = 0  # decisions flown as a fallback
    runs_with_fallback: int = 0
    continued: int = 0  # decisions flown under the continuation policy, without current approval
    abstained: int = 0  # runs that ended with nothing admitted, or with priorities refused
    revoked_refusals: int = 0  # decisions at which a plan was refused because the log had revoked its certificate
    fallback_outcomes: Counter = field(default_factory=Counter)  # how the runs that flew a fallback ended
    # For each reason, the abstaining runs in which some plan was refused for it at the last decision.
    abstention_reasons: Counter = field(default_factory=Counter)
    audited: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, run: GoalRun, outcome: str, seed: int) -> None:
        self.fallbacks += run.fallbacks
        self.runs_with_fallback += run.fallbacks > 0
        self.continued += run.continued
        self.revoked_refusals += sum(
            any(verdict.reason == REVOKED_REASON for verdict in decision.verdicts + decision.fallback_verdicts)
            for decision in run.decisions
        )
        if run.fallbacks:
            self.fallback_outcomes[outcome] += 1
        if run.status in ABSTAINING:
            self.abstained += 1
            self.abstention_reasons.update(abstention_reasons(run))
        if run.fallbacks or run.status in ABSTAINING:
            self.audited.append({"seed": seed, "outcome": outcome, **run.record()})

    def record(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "fallback_outcomes": dict(self.fallback_outcomes),
            "abstention_reasons": dict(self.abstention_reasons),
        }


def abstention_reasons(run: GoalRun) -> List[str]:
    """Why a run that abstained found nothing to fly: the reasons its last decision refused plans and fallbacks."""
    if run.status is GoalStatus.PRIORITIES_REFUSED:
        return [GoalStatus.PRIORITIES_REFUSED.value]
    last = run.decisions[-1]
    return sorted({verdict.reason for verdict in last.verdicts + last.fallback_verdicts}) or [NOTHING_PROPOSED]


def evaluate_profile(
    env: SubRepEnv,
    book: RecipeBook,
    table: EffectTable,
    priorities: Any,
    seeds: Sequence[int],
    continuation: Continuation = Continuation.FALLBACK,
    revoke_live: bool = False,
) -> Dict[str, Any]:
    """Pursue the goal from every seed under one source of priorities.

    With ``revoke_live``, the runtime is given the feedback log, so a certificate
    the flights so far have revoked is refused during the later runs.
    """
    episodes: List[EpisodeResult] = []
    statuses: Counter = Counter()
    first_plans: Counter = Counter()
    stops: Counter = Counter()
    replanned = rechecks = recertified = switches = predictions_checked = predictions_held = unchecked = 0
    safeguards = Safeguards()
    feedback = FeedbackLog(LUNAR_LANDER)

    records: List[Dict[str, Any]] = []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        run = pursue(
            LUNAR_LANDER,
            env,
            observation,
            GOAL,
            book,
            table,
            priorities,
            feedback=feedback if revoke_live else None,
            timestamp=TIMESTAMP,
            continuation=continuation,
        )
        episode = finish_episode(env, seed, run.rewards, run.observation, run.info, run.episode_over)
        episodes.append(episode)

        statuses[run.status.value] += 1
        replanned += run.replans > 0
        rechecks += run.rechecks
        recertified += run.recertified_rechecks
        switches += run.switches
        safeguards.add(run, episode.outcome, seed)
        first = run.decisions[0].chosen if run.decisions else None
        first_plans[first.entry.skill_id if first else "none"] += 1
        records.append(
            {
                **asdict(episode),
                "status": run.status.value,
                "plan": first.entry.skill_id if first else None,
                "replans": run.replans,
                "fallbacks": run.fallbacks,
            }
        )
        for decision in run.flown:
            stopped = decision.execution.stopped_at
            if stopped is not None:
                stops[f"{stopped.skill}: {stopped.status.value}"] += 1
            checked = [step.prediction_held for step in decision.execution.steps if step.prediction_held is not None]
            predictions_checked += len(checked)
            predictions_held += sum(checked)
        if run.flown:
            unchecked += not record_followed_plans(LUNAR_LANDER, feedback, env, seed, run)

    return {
        "summary": summarise(episodes),
        "statuses": dict(statuses),
        "first_plans": dict(first_plans),
        "episodes_with_replans": replanned,
        "rechecks": rechecks,
        "recertified_rechecks": recertified,
        "switches": switches,
        **safeguards.record(),
        "plan_stops": dict(stops),
        "predictions": {"checked": predictions_checked, "held": predictions_held},
        "feedback": feedback.report(),
        "feedback_unchecked_runs": unchecked,
        "episodes": records,
    }


def plan_flown_to_its_end(run: GoalRun) -> bool:
    """Whether the last plan of ``run`` ended as its measurements end: with the episode or with its last skill.

    A run the runtime stopped in the middle of a plan, because nothing more could
    be admitted, did not fly that plan as it was certified, so what it achieved
    says nothing about the certificate.
    """
    if not run.flown:
        return False
    last = run.flown[-1]
    return run.episode_over or (last.execution.completed and len(last.skills) == 1)


def record_followed_plans(domain: Domain, feedback: FeedbackLog, env: Any, seed: int, run: GoalRun) -> bool:
    """Record what each certified plan really achieved, for those flown to the end from where they were certified.

    Fallbacks are certified from the state they start in too, so they are recorded
    like any plan; a plan flown under the continuation policy has no certificate
    of its own and is not. Returns whether the run was recorded: a run stopped in
    the middle of its plan is not.
    """
    if not plan_flown_to_its_end(run):
        return False
    frames = 0
    followed_from = _last_change(run)
    for index, decision in enumerate(run.flown):
        if (decision.recertified or decision.fallback) and index >= followed_from:
            start = run.actions[:frames]
            feedback.record(decision.chosen, realised_gains(domain, env, seed, run.rewards[frames:], start))
        frames += decision.execution.frames
    return True


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


def check_table(
    committed: EffectTable, seeds: range, beacon: Optional[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """Measure the plans again on ``seeds``, as the committed table was built, and compare each group with it.

    ``beacon`` is the mission's beacon, or None for landing. Every gain whose mean on
    the new seeds falls below the committed lower bound is named.
    """
    unseen = build(seeds, beacon, min_samples=1)
    fresh = {(entry["name"], entry["group"]): entry for entry in unseen.data["plans"]}
    gains = gain_names(committed.data["objectives"])
    rows = []
    for entry in committed.data["plans"]:
        if entry["episodes"] < committed.min_samples or (entry["name"], entry["group"]) not in fresh:
            continue
        new = fresh[(entry["name"], entry["group"])]
        rows.append(
            {
                "plan": entry["name"],
                "group": entry["group"],
                "samples": new["samples"],
                "episodes": new["episodes"],
                "committed_mean_delta_r": entry["delta_r"]["mean"],
                "committed_lower_delta_r": entry["delta_r"]["lower"],
                "unseen_mean_delta_r": new["delta_r"]["mean"],
                "below_committed_lower": [name for name in gains if new[name]["mean"] < entry[name]["lower"]],
            }
        )
    return rows


def table_check_lines(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Markdown comparing the committed table with the same plans measured on the evaluation seeds."""
    below = sum(bool(row["below_committed_lower"]) for row in rows)
    lines = [
        "",
        "## Effect table on unseen seeds",
        "",
        f"The plans measured again on the evaluation seeds: {below} of {len(rows)} entries had a mean below the "
        "committed lower bound on some gain. An entry with few episodes here is a noisy check.",
        "",
        "| Plan | Start group | Samples (episodes) | Committed mean Δr | Committed lower Δr | Unseen mean Δr "
        "| Unseen mean below lower |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['plan']} | {row['group']} | {row['samples']} ({row['episodes']}) | "
            f"{row['committed_mean_delta_r']:.1f} | {row['committed_lower_delta_r']:.1f} | "
            f"{row['unseen_mean_delta_r']:.1f} | {', '.join(row['below_committed_lower']) or 'none'} |"
        )
    return lines


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
    compared: Sequence[str] = ("safety_first",),
) -> Dict[str, Any]:
    """Fly every controller from every seed and check the measurements behind the certificates.

    The profiles named in ``compared`` are also flown under the other policies for
    when no plan is admitted, and with revocations applied during the runs, for
    comparison only: the criteria judge the runtime as it runs by default.
    """
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
    int_seeds = [int(seed) for seed in seeds]
    try:
        for name in baselines:
            policy = CONTROLLERS[name](env)
            controllers[name] = _results([run_episode(env, policy, seed) for seed in int_seeds])
        for name, steps in SEQUENCES.items():
            episodes = [fixed_sequence_episode(env, book, steps, seed) for seed in int_seeds]
            controllers[f"fixed_{name}"] = _results(episodes)
        for name, priorities in profiles.items():
            goal_runs[name] = evaluate_profile(env, book, table, priorities, int_seeds)
            controllers[f"runtime_{name}"] = {"summary": goal_runs[name]["summary"]}
        if use_mdn:
            goal_runs["mdn"] = evaluate_profile(env, book, table, load_landing_mdn(), int_seeds)
            controllers["runtime_mdn"] = {"summary": goal_runs["mdn"]["summary"]}
        names = [name for name in compared if name in profiles]
        policies = {
            name: {
                POLICY_NAMES[policy]: evaluate_profile(env, book, table, profiles[name], int_seeds, policy)
                for policy in OTHER_POLICIES
            }
            for name in names
        }
        for name, runs in policies.items():
            for policy, run in runs.items():
                controllers[f"runtime_{name}_{policy}"] = {"summary": run["summary"]}
        live = {name: evaluate_profile(env, book, table, profiles[name], int_seeds, revoke_live=True) for name in names}
        recovery = check_recovery(env, book, table, PROFILES["safety_first"], int_seeds)
    finally:
        env.close()

    report = {
        "seeds": {"start": int(seeds[0]), "count": len(seeds)},
        "goal": GOAL,
        "settings": frozen_settings(table),
        "profiles": {name: _profile_record(p) for name, p in profiles.items()},
        "controllers": controllers,
        "goal_runs": goal_runs,
        "policies": policies,
        "live_revocation": live,
        "table_check": check_table(table, seeds),
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
    episodes = f"{summary['episodes']} episodes"
    below = [row for row in report["table_check"] if row["below_committed_lower"]]
    predictions = run["predictions"]
    return [
        criterion(
            "On the pad under safety-first priorities", summary["landed_on_pad_rate"], ">=", MIN_ON_PAD, episodes
        ),
        criterion("Crashes under safety-first priorities", summary["crashed_rate"], "<=", MAX_CRASHES, episodes),
        share_criterion(
            "Mid-flight re-checks admitted from the real state",
            run["recertified_rechecks"],
            run["rechecks"],
            MIN_RECERTIFIED,
            "mid-flight re-checks",
        ),
        share_criterion(
            "Book predictions that held", predictions["held"], predictions["checked"], 0.99, "predictions checked"
        ),
        revocation_criterion(report["goal_runs"].values()),
        reported(
            "Table entries whose unseen mean fell below the committed bound",
            len(below),
            f"{len(report['table_check'])} entries compared",
        ),
    ]


def criterion(name: str, value: Optional[float], sign: str, threshold: float, evidence: str) -> Dict[str, Any]:
    """A criterion ``value sign threshold``; with no value there was nothing to judge it on."""
    if value is None:
        result = INSUFFICIENT_EVIDENCE
    else:
        result = MET if _COMPARISONS[sign](value, threshold) else MISSED
    return {"criterion": name, "value": value, "needed": f"{sign} {threshold}", "result": result, "evidence": evidence}


def share_criterion(name: str, count: int, total: int, threshold: float, what: str) -> Dict[str, Any]:
    """A criterion on the share ``count / total``, which needs at least one of ``what``."""
    return criterion(name, count / total if total else None, ">=", threshold, f"{count} of {total} {what}")


def revocation_criterion(runs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """No certificate revoked, judged only on plan and start regions with enough flights to tell."""
    entries = [entry for run in runs for entry in run["feedback"]]
    judged = [entry for entry in entries if entry["status"] != CertificateStatus.TOO_FEW_SAMPLES.value]
    revoked = sum(entry["status"] == CertificateStatus.REVOKED.value for entry in judged)
    return criterion(
        "Certificates revoked, checked after the runs (not applied during them)",
        revoked if judged else None,
        "==",
        0,
        f"{len(judged)} of {len(entries)} plan and start regions had at least {MIN_SAMPLES} flights",
    )


def reported(name: str, value: float, evidence: str) -> Dict[str, Any]:
    """A number the evaluation reports without judging it."""
    return {"criterion": name, "value": value, "needed": REPORT_ONLY, "result": REPORT_ONLY, "evidence": evidence}


def criteria_table(criteria: Sequence[Mapping[str, Any]]) -> List[str]:
    """The criteria as markdown table lines."""
    rows = (
        f"| {c['criterion']} | {_value(c['value'])} | {c['needed']} | {c['result']} | {c['evidence']} |"
        for c in criteria
    )
    return ["| Criterion | Value | Needed | Result | Evidence |", "|---|---|---|---|---|", *rows]


def shortfalls(entry: Mapping[str, Any]) -> str:
    """The gains of a feedback entry whose realised upper bound fell below the certified value."""
    gains = {name: gain for name, gain in entry.items() if isinstance(gain, Mapping)}
    short = [
        name
        for name, gain in gains.items()
        if gain["realised_upper"] is not None and gain["realised_upper"] < gain["certified"]
    ]
    return ", ".join(short) or "-"


def unchecked_note(goal_runs: Mapping[str, Mapping[str, Any]]) -> str:
    """How many runs of each profile stopped in the middle of a plan, so were compared with no certificate."""
    counts = ", ".join(f"{name} {run['feedback_unchecked_runs']}" for name, run in goal_runs.items())
    return f"Runs stopped in the middle of a plan, so compared with no certificate: {counts}."


def policy_lines(report: Mapping[str, Any], success: str, label: str) -> List[str]:
    """Markdown comparing the policies for when no plan is admitted, for the priorities flown under each."""
    policies = report.get("policies") or {}
    if not policies:
        return []
    lines = [
        "",
        "## When no plan is admitted",
        "",
        "`fallback` is the default, and the one the criteria judge: it may fly a fallback admitted from the current "
        "state under the current priorities, no less safe than doing nothing, and reaching the goal less often than "
        "a new plan must; of those, the one that reaches the goal most often. `stop` flies only plans admitted from "
        "the current state. `continuation` "
        "(`Continuation.KEEP_UNREJECTED_PLAN`) lets the plan in progress go on without a new certificate. A run "
        "abstains when it ends with nothing admitted, and the lander then does nothing.",
        "",
        f"| Priorities | Policy | {label} | Crashed | Abstained | Fallback decisions (runs) | "
        "Decisions without current approval | Statuses |",
        "|---|---|---|---|---|---|---|---|",
    ]
    reasons = []
    for name, others in policies.items():
        for policy, run in ((POLICY_NAMES[Continuation.FALLBACK], report["goal_runs"][name]), *others.items()):
            summary = run["summary"]
            lines.append(
                f"| {name} | {policy} | {summary[success]:.3f} | {summary['crashed_rate']:.3f} | {run['abstained']} | "
                f"{run['fallbacks']} ({run['runs_with_fallback']}) | {run['continued']} | {_counts(run['statuses'])} |"
            )
            reasons.append(f"- {name}, {policy}: {_counts(run['abstention_reasons']) or 'none abstained'}")
    lines += [
        "",
        "Why runs abstained: for each reason, the runs in which a plan or fallback was refused for it at the last "
        "decision. A run counts once under every reason it met.",
        "",
        *reasons,
    ]
    return lines


def reliability_lines(goal_runs: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Markdown counting, for each set of priorities, how the certificates its runs relied on stood up."""
    lines = [
        "",
        "## Certificate reliability",
        "",
        f"Plan and start regions whose certificates held, were revoked, or had fewer than {MIN_SAMPLES} flights to "
        "judge them, and the runs stopped in the middle of a plan, which were not compared.",
        "",
        "| Priorities | Held | Revoked | Too few flights | Runs not compared |",
        "|---|---|---|---|---|",
    ]
    for name, run in goal_runs.items():
        statuses = Counter(entry["status"] for entry in run["feedback"])
        counts = " | ".join(str(statuses[status.value]) for status in CertificateStatus)
        lines.append(f"| {name} | {counts} | {run['feedback_unchecked_runs']} |")
    return lines


def live_revocation_lines(report: Mapping[str, Any], success: str, label: str) -> List[str]:
    """Markdown for the runs flown with revocations applied while they were flown."""
    live = report.get("live_revocation") or {}
    if not live:
        return []
    lines = [
        "",
        "## Revocations applied during the runs",
        "",
        "Apart from the benchmark: the same priorities, with the runtime given the feedback log. A certificate the "
        "flights so far have revoked is refused at every later decision, fallbacks included, until a certificate "
        "the flights support replaces it.",
        "",
        f"| Priorities | {label} | Crashed | Abstained | Decisions refusing a revoked plan | Revoked at the end | "
        "Statuses |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, run in live.items():
        summary = run["summary"]
        revoked = sum(entry["status"] == CertificateStatus.REVOKED.value for entry in run["feedback"])
        lines.append(
            f"| {name} | {summary[success]:.3f} | {summary['crashed_rate']:.3f} | {run['abstained']} | "
            f"{run['revoked_refusals']} | {revoked} | {_counts(run['statuses'])} |"
        )
    return lines


def _value(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return str(value) if isinstance(value, int) else f"{value:.3f}"


def _results(episodes: Sequence[EpisodeResult]) -> Dict[str, Any]:
    return {"summary": summarise(episodes), "episodes": [asdict(episode) for episode in episodes]}


def frozen_settings(table: EffectTable) -> Dict[str, Any]:
    """The settings an evaluation ran with, so that a later run can be checked against them."""
    return {
        "default_policy": POLICY_NAMES[Continuation.FALLBACK],
        "min_goal_rate": DEFAULT_MIN_GOAL_RATE,
        "switch_margin": SWITCH_MARGIN,
        "max_attempts": DEFAULT_MAX_ATTEMPTS,
        "max_decisions": MAX_DECISIONS,
        "feedback_min_samples": MIN_SAMPLES,
        "table": {
            key: table.data[key]
            for key in ("schema_version", "environment", "seeds", "gamma", "confidence", "min_samples", "support_margin")
        },
        "max_states_per_seed": MAX_STATES_PER_SEED,
    }


def settings_line(settings: Mapping[str, Any]) -> str:
    """The frozen settings as one markdown sentence."""
    table = settings["table"]
    return (
        f"Settings: `{settings['default_policy']}` policy, goal rate at least {settings['min_goal_rate']}, "
        f"switch margin {settings['switch_margin']}, table {table['schema_version']} from seeds "
        f"{table['seeds']['start']}-{table['seeds']['stop'] - 1} (at least {table['min_samples']} episodes per entry, "
        f"{settings['max_states_per_seed']} states per episode at most)."
    )


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
        settings_line(report["settings"]),
        "",
        "## Criteria written before the evaluation",
        "",
        *criteria_table(report["criteria"]),
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
        "Episodes with replans | Predictions held | Plan stops | Fallback decisions (runs) | Runs with a fallback ended |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in report["goal_runs"].items():
        predictions = run["predictions"]
        lines.append(
            f"| {name} | {_counts(run['statuses'])} | {_counts(run['first_plans'])} | "
            f"{run['rechecks']} | {run['recertified_rechecks']} | {run['switches']} | "
            f"{run['episodes_with_replans']} | {predictions['held']}/{predictions['checked']} | "
            f"{_counts(run['plan_stops']) or '-'} | {run['fallbacks']} ({run['runs_with_fallback']}) | "
            f"{_counts(run['fallback_outcomes']) or '-'} |"
        )
    lines += policy_lines(report, "landed_on_pad_rate", "On pad")
    lines += reliability_lines(report["goal_runs"])
    lines += live_revocation_lines(report, "landed_on_pad_rate", "On pad")

    lines += [
        "",
        "## Certificates against realised gains",
        "",
        CERTIFICATES_NOTE,
        "",
        "| Profile | Plan | Start region | Samples | Certified Δr | Realised mean Δr | Status | Fell short on |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, run in report["goal_runs"].items():
        for entry in run["feedback"]:
            lines.append(
                f"| {name} | {entry['plan']} | {entry['region']} | {entry['samples']} | "
                f"{entry['delta_r']['certified']:.1f} | {entry['delta_r']['realised_mean']:.1f} | {entry['status']} | "
                f"{shortfalls(entry)} |"
            )
    lines += ["", unchecked_note(report["goal_runs"])]
    lines += table_check_lines(report["table_check"])

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

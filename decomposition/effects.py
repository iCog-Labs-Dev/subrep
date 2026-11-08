"""Measured effects of a domain's skills and of plans built from them.

The effect of running something from a state is its payoff gain Δr and its
motive gains Δn, one per objective of the domain, over doing nothing for the
same number of frames from the same state. Box2D state cannot be copied, but
resetting the environment with a seed and replaying the same actions reproduces
a state exactly, so both runs start from identical states (a paired comparison).

Totals follow ``env.skill_executor.SkillExecutor`` (discount 0.99, payoff is
the sum of the reward vector) and the gains come from
``baseline.improvement_calculator.ImprovementCalculator``.

A plan is an ordered list of sub-skills flown from a state reached by replaying
actions from the start of an episode. If the plan leaves the lander standing on
its legs, its last sub-skill keeps running until the episode ends, so the plan's
effect includes the landing reward. If a step of the plan fails, the lander does
nothing for the rest of the episode, as it would without a plan, so a failed
flight is charged for what follows. Each step of the flight is also measured from
the state where it started; a plan's interaction is its effect minus the sum of
its steps' effects. Every plan flight also records how the episode really ended
(on the pad, off the pad, crashed, timed out, or still in flight), because a step
that completes, such as legs touching down, is not a landing until the final
result confirms it.

The table measures every plan the planner proposes at the states where the
runtime decides: the start of the episode and the state after each skill of each
plan flown from it. The rest of a plan can therefore be certified from the real
state in mid-flight, from measurements on other seeds. Replaying the seed being
flown would answer more exactly, but it would show the lander its own future,
which nothing outside a simulator can do. A plan the lander could switch to in
mid-flight is measured too, but the states along it are not visited in turn: the
number of such paths grows with every skill, and a plan flown past the measured
states goes on under the certificate it already holds.

Samples are grouped by the facts of the state they start from and summarised
by their mean, spread, count and a one-sided bootstrap lower confidence bound.
Each group also records its support: the lowest and highest value of every
observation number among the states it was measured from. States that share
facts can still differ a lot: a lander at the start of an episode and one
falling fast halfway down can both be not upright, not over the pad and not
near the ground. A lookup therefore answers only for states inside the support,
widened by a small margin, and treats every other state as unmeasured.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from decomposition.domain import GAMMA, Domain

SCHEMA_VERSION = "subrep.effects.v3"
BASELINE_ID = "idle_policy_v1"
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000
MIN_SAMPLES = 30
MAX_STATES_PER_SEED = 60  # bounds the states measured per seed when plans keep branching
# Fraction of each observation range added on both sides of a group's support.
SUPPORT_MARGIN = 0.05
DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "data" / "lunar_effects.json"
LUNAR_BEACON_TABLE = Path(__file__).resolve().parent / "data" / "lunar_beacon_effects.json"

IN_FLIGHT = "in_flight"

_DECIMALS = 6

T = TypeVar("T")
# (facts of a state, whether it is the start of an episode) -> the skill lists to measure from it
PlansAt = Callable[[Mapping[str, bool], bool], Sequence[Sequence[str]]]


# ---------------------------------------------------------------------------
# Paired measurement
# ---------------------------------------------------------------------------


def gain_names(objectives: Sequence[str]) -> Tuple[str, ...]:
    """The names of a gains vector, for example ``("delta_r", "delta_safety", "delta_fuel")``."""
    return ("delta_r", *(f"delta_{name}" for name in objectives))


def replay(env: Any, seed: int, actions: Sequence[int]) -> np.ndarray:
    """Reset ``env`` with ``seed``, replay ``actions`` and return the observation reached."""
    obs, _ = env.reset(seed=seed)
    for action in actions:
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            raise ValueError(f"the episode of seed {seed} ended before all actions were replayed")
    return np.asarray(obs, dtype=np.float32)


def idle_rewards(domain: Domain, env: Any, frames: int) -> np.ndarray:
    """Rewards of doing nothing for ``frames`` frames, or until the episode ends."""
    rewards = []
    for _ in range(frames):
        _, reward, terminated, truncated, _ = env.step(domain.idle_action)
        rewards.append(np.asarray(reward, dtype=np.float64))
        if terminated or truncated:
            break
    return np.asarray(rewards, dtype=np.float64).reshape(-1, domain.num_objectives)


def discounted_totals(rewards: np.ndarray, gamma: float = GAMMA) -> Tuple[float, np.ndarray]:
    """Discounted payoff and motive totals of a ``(frames, objectives)`` reward array."""
    rewards = np.asarray(rewards, dtype=np.float64)
    discounts = gamma ** np.arange(len(rewards))
    motives = discounts @ rewards
    return float(motives.sum()), motives


def improvement(
    rewards: np.ndarray, baseline_rewards: np.ndarray, gamma: float = GAMMA
) -> Tuple[float, Tuple[float, ...]]:
    """Payoff gain Δr and motive gains Δn of ``rewards`` over ``baseline_rewards``."""
    payoff, motives = discounted_totals(rewards, gamma)
    baseline_payoff, baseline_motives = discounted_totals(baseline_rewards, gamma)
    calculator = ImprovementCalculator({"baseline_payoff": baseline_payoff, "baseline_motives": baseline_motives})
    delta_r, delta_n = calculator.compute_improvements(payoff, motives)
    return float(delta_r), tuple(float(value) for value in delta_n)


def group_key(domain: Domain, facts: Mapping[str, bool]) -> str:
    """The group of a state, for example ``"upright=1,over_pad=0,near_ground=0"``."""
    return ",".join(f"{name}={int(facts[name])}" for name in domain.group_facts)


def plan_key(skills: Sequence[str]) -> str:
    """The name of a plan, for example ``"stabilise>move_over_pad"``."""
    return ">".join(skills)


@dataclass(frozen=True)
class Sample:
    """One paired measurement of a sub-skill or a plan from one state."""

    name: str
    group: str
    delta_r: float
    delta_n: Tuple[float, ...]
    frames: int
    completed: bool
    start: Tuple[float, ...]  # the observation the measurement started from

    @property
    def gains(self) -> np.ndarray:
        """Δr followed by Δn, as one array."""
        return np.array([self.delta_r, *self.delta_n])


@dataclass(frozen=True)
class PlanFlight:
    """One plan flown from one state of one episode, with each step measured too."""

    seed: int
    skills: Tuple[str, ...]
    plan: Sample
    steps: Tuple[Sample, ...]  # the steps that ran; fewer than ``skills`` if one failed
    outcome: str  # how the episode ended, from its final result, or IN_FLIGHT
    prefix: Tuple[int, ...] = ()  # the actions replayed from the start of the episode to reach the state
    next_states: Tuple[Tuple[int, ...], ...] = ()  # replayable action lists of the states after each step

    @property
    def interaction(self) -> np.ndarray:
        """The plan's gains minus the sum over its steps."""
        return self.plan.gains - sum((step.gains for step in self.steps), np.zeros_like(self.plan.gains))


def _sample(
    domain: Domain,
    name: str,
    start: np.ndarray,
    delta_r: float,
    delta_n: Tuple[float, ...],
    frames: int,
    completed: bool,
) -> Sample:
    group = group_key(domain, domain.facts(start))
    observation = tuple(float(value) for value in np.asarray(start, dtype=np.float64).reshape(-1))
    return Sample(name, group, delta_r, delta_n, frames, completed, observation)


def fly_plan(
    domain: Domain, env: Any, seed: int, skills: Sequence[str], prefix: Sequence[int] = ()
) -> PlanFlight:
    """Fly ``skills`` in order from the state ``prefix`` reaches in episode ``seed``, and measure the plan.

    The steps stop at the first one that does not complete or when the episode
    ends. Each step is measured too when the flight starts at the start of the
    episode, which is enough to estimate what a step alone gains and costs far
    less than doing it from every state.
    """
    if not skills:
        raise ValueError("a plan needs at least one sub-skill")

    prefix = tuple(prefix)
    obs = replay(env, seed, prefix)
    plan_start = obs
    actions: List[int] = []
    rewards: List[np.ndarray] = []
    runs = []  # (frames flown before the step, observation the step started from, its run)
    info: Mapping[str, object] = {}
    ended = False
    next_states: List[Tuple[int, ...]] = []
    for position, name in enumerate(skills, start=1):
        run = domain.run_skill(env, domain.skills[name], obs)
        runs.append((len(actions), obs, run))
        actions.extend(run.actions)
        rewards.extend(run.rewards)
        obs = run.observation
        info = run.info if run.info is not None else info
        ended = run.terminated or run.truncated
        if not ended and (position < len(skills) or not run.completed):  # nothing is decided once a plan has finished
            next_states.append(prefix + tuple(actions))
        if ended or not run.completed:
            break

    completed = len(runs) == len(skills) and runs[-1][2].completed
    if not completed and not ended:
        while not ended:
            obs, reward, terminated, truncated, info = env.step(domain.idle_action)
            rewards.append(np.asarray(reward, dtype=np.float64))
            ended = terminated or truncated
    if completed and not ended and domain.facts(obs)[domain.settling_fact]:
        last = domain.skills[skills[-1]]
        while not ended:
            obs, reward, terminated, truncated, info = env.step(int(last.policy(obs)))
            rewards.append(np.asarray(reward, dtype=np.float64))
            ended = terminated or truncated
    outcome = domain.outcome(info, obs) if ended else IN_FLIGHT

    plan_rewards = np.asarray(rewards, dtype=np.float64).reshape(-1, domain.num_objectives)
    replay(env, seed, prefix)
    delta_r, delta_n = improvement(plan_rewards, idle_rewards(domain, env, len(plan_rewards)), domain.gamma)
    plan = _sample(domain, plan_key(skills), plan_start, delta_r, delta_n, len(plan_rewards), completed)

    steps = []
    for frames_before, start, run in runs if not prefix else ():
        replay(env, seed, tuple(actions[:frames_before]))
        delta_r, delta_n = improvement(run.rewards, idle_rewards(domain, env, run.frames), domain.gamma)
        steps.append(_sample(domain, run.skill, start, delta_r, delta_n, run.frames, run.completed))

    return PlanFlight(
        seed=seed,
        skills=tuple(skills),
        plan=plan,
        steps=tuple(steps),
        outcome=outcome,
        prefix=prefix,
        next_states=tuple(next_states),
    )


def measure_seed(
    domain: Domain,
    env: Any,
    seed: int,
    plans_at: PlansAt,
    max_states: int = MAX_STATES_PER_SEED,
) -> List[PlanFlight]:
    """Fly every plan proposed at the start of episode ``seed`` and at the states where the runtime decides along them."""
    flights = [
        fly_plan(domain, env, seed, skills) for skills in plans_at(domain.facts(replay(env, seed, ())), True)
    ]
    seen = {()}
    for flight in tuple(flights):
        for state in flight.next_states:
            if state in seen or len(seen) >= max_states:
                continue
            seen.add(state)
            for skills in plans_at(domain.facts(replay(env, seed, state)), False):
                flights.append(fly_plan(domain, env, seed, skills, state))
    return flights


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def lower_bound(values: Sequence[float], seed: int) -> float:
    """One-sided bootstrap lower confidence bound of the mean of ``values``."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot bound the mean of no values")
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(BOOTSTRAP_RESAMPLES, data.size))].mean(axis=1)
    return float(np.quantile(means, 1.0 - CONFIDENCE))


def _bootstrap_seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8"))


def summarise_gains(gains: np.ndarray, objectives: Sequence[str], seed: int) -> Dict[str, Dict[str, float]]:
    """Mean, spread and lower bound of each column of a ``(samples, 1 + objectives)`` gains array."""
    summary = {}
    for name, column in zip(gain_names(objectives), np.asarray(gains, dtype=np.float64).T):
        summary[name] = {
            "mean": round(float(column.mean()), _DECIMALS),
            "std": round(float(column.std()), _DECIMALS),
            "lower": round(lower_bound(column, seed), _DECIMALS),
        }
    return summary


def summarise(samples: Sequence[Sample], objectives: Sequence[str]) -> Dict[str, object]:
    """Summarise samples that share a name and a group."""
    name, group = samples[0].name, samples[0].group
    if any(s.name != name or s.group != group for s in samples):
        raise ValueError("samples to summarise must share one name and one group")
    return {
        "name": name,
        "group": group,
        "samples": len(samples),
        "completion_rate": round(float(np.mean([s.completed for s in samples])), _DECIMALS),
        "mean_frames": round(float(np.mean([s.frames for s in samples])), _DECIMALS),
        "support": _support([s.start for s in samples]),
        **summarise_gains(np.stack([s.gains for s in samples]), objectives, _bootstrap_seed(name, group)),
    }


def _support(starts: Sequence[Tuple[float, ...]]) -> Dict[str, List[float]]:
    """The lowest and highest value of each observation number, rounded outwards."""
    observations = np.asarray(starts, dtype=np.float64)
    scale = 10.0**_DECIMALS
    return {
        "low": [float(v) for v in np.floor(observations.min(axis=0) * scale) / scale],
        "high": [float(v) for v in np.ceil(observations.max(axis=0) * scale) / scale],
    }


def _by_name_and_group(items: Iterable[T], sample_of: Callable[[T], Sample]) -> List[List[T]]:
    grouped: Dict[Tuple[str, str], List[T]] = {}
    for item in items:
        sample = sample_of(item)
        grouped.setdefault((sample.name, sample.group), []).append(item)
    return [grouped[key] for key in sorted(grouped)]


@dataclass(frozen=True)
class SeedMeasurement:
    """What one seed contributes to a table: small enough to pass between processes."""

    flights: Tuple[PlanFlight, ...]  # without their action lists, which only served to find the states
    steps: Tuple[Sample, ...]  # each step counted once


def measure_seeds(
    domain: Domain,
    seeds: Iterable[int],
    plans_at: PlansAt,
    max_states: int = MAX_STATES_PER_SEED,
) -> List[SeedMeasurement]:
    """Measure every plan at every decision state of ``seeds``.

    A step that starts after the same sub-skills in two plans is flown from the
    same state, so it is counted once per seed.
    """
    measurements: List[SeedMeasurement] = []
    env = domain.make_env(0)
    try:
        for seed in seeds:
            flights = measure_seed(domain, env, seed, plans_at, max_states)
            steps: Dict[Tuple[Tuple[int, ...], Tuple[str, ...]], Sample] = {}
            for flight in flights:
                for index, step in enumerate(flight.steps):
                    steps.setdefault((flight.prefix, flight.skills[: index + 1]), step)
            compact = tuple(replace(flight, prefix=(), next_states=()) for flight in flights)
            measurements.append(SeedMeasurement(compact, tuple(steps.values())))
    finally:
        env.close()
    return measurements


def build_table(
    domain: Domain,
    seeds: range,
    plans_at: PlansAt,
    min_samples: int = MIN_SAMPLES,
    max_states: int = MAX_STATES_PER_SEED,
) -> "EffectTable":
    """Fly the plans ``plans_at`` gives at every decision state of every seed and summarise the measured effects."""
    return table_from_measurements(domain, seeds, measure_seeds(domain, seeds, plans_at, max_states), min_samples)


def table_from_measurements(
    domain: Domain, seeds: range, measurements: Sequence[SeedMeasurement], min_samples: int = MIN_SAMPLES
) -> "EffectTable":
    """Summarise measurements of ``seeds``.

    Entries with fewer than ``min_samples`` samples could never answer a lookup,
    so they are left out.
    """
    flights = [flight for measurement in measurements for flight in measurement.flights]
    step_samples = [step for measurement in measurements for step in measurement.steps]

    objectives = domain.objectives
    steps = [
        summarise(group, objectives)
        for group in _by_name_and_group(step_samples, lambda s: s)
        if len(group) >= min_samples
    ]
    plans_summary = []
    for group in _by_name_and_group(flights, lambda f: f.plan):
        if len(group) < min_samples:
            continue
        entry = summarise([flight.plan for flight in group], objectives)
        entry["skills"] = list(group[0].skills)
        entry["outcomes"] = {
            name: round(sum(flight.outcome == name for flight in group) / len(group), _DECIMALS)
            for name in (*domain.outcomes, IN_FLIGHT)
        }
        interactions = [flight.interaction for flight in group if flight.steps]
        if interactions:
            entry["interaction"] = summarise_gains(
                np.stack(interactions), objectives, _bootstrap_seed(entry["name"], entry["group"], "interaction")
            )
        plans_summary.append(entry)

    return EffectTable(
        {
            "schema_version": SCHEMA_VERSION,
            "environment": domain.environment,
            "objectives": list(objectives),
            "baseline_id": BASELINE_ID,
            "gamma": domain.gamma,
            "confidence": CONFIDENCE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "min_samples": min_samples,
            "support_margin": SUPPORT_MARGIN,
            "group_facts": list(domain.group_facts),
            "seeds": {"start": seeds.start, "stop": seeds.stop},
            "steps": steps,
            "plans": plans_summary,
        },
        domain,
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gains:
    """A payoff gain Δr with one motive gain per objective in Δn."""

    delta_r: float
    delta_n: Tuple[float, ...]


@dataclass(frozen=True)
class Estimate:
    """The measured effect of a sub-skill or a plan from one group of states."""

    samples: int
    completion_rate: float
    mean_frames: float
    mean: Gains
    lower: Gains  # one-sided lower confidence bounds
    interaction: Optional[Gains] = None  # mean interaction; plans flown from the start of an episode only
    outcomes: Optional[Dict[str, float]] = None  # how plan flights really ended; plans only


def _gains(stats: Mapping[str, Mapping[str, float]], objectives: Sequence[str], field: str) -> Gains:
    return Gains(
        delta_r=float(stats["delta_r"][field]),
        delta_n=tuple(float(stats[f"delta_{name}"][field]) for name in objectives),
    )


class EffectTable:
    """Measured effects of sub-skills and plans, looked up by the observation of a start state.

    A lookup finds the group of the observation's facts and answers only if
    the group has at least ``min_samples`` samples and the observation lies
    inside the group's support, widened on each side by ``support_margin``
    times its range. Every other state is treated as unmeasured.
    """

    def __init__(self, data: Mapping[str, object], domain: Domain) -> None:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"expected schema_version {SCHEMA_VERSION!r}, got {data.get('schema_version')!r}")
        if data["environment"] != domain.environment:
            raise ValueError(f"table measures {data['environment']!r}, expected {domain.environment!r}")
        if tuple(data["group_facts"]) != domain.group_facts:
            raise ValueError(f"table groups by {data['group_facts']}, expected {list(domain.group_facts)}")
        if tuple(data["objectives"]) != domain.objectives:
            raise ValueError(f"table measures objectives {data['objectives']}, expected {list(domain.objectives)}")
        if data["gamma"] != domain.gamma:
            raise ValueError(f"table discounts by {data['gamma']}, expected {domain.gamma}")
        self.domain = domain
        self.data = dict(data)
        self.min_samples = int(data["min_samples"])
        self.support_margin = float(data["support_margin"])
        self._steps = {(e["name"], e["group"]): e for e in data["steps"]}
        self._plans = {(e["name"], e["group"]): e for e in data["plans"]}

    @classmethod
    def load(cls, path: Union[str, Path], domain: Domain) -> "EffectTable":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), domain)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

    def step(self, skill: str, observation: np.ndarray) -> Optional[Estimate]:
        """The effect of ``skill`` from ``observation``, or None if no measurement applies there."""
        return self._estimate(self._steps.get((skill, self._group_of(observation))), observation)

    def plan(self, skills: Sequence[str], observation: np.ndarray) -> Optional[Estimate]:
        """The effect of the plan ``skills`` from ``observation``, or None if no measurement applies there."""
        return self._estimate(self._plans.get((plan_key(skills), self._group_of(observation))), observation)

    def rough_value(self, skills: Sequence[str], weights: Sequence[float]) -> float:
        """A rough worth of a plan under ``weights``: its mean gains over all states it was measured from.

        A plan never measured counts as its steps added up, and a step never measured
        as nothing. This can only order plans to measure. It never certifies one.
        """
        name = plan_key(skills)
        measured = [entry for (plan, _), entry in self._plans.items() if plan == name]
        if measured:
            return self._mean_value(measured, weights)
        return float(
            sum(
                self._mean_value([entry for (step, _), entry in self._steps.items() if step == skill], weights)
                for skill in skills
            )
        )

    def _mean_value(self, entries: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> float:
        if not entries:
            return 0.0
        samples = sum(int(entry["samples"]) for entry in entries)
        gains = sum(
            int(entry["samples"])
            * (
                float(entry["delta_r"]["mean"])
                + sum(
                    weight * float(entry[f"delta_{name}"]["mean"])
                    for weight, name in zip(weights, self.domain.objectives)
                )
            )
            for entry in entries
        )
        return gains / samples

    def supports(self, entry: Mapping[str, object], observation: np.ndarray) -> bool:
        """Whether ``observation`` lies inside the entry's support, widened by the margin."""
        low = np.asarray(entry["support"]["low"], dtype=np.float64)
        high = np.asarray(entry["support"]["high"], dtype=np.float64)
        widen = self.support_margin * (high - low)
        values = np.asarray(observation, dtype=np.float64).reshape(-1)
        return bool(values.shape == low.shape and np.all((values >= low - widen) & (values <= high + widen)))

    def _group_of(self, observation: np.ndarray) -> str:
        return group_key(self.domain, self.domain.facts(observation))

    def _estimate(self, entry: Optional[Mapping[str, object]], observation: np.ndarray) -> Optional[Estimate]:
        if entry is None or int(entry["samples"]) < self.min_samples or not self.supports(entry, observation):
            return None
        objectives = self.domain.objectives
        return Estimate(
            samples=int(entry["samples"]),
            completion_rate=float(entry["completion_rate"]),
            mean_frames=float(entry["mean_frames"]),
            mean=_gains(entry, objectives, "mean"),
            lower=_gains(entry, objectives, "lower"),
            interaction=_gains(entry["interaction"], objectives, "mean") if "interaction" in entry else None,
            outcomes=dict(entry["outcomes"]) if "outcomes" in entry else None,
        )

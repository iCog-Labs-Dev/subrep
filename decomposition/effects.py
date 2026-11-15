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

The table measures every plan the planner proposes at every state where the
runtime could decide: the start of the episode and, breadth first, the state
after each skill of every plan flown, so the states along a plan the lander
switches to in mid-flight are measured as well as those along the plans it
starts with. ``MAX_STATES_PER_SEED`` bounds how many states one episode adds.
The rest of a plan can therefore be certified from the real state in mid-flight,
from measurements on other seeds. Replaying the seed being flown would answer
more exactly, but it would show the lander its own future, which nothing outside
a simulator can do.

A state is an observation followed by how many frames of the episode have
already been flown. The episode has a time limit the observation does not show,
so the same position late in an episode leaves less time to finish a plan: after
the beacon scan, the plan to the pad completes far less often from a state reached
late, because the episode runs out first. Leaving the time out would average early
states with late ones and misjudge both.

Samples are grouped by the facts of the state they start from and summarised
by their mean, spread, count and a one-sided bootstrap lower confidence bound.
One episode can add several samples to a group, from different states, and they
share its start and its terrain, so the bootstrap resamples whole episodes and
each entry records how many episodes it holds; an entry needs ``MIN_SAMPLES``
episodes to answer. Each group also records its support: the lowest and highest
value of every number of the states it was measured from, the frames already
flown included. States that share facts can still differ a lot: a lander at the
start of an episode and one falling fast halfway down can both be not upright,
not over the pad and not near the ground. A lookup therefore answers only for
states inside the support, widened by a small margin, and treats every other
state as unmeasured.

A group can still mix states that fare differently: the states a plan usually
starts from and rarer ones reached after a switch of plans in mid-flight. A
group's averages would then understate what a plan achieves from the usual states
and overstate it from the rare ones. Each plan's group is therefore split into
cells of similar states, by k-means over the states scaled by their spread in
the group, with about ``cell_episodes`` episodes per cell and at least
``MIN_SAMPLES`` in each; a cell with fewer is merged into its neighbours. The
cells are found from the states alone, never from how the flights from them
went, so an estimate from a cell is not biased towards cells where flights
happened to go well. A lookup answers from the cell nearest the state, and names
it as the region the estimate holds for. The support still decides whether a
state is measured at all.
"""

from __future__ import annotations

import json
import zlib
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from decomposition.domain import GAMMA, Domain

SCHEMA_VERSION = "subrep.effects.v6"
BASELINE_ID = "idle_policy_v1"
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000
MIN_SAMPLES = 30
MAX_STATES_PER_SEED = 60  # bounds the states measured per seed when plans keep branching
CELL_EPISODES: Optional[int] = 200  # episodes per cell a group is split into; None keeps each group whole
KMEANS_ITERATIONS = 100
# Fraction of the range of each number of the states added on both sides of a group's support.
SUPPORT_MARGIN = 0.05
DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "data" / "lunar_effects.json"
LUNAR_BEACON_TABLE = Path(__file__).resolve().parent / "data" / "lunar_beacon_effects.json"

IN_FLIGHT = "in_flight"

DECIMALS = 6  # every value in a table is stored rounded to this many decimals

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


def state(observation: np.ndarray, elapsed: float) -> np.ndarray:
    """A state as the table measures it: the observation followed by the frames of the episode already flown."""
    if elapsed < 0:
        raise ValueError(f"the frames already flown cannot be negative, got {elapsed}")
    return np.append(np.asarray(observation, dtype=np.float64).reshape(-1), float(elapsed))


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
    seed: int  # the episode it was measured in
    elapsed: int  # frames of the episode flown before ``start``

    @property
    def gains(self) -> np.ndarray:
        """Δr followed by Δn, as one array."""
        return np.array([self.delta_r, *self.delta_n])

    @property
    def state(self) -> np.ndarray:
        """The state the measurement started from, as ``state`` builds it."""
        return state(self.start, self.elapsed)


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
    seed: int,
    elapsed: int,
    name: str,
    start: np.ndarray,
    delta_r: float,
    delta_n: Tuple[float, ...],
    frames: int,
    completed: bool,
) -> Sample:
    group = group_key(domain, domain.facts(start))
    observation = tuple(float(value) for value in np.asarray(start, dtype=np.float64).reshape(-1))
    return Sample(name, group, delta_r, delta_n, frames, completed, observation, seed, elapsed)


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
    plan = _sample(
        domain, seed, len(prefix), plan_key(skills), plan_start, delta_r, delta_n, len(plan_rewards), completed
    )

    steps = []
    for frames_before, start, run in runs if not prefix else ():
        replay(env, seed, tuple(actions[:frames_before]))
        delta_r, delta_n = improvement(run.rewards, idle_rewards(domain, env, run.frames), domain.gamma)
        sample = _sample(domain, seed, frames_before, run.skill, start, delta_r, delta_n, run.frames, run.completed)
        steps.append(sample)

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
    """Fly every plan proposed at every state of episode ``seed`` where the runtime could decide.

    The states are the start of the episode and then, breadth first, the state
    after each skill of every plan flown, until ``max_states`` states, the start
    included, have been measured.
    """
    flights: List[PlanFlight] = []
    seen = set()
    waiting = deque([()])  # states as the actions that reach them from the start
    while waiting and len(seen) < max_states:
        state = waiting.popleft()
        if state in seen:
            continue
        seen.add(state)
        for skills in plans_at(domain.facts(replay(env, seed, state)), not state):
            flight = fly_plan(domain, env, seed, skills, state)
            flights.append(flight)
            waiting.extend(flight.next_states)
    return flights


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def lower_bound(values: Sequence[float], seed: int, episodes: Optional[Sequence[int]] = None) -> float:
    """One-sided bootstrap lower confidence bound of the mean of ``values``.

    ``episodes`` names the episode each value was measured in. Values from one
    episode are not independent, so whole episodes are resampled; resampling
    the values one by one would claim more certainty than they give.
    """
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot bound the mean of no values")
    rng = np.random.default_rng(seed)
    if episodes is None:
        means = data[rng.integers(0, data.size, size=(BOOTSTRAP_RESAMPLES, data.size))].mean(axis=1)
    else:
        _, episode = np.unique(np.asarray(episodes), return_inverse=True)
        if episode.size != data.size:
            raise ValueError("every value needs the episode it was measured in")
        sums, counts = np.bincount(episode, weights=data), np.bincount(episode).astype(np.float64)
        picks = rng.integers(0, sums.size, size=(BOOTSTRAP_RESAMPLES, sums.size))
        means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return float(np.quantile(means, 1.0 - CONFIDENCE))


def _bootstrap_seed(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8"))


def summarise_gains(
    gains: np.ndarray, objectives: Sequence[str], seed: int, episodes: Optional[Sequence[int]] = None
) -> Dict[str, Dict[str, float]]:
    """Mean, spread and lower bound of each column of a ``(samples, 1 + objectives)`` gains array.

    ``episodes`` names the episode of each row, so the bound resamples whole episodes.
    """
    summary = {}
    for name, column in zip(gain_names(objectives), np.asarray(gains, dtype=np.float64).T):
        summary[name] = {
            "mean": round(float(column.mean()), DECIMALS),
            "std": round(float(column.std()), DECIMALS),
            "lower": round(lower_bound(column, seed, episodes), DECIMALS),
        }
    return summary


def episode_count(samples: Iterable[Sample]) -> int:
    """How many distinct episodes ``samples`` were measured in."""
    return len({sample.seed for sample in samples})


def summarise(samples: Sequence[Sample], objectives: Sequence[str], salt: Sequence[str] = ()) -> Dict[str, object]:
    """Summarise samples that share a name and a group.

    ``salt`` tells the bootstrap of a part of the group, such as one cell, from
    that of the whole group.
    """
    name, group = samples[0].name, samples[0].group
    if any(s.name != name or s.group != group for s in samples):
        raise ValueError("samples to summarise must share one name and one group")
    episodes = [s.seed for s in samples]
    gains = np.stack([s.gains for s in samples])
    return {
        "name": name,
        "group": group,
        "samples": len(samples),
        "episodes": episode_count(samples),
        "completion_rate": round(float(np.mean([s.completed for s in samples])), DECIMALS),
        "mean_frames": round(float(np.mean([s.frames for s in samples])), DECIMALS),
        "support": _support([s.state for s in samples]),
        **summarise_gains(gains, objectives, _bootstrap_seed(name, group, *salt), episodes),
    }


def cell_scale(states: np.ndarray) -> np.ndarray:
    """The spread of each number of the states, which distances between them are measured in; 1 where it is 0."""
    spread = np.asarray(states, dtype=np.float64).std(axis=0)
    return np.where(spread > 0.0, spread, 1.0)


def partition(
    states: np.ndarray, episodes: Sequence[int], cell_episodes: Optional[int], min_episodes: int, seed: int
) -> np.ndarray:
    """The cell of each state: clusters of similar states, each measured in at least ``min_episodes`` episodes.

    k-means starts with about one cell per ``cell_episodes`` episodes. While a cell
    holds fewer than ``min_episodes`` episodes, the smallest is dropped and k-means
    goes on from the centres left, so a few outlying states join their nearest
    cell rather than keep the whole group from being split. Every state ends in the
    cell whose centre is nearest, as a lookup finds it. There is one cell when
    ``cell_episodes`` is None. The cells depend on the states, the episodes they
    come from and ``seed`` only, never on what was measured from them.
    """
    states = np.asarray(states, dtype=np.float64)
    episodes = np.asarray(episodes)
    if states.ndim != 2 or len(states) != len(episodes):
        raise ValueError("every state needs the episode it was measured in")
    if cell_episodes is not None and cell_episodes < 1:
        raise ValueError(f"cell_episodes must be at least 1, got {cell_episodes}")
    whole = np.zeros(len(states), dtype=np.int64)
    scaled = states / cell_scale(states)
    count = 1 if cell_episodes is None else len(np.unique(episodes)) // cell_episodes
    count = min(count, len(np.unique(scaled, axis=0)))
    if count < 2:
        return whole
    labels, centres = _lloyd(scaled, _first_centres(scaled, count, np.random.default_rng(seed)))
    while len(centres) > 1:
        sizes = [len(np.unique(episodes[labels == cell])) for cell in range(len(centres))]
        smallest = int(np.argmin(sizes))
        if sizes[smallest] >= min_episodes:
            return labels
        labels, centres = _lloyd(scaled, np.delete(centres, smallest, axis=0))
    return whole


def _first_centres(points: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++: each new centre is drawn with probability growing with its distance from the centres so far."""
    centres = [points[rng.integers(len(points))]]
    for _ in range(1, count):
        nearest = np.min([np.sum((points - centre) ** 2, axis=1) for centre in centres], axis=0)
        centres.append(points[rng.choice(len(points), p=nearest / nearest.sum())])
    return np.array(centres)


def _lloyd(points: np.ndarray, centres: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means from ``centres``: the cell of every point, numbered from 0, and the centres of the cells kept."""
    centres = np.array(centres, dtype=np.float64)
    labels = np.full(len(points), -1)
    for _ in range(KMEANS_ITERATIONS):
        new = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for cell in range(len(centres)):
            if np.any(labels == cell):
                centres[cell] = points[labels == cell].mean(axis=0)
    kept = np.unique(labels)  # a centre no point is nearest to is dropped
    return np.searchsorted(kept, labels), centres[kept]


def _plan_summary(flights: Sequence[PlanFlight], domain: Domain, salt: Sequence[str] = ()) -> Dict[str, object]:
    """Summary of flights of one plan from one group of states, or from one cell of the group."""
    entry = summarise([flight.plan for flight in flights], domain.objectives, salt)
    entry["outcomes"] = {
        name: round(sum(flight.outcome == name for flight in flights) / len(flights), DECIMALS)
        for name in (*domain.outcomes, IN_FLIGHT)
    }
    stepped = [flight for flight in flights if flight.steps]
    if stepped:
        entry["interaction"] = summarise_gains(
            np.stack([flight.interaction for flight in stepped]),
            domain.objectives,
            _bootstrap_seed(entry["name"], entry["group"], *salt, "interaction"),
            [flight.seed for flight in stepped],
        )
    return entry


_CELL_FIELDS = ("name", "group", "support")  # shared with the whole group, so not repeated in a cell


def _plan_entry(
    flights: Sequence[PlanFlight], domain: Domain, min_samples: int, cell_episodes: Optional[int]
) -> Dict[str, object]:
    """The table entry of one plan from one group of states, split into cells when there are enough episodes."""
    entry = _plan_summary(flights, domain)
    entry["skills"] = list(flights[0].skills)
    states = np.array([flight.plan.state for flight in flights])
    seed = _bootstrap_seed(str(entry["name"]), str(entry["group"]), "cells")
    labels = partition(states, [flight.seed for flight in flights], cell_episodes, min_samples, seed)
    if labels.max() > 0:
        entry["cell_scale"] = [round(float(value), DECIMALS) for value in cell_scale(states)]
        entry["cells"] = []
        for cell in range(int(labels.max()) + 1):
            members = [flight for flight, label in zip(flights, labels) if label == cell]
            summary = _plan_summary(members, domain, (f"cell{cell}",))
            centroid = states[labels == cell].mean(axis=0)
            entry["cells"].append(
                {
                    "centroid": [round(float(value), DECIMALS) for value in centroid],
                    **{key: value for key, value in summary.items() if key not in _CELL_FIELDS},
                }
            )
    return entry


def _support(states: Sequence[np.ndarray]) -> Dict[str, List[float]]:
    """The lowest and highest value of each number of the states, rounded outwards."""
    values = np.asarray(states, dtype=np.float64)
    scale = 10.0**DECIMALS
    return {
        "low": [float(v) for v in np.floor(values.min(axis=0) * scale) / scale],
        "high": [float(v) for v in np.ceil(values.max(axis=0) * scale) / scale],
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
    cell_episodes: Optional[int] = CELL_EPISODES,
) -> "EffectTable":
    """Fly the plans ``plans_at`` gives at every decision state of every seed and summarise the measured effects."""
    measured = measure_seeds(domain, seeds, plans_at, max_states)
    return table_from_measurements(domain, seeds, measured, min_samples, cell_episodes)


def table_from_measurements(
    domain: Domain,
    seeds: range,
    measurements: Sequence[SeedMeasurement],
    min_samples: int = MIN_SAMPLES,
    cell_episodes: Optional[int] = CELL_EPISODES,
) -> "EffectTable":
    """Summarise measurements of ``seeds``.

    Entries measured in fewer than ``min_samples`` episodes could never answer a
    lookup, so they are left out. Each plan's group is split into cells of about
    ``cell_episodes`` episodes, as the module describes.
    """
    flights = [flight for measurement in measurements for flight in measurement.flights]
    step_samples = [step for measurement in measurements for step in measurement.steps]

    objectives = domain.objectives
    steps = [
        summarise(group, objectives)
        for group in _by_name_and_group(step_samples, lambda s: s)
        if episode_count(group) >= min_samples
    ]
    plans_summary = [
        _plan_entry(group, domain, min_samples, cell_episodes)
        for group in _by_name_and_group(flights, lambda f: f.plan)
        if episode_count(flight.plan for flight in group) >= min_samples
    ]

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
            "cell_episodes": cell_episodes,
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
    episodes: int  # the distinct episodes the samples come from, at most ``samples``
    completion_rate: float
    mean_frames: float
    mean: Gains
    lower: Gains  # one-sided lower confidence bounds
    interaction: Optional[Gains] = None  # mean interaction; plans flown from the start of an episode only
    outcomes: Optional[Dict[str, float]] = None  # how plan flights really ended; plans only
    region: str = ""  # the states it was measured from: a group, or one cell of it such as "...|cell=2"


def _gains(stats: Mapping[str, Mapping[str, float]], objectives: Sequence[str], field: str) -> Gains:
    return Gains(
        delta_r=float(stats["delta_r"][field]),
        delta_n=tuple(float(stats[f"delta_{name}"][field]) for name in objectives),
    )


class EffectTable:
    """Measured effects of sub-skills and plans, looked up by the state they would start from.

    A lookup takes the observation and how many frames of the episode were flown
    before it. It finds the group of the observation's facts and answers only if
    the group was measured in at least ``min_samples`` episodes and the state lies
    inside the group's support, widened on each side by ``support_margin`` times
    its range; a state later in the episode than any measured is outside it. Every
    other state is treated as unmeasured. A plan's group split into cells answers
    from the cell whose centroid is nearest the state, in units of the group's
    spread.
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

    def step(self, skill: str, observation: np.ndarray, elapsed: float) -> Optional[Estimate]:
        """The effect of ``skill`` from ``observation`` after ``elapsed`` frames; None if nothing measured applies."""
        entry = self._steps.get((skill, self._group_of(observation)))
        return self._estimate(entry, state(observation, elapsed))

    def plan(self, skills: Sequence[str], observation: np.ndarray, elapsed: float) -> Optional[Estimate]:
        """The effect of ``skills`` from ``observation`` after ``elapsed`` frames; None if nothing measured applies."""
        entry = self._plans.get((plan_key(skills), self._group_of(observation)))
        return self._estimate(entry, state(observation, elapsed))

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

    def supports(self, entry: Mapping[str, Any], observation: np.ndarray, elapsed: float) -> bool:
        """Whether the state ``observation`` after ``elapsed`` frames lies inside the entry's support, widened."""
        return self._covers(entry, state(observation, elapsed))

    def _covers(self, entry: Mapping[str, Any], values: np.ndarray) -> bool:
        low = np.asarray(entry["support"]["low"], dtype=np.float64)
        high = np.asarray(entry["support"]["high"], dtype=np.float64)
        widen = self.support_margin * (high - low)
        return bool(values.shape == low.shape and np.all((values >= low - widen) & (values <= high + widen)))

    def _group_of(self, observation: np.ndarray) -> str:
        return group_key(self.domain, self.domain.facts(observation))

    def nearest_cell(self, entry: Mapping[str, Any], values: np.ndarray) -> Optional[int]:
        """The index of the entry's cell nearest the state ``values``, or None for an entry that is not split."""
        cells = entry.get("cells")
        if not cells:
            return None
        scale = np.asarray(entry["cell_scale"], dtype=np.float64)
        centroids = np.asarray([cell["centroid"] for cell in cells], dtype=np.float64)
        return int((((centroids - np.asarray(values, dtype=np.float64)) / scale) ** 2).sum(axis=1).argmin())

    def _estimate(self, entry: Optional[Mapping[str, Any]], values: np.ndarray) -> Optional[Estimate]:
        if entry is None or int(entry["episodes"]) < self.min_samples or not self._covers(entry, values):
            return None
        cell = self.nearest_cell(entry, values)
        stats = entry if cell is None else entry["cells"][cell]
        region = str(entry["group"]) if cell is None else f"{entry['group']}|cell={cell}"
        objectives = self.domain.objectives
        return Estimate(
            samples=int(stats["samples"]),
            episodes=int(stats["episodes"]),
            completion_rate=float(stats["completion_rate"]),
            mean_frames=float(stats["mean_frames"]),
            mean=_gains(stats, objectives, "mean"),
            lower=_gains(stats, objectives, "lower"),
            interaction=_gains(stats["interaction"], objectives, "mean") if "interaction" in stats else None,
            outcomes=dict(stats["outcomes"]) if "outcomes" in stats else None,
            region=region,
        )

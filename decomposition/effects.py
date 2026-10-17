"""Measured effects of the LunarLander sub-skills and of plans built from them.

The effect of running something from a state is its payoff gain Δr and its
motive gains Δn = (ΔSafety, ΔFuel) over doing nothing for the same number of
frames from the same state. Box2D state cannot be copied, but resetting the
environment with a seed and replaying the same actions reproduces a state
exactly, so both runs start from identical states (a paired comparison).

Totals follow ``env.skill_executor.SkillExecutor`` (discount 0.99, payoff is
the sum of the reward vector) and the gains come from
``baseline.improvement_calculator.ImprovementCalculator``.

A plan is an ordered list of sub-skills flown from the start of an episode. If
the plan leaves the lander standing on its legs, its last sub-skill keeps
running until the episode ends, so the plan's effect includes the landing
reward. Each step of the flight is also measured from the state where it
started; a plan's interaction is its effect minus the sum of its steps'
effects.

Samples are grouped by the facts of the state they start from and summarised
by their mean, spread, count and a one-sided bootstrap lower confidence bound.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from env.lunar_facts import lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import NO_ENGINE, SUBSKILLS, run_subskill

SCHEMA_VERSION = "subrep.lunar_effects.v1"
ENVIRONMENT = "mo-lunar-lander-v3"
BASELINE_ID = "idle_policy_v1"
GAMMA = 0.99
GROUP_FACTS = ("upright", "over_pad", "near_ground")
CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000
MIN_SAMPLES = 30
DEFAULT_TABLE_PATH = Path(__file__).resolve().parent / "data" / "lunar_effects.json"

_GAIN_NAMES = ("delta_r", "delta_safety", "delta_fuel")
_DECIMALS = 6

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Paired measurement
# ---------------------------------------------------------------------------


def replay(env: SubRepEnv, seed: int, actions: Sequence[int]) -> np.ndarray:
    """Reset ``env`` with ``seed``, replay ``actions`` and return the observation reached."""
    obs, _ = env.reset(seed=seed)
    for action in actions:
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            raise ValueError(f"the episode of seed {seed} ended before all actions were replayed")
    return np.asarray(obs, dtype=np.float32)


def idle_rewards(env: SubRepEnv, frames: int) -> np.ndarray:
    """Rewards of doing nothing for ``frames`` frames, or until the episode ends."""
    rewards = []
    for _ in range(frames):
        _, reward, terminated, truncated, _ = env.step(NO_ENGINE)
        rewards.append(np.asarray(reward, dtype=np.float64))
        if terminated or truncated:
            break
    return np.asarray(rewards, dtype=np.float64).reshape(-1, 2)


def discounted_totals(rewards: np.ndarray, gamma: float = GAMMA) -> Tuple[float, np.ndarray]:
    """Discounted payoff and motive totals of a ``(frames, 2)`` reward array."""
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1, 2)
    discounts = gamma ** np.arange(len(rewards))
    motives = discounts @ rewards
    return float(motives.sum()), motives


def improvement(rewards: np.ndarray, baseline_rewards: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """Payoff gain Δr and motive gains Δn of ``rewards`` over ``baseline_rewards``."""
    payoff, motives = discounted_totals(rewards)
    baseline_payoff, baseline_motives = discounted_totals(baseline_rewards)
    calculator = ImprovementCalculator({"baseline_payoff": baseline_payoff, "baseline_motives": baseline_motives})
    delta_r, delta_n = calculator.compute_improvements(payoff, motives)
    return float(delta_r), (float(delta_n[0]), float(delta_n[1]))


def group_key(facts: Mapping[str, bool]) -> str:
    """The group of a state, for example ``"upright=1,over_pad=0,near_ground=0"``."""
    return ",".join(f"{name}={int(facts[name])}" for name in GROUP_FACTS)


def plan_key(skills: Sequence[str]) -> str:
    """The name of a plan, for example ``"stabilise>move_over_pad"``."""
    return ">".join(skills)


@dataclass(frozen=True)
class Sample:
    """One paired measurement of a sub-skill or a plan from one state."""

    name: str
    group: str
    delta_r: float
    delta_n: Tuple[float, float]
    frames: int
    completed: bool

    @property
    def gains(self) -> np.ndarray:
        """(Δr, ΔSafety, ΔFuel) as one array."""
        return np.array([self.delta_r, *self.delta_n])


@dataclass(frozen=True)
class PlanFlight:
    """One plan flown from the start of one episode, with each step measured too."""

    seed: int
    skills: Tuple[str, ...]
    plan: Sample
    steps: Tuple[Sample, ...]  # the steps that ran; fewer than ``skills`` if one failed

    @property
    def interaction(self) -> np.ndarray:
        """The plan's (Δr, ΔSafety, ΔFuel) minus the sum over its steps."""
        return self.plan.gains - sum((step.gains for step in self.steps), np.zeros(3))


def fly_plan(env: SubRepEnv, seed: int, skills: Sequence[str]) -> PlanFlight:
    """Fly ``skills`` in order from the start of episode ``seed`` and measure the plan and each step.

    The flight stops at the first step that does not complete or when the
    episode ends.
    """
    if not skills:
        raise ValueError("a plan needs at least one sub-skill")

    obs = replay(env, seed, ())
    plan_group = group_key(lander_facts(obs))
    actions: List[int] = []
    rewards: List[np.ndarray] = []
    runs = []  # (frames flown before the step, group the step started in, its run)
    ended = False
    for name in skills:
        run = run_subskill(env, SUBSKILLS[name], obs)
        runs.append((len(actions), group_key(lander_facts(obs)), run))
        actions.extend(run.actions)
        rewards.extend(run.rewards)
        obs = run.observation
        ended = run.terminated or run.truncated
        if ended or not run.completed:
            break

    completed = len(runs) == len(skills) and runs[-1][2].completed
    if completed and not ended and lander_facts(obs)["landed"]:
        last = SUBSKILLS[skills[-1]]
        while not ended:
            obs, reward, terminated, truncated, _ = env.step(int(last.policy(obs)))
            rewards.append(np.asarray(reward, dtype=np.float64))
            ended = terminated or truncated

    plan_rewards = np.asarray(rewards, dtype=np.float64).reshape(-1, 2)
    replay(env, seed, ())
    delta_r, delta_n = improvement(plan_rewards, idle_rewards(env, len(plan_rewards)))
    plan = Sample(plan_key(skills), plan_group, delta_r, delta_n, len(plan_rewards), completed)

    steps = []
    for start, group, run in runs:
        replay(env, seed, actions[:start])
        delta_r, delta_n = improvement(run.rewards, idle_rewards(env, run.frames))
        steps.append(Sample(run.skill, group, delta_r, delta_n, run.frames, run.completed))

    return PlanFlight(seed=seed, skills=tuple(skills), plan=plan, steps=tuple(steps))


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


def summarise_gains(gains: np.ndarray, seed: int) -> Dict[str, Dict[str, float]]:
    """Mean, spread and lower bound of each column of a ``(samples, 3)`` gains array."""
    summary = {}
    for name, column in zip(_GAIN_NAMES, np.asarray(gains, dtype=np.float64).T):
        summary[name] = {
            "mean": round(float(column.mean()), _DECIMALS),
            "std": round(float(column.std()), _DECIMALS),
            "lower": round(lower_bound(column, seed), _DECIMALS),
        }
    return summary


def summarise(samples: Sequence[Sample]) -> Dict[str, object]:
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
        **summarise_gains(np.stack([s.gains for s in samples]), _bootstrap_seed(name, group)),
    }


def _by_name_and_group(items: Iterable[T], sample_of: Callable[[T], Sample]) -> List[List[T]]:
    grouped: Dict[Tuple[str, str], List[T]] = {}
    for item in items:
        sample = sample_of(item)
        grouped.setdefault((sample.name, sample.group), []).append(item)
    return [grouped[key] for key in sorted(grouped)]


def build_table(seeds: range, plans: Sequence[Sequence[str]], min_samples: int = MIN_SAMPLES) -> "EffectTable":
    """Fly every plan from every seed and summarise the measured effects.

    A step that starts after the same sub-skills in two plans is flown from the
    same state, so it is counted once per seed.
    """
    env = SubRepEnv(seed=seeds.start)
    try:
        flights = [fly_plan(env, seed, skills) for seed in seeds for skills in plans]
    finally:
        env.close()

    step_samples: Dict[Tuple[int, Tuple[str, ...]], Sample] = {}
    for flight in flights:
        for index, step in enumerate(flight.steps):
            step_samples.setdefault((flight.seed, flight.skills[: index + 1]), step)

    steps = [summarise(group) for group in _by_name_and_group(step_samples.values(), lambda s: s)]
    plans_summary = []
    for group in _by_name_and_group(flights, lambda f: f.plan):
        entry = summarise([flight.plan for flight in group])
        entry["skills"] = list(group[0].skills)
        interaction = np.stack([flight.interaction for flight in group])
        entry["interaction"] = summarise_gains(interaction, _bootstrap_seed(entry["name"], entry["group"], "interaction"))
        plans_summary.append(entry)

    return EffectTable(
        {
            "schema_version": SCHEMA_VERSION,
            "environment": ENVIRONMENT,
            "baseline_id": BASELINE_ID,
            "gamma": GAMMA,
            "confidence": CONFIDENCE,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "min_samples": min_samples,
            "group_facts": list(GROUP_FACTS),
            "seeds": {"start": seeds.start, "stop": seeds.stop},
            "steps": steps,
            "plans": plans_summary,
        }
    )


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gains:
    """A payoff gain Δr with motive gains Δn = (ΔSafety, ΔFuel)."""

    delta_r: float
    delta_n: Tuple[float, float]


@dataclass(frozen=True)
class Estimate:
    """The measured effect of a sub-skill or a plan from one group of states."""

    samples: int
    completion_rate: float
    mean_frames: float
    mean: Gains
    lower: Gains  # one-sided lower confidence bounds
    interaction: Optional[Gains] = None  # mean interaction; plans only


def _gains(stats: Mapping[str, Mapping[str, float]], field: str) -> Gains:
    return Gains(
        delta_r=float(stats["delta_r"][field]),
        delta_n=(float(stats["delta_safety"][field]), float(stats["delta_fuel"][field])),
    )


class EffectTable:
    """Measured effects of sub-skills and plans, looked up by the facts of a start state.

    Groups with fewer than ``min_samples`` samples are treated as unknown.
    """

    def __init__(self, data: Mapping[str, object]) -> None:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"expected schema_version {SCHEMA_VERSION!r}, got {data.get('schema_version')!r}")
        if tuple(data["group_facts"]) != GROUP_FACTS:
            raise ValueError(f"table groups by {data['group_facts']}, expected {list(GROUP_FACTS)}")
        self.data = dict(data)
        self.min_samples = int(data["min_samples"])
        self._steps = {(e["name"], e["group"]): e for e in data["steps"]}
        self._plans = {(e["name"], e["group"]): e for e in data["plans"]}

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EffectTable":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8")

    def step(self, skill: str, facts: Mapping[str, bool]) -> Optional[Estimate]:
        """The effect of ``skill`` from a state with ``facts``, or None if not measured enough."""
        return self._estimate(self._steps.get((skill, group_key(facts))))

    def plan(self, skills: Sequence[str], facts: Mapping[str, bool]) -> Optional[Estimate]:
        """The effect of the plan ``skills`` from a state with ``facts``, or None if not measured enough."""
        return self._estimate(self._plans.get((plan_key(skills), group_key(facts))))

    def _estimate(self, entry: Optional[Mapping[str, object]]) -> Optional[Estimate]:
        if entry is None or int(entry["samples"]) < self.min_samples:
            return None
        return Estimate(
            samples=int(entry["samples"]),
            completion_rate=float(entry["completion_rate"]),
            mean_frames=float(entry["mean_frames"]),
            mean=_gains(entry, "mean"),
            lower=_gains(entry, "lower"),
            interaction=_gains(entry["interaction"], "mean") if "interaction" in entry else None,
        )

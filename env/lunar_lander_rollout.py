"""Candidate rollouts on MO-LunarLander, for the MetaMo integration.

The LunarLander counterpart of env/minecraft_rollout.py: it scores candidate
policies against a baseline in the form SubRep's gates expect, and turns the
executed episode into the `SkillOutcome` the bridge feeds back to MetaMo.

Three rules hold throughout:

* Same seed. The baseline and every candidate start from `env.reset(seed)`,
  with the action space and torch reseeded too, so random candidates replay
  and luck does not decide which one wins.
* Separate task reward. r_hat is the landing result alone
  (`info["terminal_reward"]`, +100 landed / -100 crashed), not the sum of the
  objectives. n_hat stays the [Safety, Fuel] vector. The PDS gate
  (delta_r + min(delta_n) >= -eps) needs the two signals apart.
* Replay, not copy. Execution re-runs the chosen policy from the seed rather
  than deep-copying the environment: Box2D state does not copy cleanly, and
  `reset(seed)` is already proven to reproduce the start
  (data_collector/collect_candidate_sets.py, `collect_context`).

`SkillExecutor` (env/skill_executor.py) is not reused: it builds the payoff
from the objective vector, never sees `info`, and prints every episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from baseline.improvement_calculator import ImprovementCalculator
from bridge.protocol import SkillOutcome
from data_collector.collect_candidate_sets import CandidatePolicy, _fixed_action_policy
from env.skill_executor import SkillExecutor
from utils.mdn_contracts import CandidateSkillRecord

# No discounting. The landing result arrives on the final step, so any
# gamma < 1 shrinks it by gamma^(steps - 1) and makes a later crash read as
# better than an earlier one. Compared over seeds 1-20 with the default
# candidates:
#   gamma 0.9   -- a landing at ~200 steps is discounted to ~0; landing and
#                  crash both give delta_r ~0.
#   gamma 0.99  -- main_engine crashes every time yet scores delta_r ~+18,
#                  a third of a real landing (~+58).
#   gamma 0.999 -- landings stand out (~+169), small timing bias remains.
#   gamma 1.0   -- crash-vs-crash delta_r is exactly 0 and a landing scores
#                  the full +200 over a crashing baseline.
# Undiscounted sums are only safe because every episode is finite: this relies
# on the environment's 1000-step time limit (EPISODE_STEP_LIMIT), which ends
# an uncapped episode by truncation if it neither lands nor crashes.
DEFAULT_GAMMA = 1.0

# Run each episode until the environment ends it. The collector's 200-step cap
# is deliberately not copied: most landings take longer than that, so a capped
# candidate is cut off before it lands or crashes, its landing result is 0,
# and the task reward carries no information.
DEFAULT_MAX_STEPS: Optional[int] = None

# mo-lunar-lander-v3's TimeLimit. The longest an uncapped episode can run, so
# it is the effort denominator when `max_steps` is None.
EPISODE_STEP_LIMIT = 1000

# Doing nothing: the main engine and both side engines stay off.
NOOP_BASELINE = CandidatePolicy("noop", _fixed_action_policy(0))

# Lower bound on both appraisal scales. With gamma = 1.0, delta_r only takes
# the values 0, +-100 (one side timed out instead of ending) and +-200 (one
# landed, the other crashed), so there is no crash-timing noise to guard
# against. Measured over seeds 1-20 with the default candidates (uncapped):
# the scales are (65, 301) pooled and never below (50, 195) for a single
# seed, so the floor does not bind on an uncapped set. It guards capped runs
# instead: with max_steps <= 50 no candidate lands or crashes, every delta_r
# is 0 (the payoff scale falls back to 1.0), and the motive scale drops to
# ~6 at max_steps = 10. The floor keeps such a set from producing scales so
# small that the next non-zero outcome saturates every stimulus.
APPRAISAL_SCALE_FLOOR = 10.0


def _policy_callable(policy: Any) -> Any:
    """The callable behind a `CandidatePolicy`, or `policy` itself."""
    return getattr(policy, "policy_fn", policy)


def _check_same_start(obs: np.ndarray, reference: np.ndarray, skill_id: str) -> None:
    """Raise unless `obs` is the reference start, as `collect_context` does."""
    if not np.allclose(obs, reference, rtol=1e-6, atol=1e-6):
        raise RuntimeError(
            f"Reset seed did not reproduce the baseline start for {skill_id!r}"
        )


def appraisal_scales(
    records: Sequence[Any],
    *,
    floor: Optional[float] = APPRAISAL_SCALE_FLOOR,
) -> Tuple[float, float]:
    """Characteristic |delta_r| and |delta_n| magnitudes for the governor.

    The stimulus builder squashes through tanh, so the scales decide how loud
    one episode is: too large and a crash barely registers, too small and
    every stimulus saturates at 0 or 1. The aim is that a crash reads as a
    strong threat without pinning MetaMo's modulators after one episode.
    Returns (payoff_scale, motive_scale), each at least `floor`.
    """
    payoff_scale = float(np.mean([abs(r.delta_r) for r in records])) or 1.0
    motive_scale = float(
        np.mean([np.mean(np.abs(r.delta_n)) for r in records])
    ) or 1.0
    if floor is not None:
        payoff_scale = max(payoff_scale, float(floor))
        motive_scale = max(motive_scale, float(floor))
    return payoff_scale, motive_scale


def run_episode(
    env: Any,
    policy: Any,
    *,
    seed: int,
    gamma: float = DEFAULT_GAMMA,
    max_steps: Optional[int] = DEFAULT_MAX_STEPS,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """Run `policy` for one episode from `env.reset(seed=seed)`.

    `policy` is a `CandidatePolicy` or a bare callable returning an action or
    an `(action, probability)` tuple. `max_steps=None` runs until the
    environment ends the episode; the env is never stepped after that.

    Returns (r_hat, n_hat, info), discounted with the same gamma:

        r_hat = sum_t gamma^t * terminal_reward_t   (landing result only)
        n_hat = sum_t gamma^t * [Safety_t, Fuel_t]

    Running the same policy twice with the same seed gives identical results.
    The global torch random state is left as it was. The `random` candidate
    replays only when it was built from this same `env`, since it samples
    `env.env.action_space`.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be None or >= 1, got {max_steps}")

    policy_fn = _policy_callable(policy)
    obs, _ = env.reset(seed=seed)
    start_obs = np.array(obs, dtype=np.float32, copy=True)
    # The `random` candidate samples this space.
    env.env.action_space.seed(seed)

    r_hat = 0.0
    n_hat = np.zeros(2, dtype=np.float64)
    discount = 1.0
    steps = 0
    terminated = False
    truncated = False
    info: Dict[str, Any] = {}

    # Seed torch for `ppo_stochastic` inside a fork, so the global state the
    # CVaR gate draws from is restored afterwards. The whole episode runs
    # inside it, since the policy draws on every step.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        if hasattr(policy_fn, "reset"):
            policy_fn.reset(seed=seed)

        while max_steps is None or steps < max_steps:
            action, _ = SkillExecutor._parse_policy_output(policy_fn(obs))
            obs, reward_vec, terminated, truncated, info = env.step(int(action))
            r_hat += discount * float(info["terminal_reward"])
            n_hat += discount * np.asarray(reward_vec, dtype=np.float64)
            steps += 1
            if terminated or truncated:
                break
            discount *= gamma

    if terminated:
        stop_reason = "terminated"
    elif truncated:
        stop_reason = "truncated"
    else:
        stop_reason = "max_steps"

    return float(r_hat), n_hat.astype(np.float32), {
        "seed": int(seed),
        "start_obs": start_obs,
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "stop_reason": stop_reason,
        "landing_success": bool(info.get("landing_success", False)),
        "crashed": bool(info.get("crashed", False)),
        "terminal_reward": float(info.get("terminal_reward", 0.0)),
        "gamma": float(gamma),
        "max_steps": max_steps,
    }


def evaluate_candidates(
    env: Any,
    policies: Sequence[CandidatePolicy],
    *,
    seed: int,
    gamma: float = DEFAULT_GAMMA,
    max_steps: Optional[int] = DEFAULT_MAX_STEPS,
    baseline: CandidatePolicy = NOOP_BASELINE,
) -> Tuple[Dict[str, Any], List[CandidateSkillRecord]]:
    """Score every candidate against the baseline, all from the same seed.

    The candidate whose `skill_id` matches the baseline's is skipped, so it is
    not scored against itself. Raises RuntimeError if any run starts from a
    different observation than the baseline.

    Returns (baseline_stats, records): `baseline_stats` carries the keys
    `ImprovementCalculator` requires; `records` has one PDS candidate per
    non-baseline policy.
    """
    baseline_payoff, baseline_motives, baseline_info = run_episode(
        env, baseline, seed=seed, gamma=gamma, max_steps=max_steps
    )
    baseline_stats: Dict[str, Any] = {
        "baseline_payoff": baseline_payoff,
        "baseline_motives": baseline_motives,
        "baseline_id": baseline.skill_id,
        "seed": int(seed),
        "info": baseline_info,
    }
    calculator = ImprovementCalculator(baseline_stats)

    records: List[CandidateSkillRecord] = []
    for candidate in policies:
        if candidate.skill_id == baseline.skill_id:
            continue
        payoff, motives, info = run_episode(
            env, candidate, seed=seed, gamma=gamma, max_steps=max_steps
        )
        _check_same_start(info["start_obs"], baseline_info["start_obs"], candidate.skill_id)
        delta_r, delta_n = calculator.compute_improvements(
            skill_payoff=payoff,
            skill_motives=motives,
        )
        records.append(
            CandidateSkillRecord(
                skill_id=candidate.skill_id,
                delta_r=float(delta_r),
                delta_n=tuple(float(v) for v in delta_n),
                is_certified=False,
                gate_type="PDS",
                baseline_id=baseline.skill_id,
                metadata={
                    "seed": int(seed),
                    "steps": info["steps"],
                    "landing_success": info["landing_success"],
                    "crashed": info["crashed"],
                    "truncated": info["truncated"],
                },
            )
        )
    return baseline_stats, records


def execute_candidate(
    env: Any,
    policy: Any,
    *,
    seed: int,
    gamma: float = DEFAULT_GAMMA,
    max_steps: Optional[int] = DEFAULT_MAX_STEPS,
    baseline: CandidatePolicy = NOOP_BASELINE,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    """Fly the chosen policy from `seed`; return its realized (delta_r, delta_n, info).

    The baseline is flown from the same seed with the same gamma and step
    limit, so the improvement is measured against what doing nothing would
    have produced from this exact start -- the same comparison
    `evaluate_candidates` made. `info` is the chosen episode's info, plus the
    baseline's step count.
    """
    baseline_payoff, baseline_motives, baseline_info = run_episode(
        env, baseline, seed=seed, gamma=gamma, max_steps=max_steps
    )
    payoff, motives, info = run_episode(
        env, policy, seed=seed, gamma=gamma, max_steps=max_steps
    )
    _check_same_start(
        info["start_obs"],
        baseline_info["start_obs"],
        str(getattr(policy, "skill_id", "executed policy")),
    )
    calculator = ImprovementCalculator(
        {"baseline_payoff": baseline_payoff, "baseline_motives": baseline_motives}
    )
    delta_r, delta_n = calculator.compute_improvements(
        skill_payoff=payoff,
        skill_motives=motives,
    )
    info = dict(info)
    info["baseline_steps"] = baseline_info["steps"]
    return float(delta_r), np.asarray(delta_n, dtype=np.float64), info


def episode_outcome(
    delta_r: float,
    delta_n: Sequence[float],
    info: Dict[str, Any],
    *,
    admitted: bool,
    max_steps: Optional[int] = DEFAULT_MAX_STEPS,
) -> SkillOutcome:
    """The `SkillOutcome` for an executed episode, for the governor's `step()`.

    `delta_r`, `delta_n` and `info` come from `execute_candidate`; `admitted`
    is whether the chosen candidate passed the gates. When `max_steps` is not
    given, the step limit the episode actually ran with (`info["max_steps"]`)
    is used, so effort cannot be measured against a different limit.
    """
    # Effort is flight time: the episode's steps over the most it could have
    # run (`max_steps`, capped by the env's own time limit). A quick touchdown
    # reads as low effort, a long hover or a timeout as high effort. Fuel is
    # not used here because it is already an objective in delta_n.
    if max_steps is None:
        max_steps = info.get("max_steps")
    limit = EPISODE_STEP_LIMIT if max_steps is None else min(int(max_steps), EPISODE_STEP_LIMIT)
    effort = float(np.clip(float(info["steps"]) / float(limit), 0.0, 1.0))
    return SkillOutcome(
        delta_r=float(delta_r),
        delta_n=np.asarray(delta_n, dtype=np.float64),
        admitted=bool(admitted),
        effort=effort,
    )

"""Option rollouts on the Minecraft stub, for the MetaMo integration.

The single place where the MetaMo path estimates delta_r / delta_n: the demo
(demo/run_metamo_pipeline.py), the end-to-end tests and the ablation harness
all use it, so each estimation rule lives here once.

It deliberately does NOT use `IdlePolicy` (baseline/idle_policy.py). That code
was written for the 2-objective LunarLander environment and is correct there;
the MetaMo path needs rules that differ for the 6-objective stub, and shared
SubRep code is left untouched.

Rollouts from a given state use `copy.deepcopy(env)` (`evaluate_option`). The
stub is pure numpy, so a deep copy is an exact snapshot. That function is the
one place to change if a real environment ever replaces the stub.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from env.minecraft_stub import SKILL_NAMES
from utils.mdn_contracts import CandidateSkillRecord

# A survival-oriented starting goal vector, in MetaMo's goal order
# (core/config.py:1-3): Individuation, Transcendence, Help, Curiosity, Novelty,
# Self, Ethical, Social.
#
# MetaMo's own default (core/engine.py:47-58) is tuned for a chat assistant
# (Help=0.8, Ethical=0.9) and makes Reputation dominate from the first step,
# which is the wrong prior for an agent that has to survive the night.
#
# Defined once here so the demo, the tests and the ablation harness cannot
# drift apart.
MINECRAFT_INITIAL_GOALS = np.array(
    [0.70, 0.40, 0.30, 0.50, 0.40, 0.60, 0.50, 0.30], dtype=np.float64
)

# Option duration, in environment steps. The reference specification draws a
# per-option duration tau; a fixed H simplifies that and keeps variance low for
# the ablation. Candidate evaluation, execution and feedback all use the SAME
# horizon -- measuring an option over one duration and running it for another
# is what made the old estimates meaningless.
DEFAULT_HORIZON = 3


def appraisal_scales(
    candidates: Sequence[Any],
    *,
    floor: Optional[float] = None,
) -> Tuple[float, float]:
    """Characteristic |delta_r| and |delta_n| magnitudes for the governor.

    The stimulus builder squashes through tanh, so leaving the governor's
    scales at 1.0 while deltas run to |10| saturates risk on the first step
    and pins the modulators at their bounds, flattening the coupling into a
    constant. Returns (payoff_scale, motive_scale).
    """
    payoff_scale = float(np.mean([abs(c.delta_r) for c in candidates])) or 1.0
    motive_scale = float(
        np.mean([np.mean(np.abs(c.delta_n)) for c in candidates])
    ) or 1.0
    if floor is not None:
        payoff_scale = max(payoff_scale, float(floor))
        motive_scale = max(motive_scale, float(floor))
    return payoff_scale, motive_scale


def run_option(
    env: Any,
    action: int,
    *,
    horizon: Optional[int],
    gamma: float,
) -> Tuple[float, np.ndarray]:
    """Step `env` ITSELF with `action` for up to `horizon` steps.

    `horizon=None` runs to the end of the episode. Stops early on termination
    or truncation and never steps past the end. Mutates `env`.

    Returns the discounted (r_hat, n_hat).
    """
    if horizon is not None and horizon < 1:
        raise ValueError(f"horizon must be None or >= 1, got {horizon}")

    discount = 1.0
    r_hat = 0.0
    n_hat: Optional[np.ndarray] = None
    steps = 0

    while horizon is None or steps < horizon:
        _, reward_vec, terminated, truncated, _ = env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32)
        if n_hat is None:
            n_hat = np.zeros_like(reward_vec)
        r_hat += discount * float(np.sum(reward_vec))
        n_hat += discount * reward_vec
        steps += 1
        if terminated or truncated:
            break
        discount *= gamma

    return float(r_hat), np.asarray(n_hat, dtype=np.float32)


def evaluate_option(
    env: Any,
    action: int,
    *,
    horizon: Optional[int],
    gamma: float,
) -> Tuple[float, np.ndarray]:
    """`run_option` on a deep copy of `env`. Never touches the live env."""
    return run_option(copy.deepcopy(env), action, horizon=horizon, gamma=gamma)


def execute_option(
    env: Any,
    action: int,
    *,
    horizon: Optional[int],
    gamma: float,
    baseline_action: int = 0,
) -> Tuple[float, np.ndarray]:
    """Run `action` on the LIVE env and return its realized (delta_r, delta_n).

    The baseline is rolled out on a copy from the same starting state over the
    same horizon, so the realized improvement is measured against what doing
    nothing would have produced right here -- not against an average over some
    other part of the episode. Advances `env`.
    """
    baseline_payoff, baseline_motives = evaluate_option(
        env, baseline_action, horizon=horizon, gamma=gamma
    )
    payoff, motives = run_option(env, action, horizon=horizon, gamma=gamma)
    delta_n = np.asarray(motives - baseline_motives, dtype=np.float64)
    return float(payoff - baseline_payoff), delta_n


def evaluate_candidates(
    env: Any,
    *,
    gamma: float,
    horizon: Optional[int],
    baseline_action: int = 0,
) -> Tuple[Dict[str, Any], List[CandidateSkillRecord]]:
    """Evaluate the baseline and every other action from env's CURRENT state.

    Every rollout starts from the same state (and the same RNG state), so the
    baseline and each candidate are compared under identical conditions. The
    caller decides when to reset.

    Returns (baseline_stats, records): `baseline_stats` carries the keys
    `ImprovementCalculator` requires; `records` has one PDS candidate per
    non-baseline action.
    """
    baseline_payoff, baseline_motives = evaluate_option(
        env, baseline_action, horizon=horizon, gamma=gamma
    )
    baseline_stats: Dict[str, Any] = {
        "baseline_payoff": baseline_payoff,
        "baseline_motives": baseline_motives,
    }
    calculator = ImprovementCalculator(baseline_stats)

    records: List[CandidateSkillRecord] = []
    for action, skill_id in enumerate(SKILL_NAMES):
        if action == baseline_action:
            continue
        payoff, motives = evaluate_option(env, action, horizon=horizon, gamma=gamma)
        delta_r, delta_n = calculator.compute_improvements(
            skill_payoff=payoff,
            skill_motives=motives,
        )
        records.append(
            CandidateSkillRecord(
                skill_id=skill_id,
                delta_r=float(delta_r),
                delta_n=tuple(float(v) for v in delta_n),
                is_certified=False,
                gate_type="PDS",
                metadata={"action": action},
            )
        )
    return baseline_stats, records

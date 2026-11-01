"""Check plan certificates against what the plans really achieve.

A certificate rests on one-sided lower confidence bounds of a plan's mean
gains, measured on the seeds the effect table was built from. Feedback records
the gains a plan really achieved when it was flown later, paired against doing
nothing from the same start, and compares them with what was certified.

A certificate is revoked when the realised gains are significantly below it:
for Δr or any motive gain, the one-sided upper confidence bound of the realised
mean falls below the certified value. The runtime skips plans whose
certificates are revoked until the table is rebuilt. These checks are not
corrected for being several, which errs towards revoking: a plan is skipped a
little more often than the confidence level alone would say, never less.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from decomposition.domain import Domain
from decomposition.effects import (
    CONFIDENCE,
    Gains,
    gain_names,
    group_key,
    idle_rewards,
    improvement,
    lower_bound,
    replay,
)
from decomposition.verifier import PlanVerdict, plan_skill_id

MIN_SAMPLES = 30
REVOKED_REASON = "certificate revoked: realised gains fall short of it"


class CertificateStatus(str, Enum):
    HELD = "held"
    REVOKED = "revoked"
    TOO_FEW_SAMPLES = "too_few_samples"


def realised_gains(domain: Domain, env: Any, seed: int, rewards: np.ndarray) -> Gains:
    """Gains of a run flown from the start of episode ``seed`` over doing nothing from that start.

    Replaying the seed resets ``env``.
    """
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1, domain.num_objectives)
    replay(env, seed, ())
    delta_r, delta_n = improvement(rewards, idle_rewards(domain, env, len(rewards)))
    return Gains(delta_r, delta_n)


@dataclass(frozen=True)
class Outcome:
    """What one certified plan really achieved from one state."""

    plan: str
    group: str
    certified: Gains
    realised: Gains


def _vector(gains: Gains) -> np.ndarray:
    return np.array([gains.delta_r, *gains.delta_n], dtype=np.float64)


def _upper_bound(values: np.ndarray, seed: int) -> float:
    return -lower_bound(-values, seed)


class FeedbackLog:
    """Realised outcomes of certified plans, grouped by plan and start state."""

    def __init__(self, domain: Domain, min_samples: int = MIN_SAMPLES) -> None:
        if min_samples < 2:
            raise ValueError(f"min_samples must be at least 2, got {min_samples}")
        self.domain = domain
        self.min_samples = min_samples
        self._outcomes: Dict[Tuple[str, str], List[Outcome]] = {}
        self._report: Optional[List[Dict[str, object]]] = None  # cleared whenever an outcome is recorded

    def record(self, verdict: PlanVerdict, facts: Mapping[str, bool], realised: Gains) -> Outcome:
        """Record what an admitted plan, certified from a state with ``facts``, really achieved."""
        if not verdict.admitted:
            raise ValueError("only admitted plans have certificates to check")
        certificate = verdict.entry.certificate
        outcome = Outcome(
            plan=plan_skill_id(verdict.candidate),
            group=group_key(self.domain, facts),
            certified=Gains(certificate.delta_r, tuple(certificate.delta_n)),
            realised=realised,
        )
        self._outcomes.setdefault((outcome.plan, outcome.group), []).append(outcome)
        self._report = None
        return outcome

    def report(self) -> List[Dict[str, object]]:
        """One entry per plan and start group: certified values, realised means and bounds, status."""
        if self._report is None:
            self._report = [self._summary(outcomes) for _, outcomes in sorted(self._outcomes.items())]
        return [dict(entry) for entry in self._report]

    def revoked(self) -> FrozenSet[Tuple[str, str]]:
        """(plan, group) pairs whose certificates are revoked."""
        return frozenset(
            (entry["plan"], entry["group"])
            for entry in self.report()
            if entry["status"] == CertificateStatus.REVOKED.value
        )

    def screen(self, verdicts: Sequence[PlanVerdict], facts: Mapping[str, bool]) -> Tuple[PlanVerdict, ...]:
        """The verdicts with revoked plans turned into rejections, order kept."""
        revoked, group = self.revoked(), group_key(self.domain, facts)
        return tuple(
            replace(verdict, reason=REVOKED_REASON, entry=None, score=None)
            if verdict.admitted and (plan_skill_id(verdict.candidate), group) in revoked
            else verdict
            for verdict in verdicts
        )

    def _summary(self, outcomes: Sequence[Outcome]) -> Dict[str, object]:
        first = outcomes[0]
        certified = _vector(first.certified)
        realised = np.stack([_vector(outcome.realised) for outcome in outcomes])
        enough = len(outcomes) >= self.min_samples
        components = {}
        short = False
        for index, name in enumerate(gain_names(self.domain.objectives)):
            column = realised[:, index]
            upper = _upper_bound(column, seed=index) if enough else None
            short = short or (upper is not None and upper < certified[index])
            components[name] = {
                "certified": float(certified[index]),
                "realised_mean": float(column.mean()),
                "realised_upper": upper,
            }
        if not enough:
            status = CertificateStatus.TOO_FEW_SAMPLES
        else:
            status = CertificateStatus.REVOKED if short else CertificateStatus.HELD
        return {
            "plan": first.plan,
            "group": first.group,
            "samples": len(outcomes),
            "confidence": CONFIDENCE,
            "status": status.value,
            **components,
        }

"""Check plan certificates against what the plans really achieve.

A certificate rests on one-sided lower confidence bounds of a plan's mean
gains, measured on the seeds the effect table was built from. Feedback records
the gains a plan really achieved when it was flown later, paired against doing
nothing from the same state, and compares them with what was certified.

A certificate is revoked when the realised gains are significantly below it:
for Δr or any motive gain, the one-sided upper confidence bound of the realised
mean falls below the certified value. Flights are grouped by the region of
states their certificate was measured from: a group of states, or one cell of a
group split into cells. The flights of a plan from a region judge every
certificate for that plan from that region, not only the one they were flown
under. Given the log, the runtime therefore skips a revoked plan
until new evidence approves it: a rebuilt table whose certificate claims no
more than the flights support. The report keeps every certificate it judged, for
audit. These checks are not corrected for being several, which errs towards
revoking: a plan is skipped a little more often than the confidence level alone
would say, never less.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from decomposition.domain import Domain
from decomposition.effects import (
    CONFIDENCE,
    Gains,
    gain_names,
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


def realised_gains(
    domain: Domain, env: Any, seed: int, rewards: np.ndarray, start: Sequence[int] = ()
) -> Gains:
    """Gains of a run flown from the state ``start`` reaches in episode ``seed`` over doing nothing from there.

    Replaying the seed resets ``env``.
    """
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1, domain.num_objectives)
    replay(env, seed, start)
    delta_r, delta_n = improvement(rewards, idle_rewards(domain, env, len(rewards)), domain.gamma)
    return Gains(delta_r, delta_n)


@dataclass(frozen=True)
class Outcome:
    """What one certified plan really achieved from one state."""

    plan: str
    region: str  # the states the certificate was measured from, as the estimate names them
    certified: Gains
    realised: Gains


def _vector(gains: Gains) -> np.ndarray:
    return np.array([gains.delta_r, *gains.delta_n], dtype=np.float64)


def _upper_bound(values: np.ndarray, seed: int) -> float:
    return -lower_bound(-values, seed)


def _certified(verdict: PlanVerdict) -> Gains:
    certificate = verdict.entry.certificate
    return Gains(certificate.delta_r, tuple(certificate.delta_n))


class FeedbackLog:
    """Realised outcomes of certified plans, grouped by plan and the region of states certified from."""

    def __init__(self, domain: Domain, min_samples: int = MIN_SAMPLES) -> None:
        if min_samples < 2:
            raise ValueError(f"min_samples must be at least 2, got {min_samples}")
        self.domain = domain
        self.min_samples = min_samples
        self._outcomes: Dict[Tuple[str, str], List[Outcome]] = {}
        # Both are cleared whenever an outcome is recorded.
        self._report: Optional[List[Dict[str, object]]] = None
        self._upper: Dict[Tuple[str, str], Optional[np.ndarray]] = {}

    def record(self, verdict: PlanVerdict, realised: Gains) -> Outcome:
        """Record what an admitted plan really achieved from the state it was certified from."""
        if not verdict.admitted:
            raise ValueError("only admitted plans have certificates to check")
        outcome = Outcome(
            plan=plan_skill_id(verdict.candidate),
            region=verdict.estimate.region,
            certified=_certified(verdict),
            realised=realised,
        )
        key = (outcome.plan, outcome.region)
        self._outcomes.setdefault(key, []).append(outcome)
        self._report = None
        self._upper.pop(key, None)
        return outcome

    def report(self) -> List[Dict[str, object]]:
        """One entry per plan and region.

        Each entry judges the latest certificate recorded, with its certified
        values and the realised means and bounds, and lists every certificate
        recorded for the plan and region, in order, with how it stands now.
        """
        if self._report is None:
            self._report = [self._summary(key, outcomes) for key, outcomes in sorted(self._outcomes.items())]
        return [dict(entry) for entry in self._report]

    def revoked(self) -> FrozenSet[Tuple[str, str]]:
        """(plan, region) pairs whose latest certificates are revoked."""
        return frozenset(
            (entry["plan"], entry["region"])
            for entry in self.report()
            if entry["status"] == CertificateStatus.REVOKED.value
        )

    def status(self, plan: str, region: str, certified: Gains) -> CertificateStatus:
        """How a certificate claiming ``certified`` for ``plan`` from ``region`` stands against the flights recorded."""
        upper = self._upper_bounds((plan, region))
        if upper is None:
            return CertificateStatus.TOO_FEW_SAMPLES
        return CertificateStatus.REVOKED if bool(np.any(upper < _vector(certified))) else CertificateStatus.HELD

    def screen(self, verdicts: Sequence[PlanVerdict]) -> Tuple[PlanVerdict, ...]:
        """The verdicts with revoked certificates turned into rejections, order kept."""
        return tuple(
            replace(verdict, reason=REVOKED_REASON, entry=None, score=None)
            if verdict.admitted and self.is_revoked(verdict)
            else verdict
            for verdict in verdicts
        )

    def is_revoked(self, verdict: PlanVerdict) -> bool:
        """Whether the flights recorded for the verdict's plan from its region fall short of what it claims.

        An admitted verdict claims its certificate. A verdict refused after its plan
        was measured claims the lower bounds a certificate would have carried, so a
        refused plan cannot go on under evidence the flights have already revoked. A
        plan never measured from such a state claims nothing and is never revoked.
        """
        if verdict.estimate is None:
            return False
        claimed = _certified(verdict) if verdict.admitted else verdict.estimate.lower
        status = self.status(plan_skill_id(verdict.candidate), verdict.estimate.region, claimed)
        return status is CertificateStatus.REVOKED

    def _upper_bounds(self, key: Tuple[str, str]) -> Optional[np.ndarray]:
        """Upper confidence bounds of the realised mean gains, or None with too few flights to tell."""
        if key not in self._upper:
            outcomes = self._outcomes.get(key, [])
            upper = None
            if len(outcomes) >= self.min_samples:
                realised = np.stack([_vector(outcome.realised) for outcome in outcomes])
                upper = np.array([_upper_bound(realised[:, index], seed=index) for index in range(realised.shape[1])])
            self._upper[key] = upper
        return self._upper[key]

    def _summary(self, key: Tuple[str, str], outcomes: Sequence[Outcome]) -> Dict[str, object]:
        plan, region = key
        latest = _vector(outcomes[-1].certified)
        realised = np.stack([_vector(outcome.realised) for outcome in outcomes])
        upper = self._upper_bounds(key)
        components = {
            name: {
                "certified": float(latest[index]),
                "realised_mean": float(realised[:, index].mean()),
                "realised_upper": None if upper is None else float(upper[index]),
            }
            for index, name in enumerate(gain_names(self.domain.objectives))
        }
        certificates = list(dict.fromkeys(outcome.certified for outcome in outcomes))
        return {
            "plan": plan,
            "region": region,
            "samples": len(outcomes),
            "confidence": CONFIDENCE,
            "status": self.status(plan, region, outcomes[-1].certified).value,
            **components,
            "certificates": [
                {"certified": _vector(gains).tolist(), "status": self.status(plan, region, gains).value}
                for gains in certificates
            ],
        }

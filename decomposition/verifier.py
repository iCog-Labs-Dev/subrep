"""Certify and rank candidate plans with SubRep's existing admission gates.

Each candidate is looked up in the effect table by the facts of the current
state, and the one-sided lower confidence bounds of its measured gains
(Δr, Δn) go through the gates the rest of SubRep uses:

- the CDS gate first, as the stronger guarantee;
- the PDS gate with the priorities' ε when CDS fails;
- when the priorities carry an MDN region and a CVaR tail level, the CVaR
  check must pass as well.

The worst case is taken over the full simplex of weightings, or over the MDN
region W_x when the priorities carry one. An admitted plan gets a standard
``Certificate`` inside a ``SkillEntry`` and is scored with
``library.skill_selector.score_skill_entry`` under the priorities' weights.
Plans are ranked by score; ties go to the plan with fewer steps, then to the
option name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from certification.cds_test import CDSGate
from certification.certificate_schema import Certificate
from certification.cvar_test import CVaRGate
from certification.pds_test import PDSGate
from decomposition.effects import EffectTable, Estimate, Gains
from decomposition.planner import CandidatePlan
from library.skill_metadata import FULL_SIMPLEX, MDN_WX, SkillEntry
from library.skill_selector import score_skill_entry

CERTIFICATE_VERSION = "decomposition-0.1.0"
CVAR_SAMPLES = 1000
CVAR_SEED = 0
_SIMPLEX_TOLERANCE = 1e-6


def _finite_array(name: str, values: object, ndim: int = 1) -> np.ndarray:
    """``values`` as a non-empty float array of ``ndim`` dimensions with no NaN or infinity."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only numbers, got {values!r}") from error
    if array.ndim != ndim or array.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array, got {values!r}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numbers, got {values!r}")
    return array


def _finite_number(name: str, value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


@dataclass(frozen=True)
class MdnRegion:
    """The MDN weight region W_x = {w in the simplex : w_i <= s_i} with its audit fields.

    Checked as the certificate schema checks an MDN region: every value finite,
    alpha positive, support values in [0, 1] and summing to at least 1 so the
    region is not empty, and one square row of directions per objective.
    """

    context: Tuple[float, ...]
    alpha: Tuple[float, ...]
    support_directions: Tuple[Tuple[float, ...], ...]
    support_values: Tuple[float, ...]

    def __post_init__(self) -> None:
        _finite_array("context", self.context)
        alpha = _finite_array("alpha", self.alpha)
        support = _finite_array("support_values", self.support_values)
        directions = _finite_array("support_directions", self.support_directions, ndim=2)
        if alpha.shape != support.shape or directions.shape != (support.size, support.size):
            raise ValueError("alpha, support_values and support_directions must cover the same objectives")
        if np.any(alpha <= 0.0):
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if np.any(support < 0.0) or np.any(support > 1.0):
            raise ValueError(f"support_values must lie in [0, 1], got {self.support_values}")
        if support.sum() < 1.0 - _SIMPLEX_TOLERANCE:
            raise ValueError(f"support_values must sum to at least 1 or the region is empty, got {self.support_values}")


@dataclass(frozen=True)
class Priorities:
    """How much each objective matters and how much risk is allowed."""

    weights: Tuple[float, ...]  # on the simplex over (Safety, Fuel); used for ranking
    pds_epsilon: float = 0.0
    cvar_tail_level: Optional[float] = None
    region: Optional[MdnRegion] = None

    def __post_init__(self) -> None:
        weights = _finite_array("weights", self.weights)
        if np.any(weights < -_SIMPLEX_TOLERANCE) or abs(weights.sum() - 1.0) > _SIMPLEX_TOLERANCE:
            raise ValueError(f"weights must lie on the simplex, got {self.weights}")
        if _finite_number("pds_epsilon", self.pds_epsilon) < 0.0:
            raise ValueError(f"pds_epsilon must be non-negative, got {self.pds_epsilon}")
        if self.region is not None:
            support = np.asarray(self.region.support_values, dtype=np.float64)
            if support.shape != weights.shape:
                raise ValueError("the MDN region and the weights must cover the same objectives")
            if np.any(weights > support + _SIMPLEX_TOLERANCE):
                raise ValueError(f"weights {self.weights} lie outside the MDN region {self.region.support_values}")
        if self.cvar_tail_level is not None:
            if not 0.0 < _finite_number("cvar_tail_level", self.cvar_tail_level) <= 1.0:
                raise ValueError(f"cvar_tail_level must be in (0, 1], got {self.cvar_tail_level}")
            if self.region is None:
                raise ValueError("the CVaR check needs an MDN region for its alpha")


@dataclass(frozen=True)
class PlanVerdict:
    """The outcome of certifying one candidate plan."""

    candidate: CandidatePlan
    reason: str
    estimate: Optional[Estimate] = None  # None when the plan was not measured from such a state
    cds_margin: Optional[float] = None
    pds_margin: Optional[float] = None
    cvar: Optional[float] = None
    entry: Optional[SkillEntry] = None  # the certified library entry; None when rejected
    score: Optional[float] = None

    @property
    def admitted(self) -> bool:
        return self.entry is not None

    def record(self) -> Dict[str, object]:
        """A JSON-ready record of the plan, its measured effect and the gate results."""
        candidate = self.candidate
        return {
            "skill_id": plan_skill_id(candidate),
            "goal": candidate.goal,
            "option": candidate.option,
            "skills": list(candidate.skills),
            "methods": [list(method) for method in candidate.methods],
            "admitted": self.admitted,
            "reason": self.reason,
            "estimate": None if self.estimate is None else _estimate_record(self.estimate),
            "gates": {"cds_margin": self.cds_margin, "pds_margin": self.pds_margin, "cvar": self.cvar},
            "score": self.score,
            "certificate": None if self.entry is None else self.entry.certificate.to_dict(),
        }


def plan_skill_id(candidate: CandidatePlan) -> str:
    return f"plan:{candidate.goal}/{candidate.option}"


def verify(
    candidates: Sequence[CandidatePlan],
    facts: Mapping[str, bool],
    table: EffectTable,
    priorities: Priorities,
    timestamp: Optional[str] = None,
) -> List[PlanVerdict]:
    """Certify every candidate from a state with ``facts``; admitted plans come first, best first."""
    timestamp = timestamp or datetime.now(timezone.utc).isoformat()
    verdicts = [
        _verify(candidate, table.plan(candidate.skills, facts), table, priorities, timestamp)
        for candidate in candidates
    ]
    return sorted(verdicts, key=_rank)


def best_plan(verdicts: Sequence[PlanVerdict]) -> Optional[PlanVerdict]:
    """The highest-ranked admitted plan, or None when no plan was admitted."""
    return next((verdict for verdict in verdicts if verdict.admitted), None)


def _verify(
    candidate: CandidatePlan,
    estimate: Optional[Estimate],
    table: EffectTable,
    priorities: Priorities,
    timestamp: str,
) -> PlanVerdict:
    if estimate is None:
        return PlanVerdict(candidate, reason="not measured from a state like this one")

    delta_r = estimate.lower.delta_r
    delta_n = np.asarray(estimate.lower.delta_n, dtype=np.float64)
    support = None if priorities.region is None else np.asarray(priorities.region.support_values, dtype=np.float64)
    cds, pds = CDSGate(), PDSGate(epsilon=priorities.pds_epsilon)
    cds_margin = cds.get_admission_margin(delta_r, delta_n, support_values=support)
    pds_margin = pds.get_admission_margin(delta_r, delta_n, support_values=support)
    cvar = _cvar(delta_r, delta_n, priorities)
    results = {"estimate": estimate, "cds_margin": cds_margin, "pds_margin": pds_margin, "cvar": cvar}

    if cds.admit(delta_r, delta_n, support_values=support):
        gate, margin, epsilon = "CDS", cds_margin, 0.0
    elif pds.admit(delta_r, delta_n, support_values=support):
        gate, margin, epsilon = "PDS", pds_margin, priorities.pds_epsilon
    else:
        return PlanVerdict(candidate, reason="rejected by the CDS and PDS gates", **results)
    if cvar is not None and cvar < 0.0:
        return PlanVerdict(candidate, reason="rejected by the CVaR check", **results)

    certificate = _certificate(candidate, estimate, gate, margin, epsilon, table, priorities, timestamp)
    entry = SkillEntry(
        skill_id=certificate.skill_id,
        gate_type=certificate.gate_type,
        certificate=certificate,
        weight_region_type=certificate.weight_region_type,
        certification_context=certificate.certification_context,
        mdn_alpha=certificate.mdn_alpha,
        wx_support_directions=certificate.wx_support_directions,
        wx_support_values=certificate.wx_support_values,
    )
    score = score_skill_entry(entry, np.asarray(priorities.weights, dtype=np.float64))
    return PlanVerdict(candidate, reason=f"admitted by the {gate} gate", entry=entry, score=score, **results)


def _cvar(delta_r: float, delta_n: np.ndarray, priorities: Priorities) -> Optional[float]:
    """CVaR of Δr + w·Δn for w drawn from the MDN's Dirichlet; reproducible and isolated from torch's RNG."""
    if priorities.cvar_tail_level is None:
        return None
    gate = CVaRGate(confidence=priorities.cvar_tail_level, n_samples=CVAR_SAMPLES)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(CVAR_SEED)
        return gate.get_cvar(delta_r, delta_n, np.asarray(priorities.region.alpha, dtype=np.float64))


def _certificate(
    candidate: CandidatePlan,
    estimate: Estimate,
    gate: str,
    margin: float,
    epsilon: float,
    table: EffectTable,
    priorities: Priorities,
    timestamp: str,
) -> Certificate:
    region = priorities.region
    return Certificate(
        skill_id=plan_skill_id(candidate),
        gate_type=gate,
        delta_r=estimate.lower.delta_r,
        delta_n=estimate.lower.delta_n,
        admission_margin=margin,
        epsilon=epsilon,
        timestamp=timestamp,
        seed=int(table.data["seeds"]["start"]),
        gamma=float(table.data["gamma"]),
        baseline_id=str(table.data["baseline_id"]),
        environment=str(table.data["environment"]),
        episode_length=max(1, int(round(estimate.mean_frames))),
        version=CERTIFICATE_VERSION,
        weight_region_type=FULL_SIMPLEX if region is None else MDN_WX,
        certification_context=None if region is None else region.context,
        mdn_alpha=None if region is None else region.alpha,
        wx_support_directions=None if region is None else region.support_directions,
        wx_support_values=None if region is None else region.support_values,
    )


def _rank(verdict: PlanVerdict) -> Tuple[int, float, int, str]:
    score = -verdict.score if verdict.admitted else 0.0
    return (0 if verdict.admitted else 1, score, len(verdict.candidate.skills), verdict.candidate.option)


def _gains_record(gains: Gains) -> Dict[str, object]:
    return {"delta_r": gains.delta_r, "delta_n": list(gains.delta_n)}


def _estimate_record(estimate: Estimate) -> Dict[str, object]:
    return {
        "samples": estimate.samples,
        "completion_rate": estimate.completion_rate,
        "mean_frames": estimate.mean_frames,
        "mean": _gains_record(estimate.mean),
        "lower": _gains_record(estimate.lower),
        "interaction": None if estimate.interaction is None else _gains_record(estimate.interaction),
    }

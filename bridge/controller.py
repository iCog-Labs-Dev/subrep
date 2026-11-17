"""Per-step orchestration of the MetaMo -> SubRep loop.

One `step()` does:
  1. read the governor's current signal (weights, epsilon, alpha),
  2. seed torch so CVaR sampling is reproducible,
  3. certify candidate skills with those per-step budgets,
  4. select among admitted skills using the governor's weights,
  5. feed the executed outcome back into the governor.

Determinism note: `CVaRGate.get_cvar` draws from `Dirichlet(...).sample()` on
the global torch RNG (certification/cvar_test.py:51) with no seeding of its
own, so certification is otherwise NOT reproducible run to run. The controller
seeds before each certification pass and records the seed it used.

Epsilon scale (optional, off by default): given an `EpsilonScale`
(bridge/epsilon_scale.py), the controller first rejects candidates that fail
the task, without sending them to the pipeline, then stamps the others with
the tolerance epsilon * unit instead of the raw epsilon. It fails closed: a
missing or non-finite payoff, or a record it cannot stamp and mark rejected,
raises instead of letting a candidate through. Without one, every call is
exactly as before.

`last_records` holds the records of the latest `certify()` call, so reports
show this step's decisions, including candidates rejected before the
pipeline ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .epsilon_scale import TASK_FAILURE, EpsilonScale
from .protocol import GovernorSignal, SkillOutcome


def _with_epsilon(candidate: Any, epsilon: float) -> Any:
    """Stamp this step's PDS budget onto a candidate record.

    `RuntimeCertificationPipeline.certify_candidate_skills` has no `epsilon`
    parameter -- it resolves the budget per candidate via
    `_candidate_epsilon`, which reads `candidate.epsilon` and falls back to the
    static `config.pds_epsilon` when that is None
    (utils/mdn_runtime_pipeline.py:301-309). So the only way the governor's
    per-step epsilon can reach the PDS gate is on the record itself.

    Duck-typed on purpose: callers may pass lightweight stand-ins that are not
    `CandidateSkillRecord`, and those are left untouched.
    """
    if is_dataclass(candidate) and any(f.name == "epsilon" for f in fields(candidate)):
        return replace(candidate, epsilon=float(epsilon))
    return candidate


# Fields the epsilon scale must be able to set on every candidate record.
_SCALED_RECORD_FIELDS = frozenset({"epsilon", "is_certified", "metadata"})


def _require_scalable(candidate: Any) -> None:
    """Refuse a record the epsilon scale cannot stamp and mark rejected.

    With the scale on, every candidate needs its tolerance stamped
    (`epsilon`), its approval cleared or set (`is_certified`) and, when it
    fails the task, its reason recorded (`metadata`). A record lacking any of
    these would keep whatever approval it arrived with, or reach the gate with
    the pipeline's static epsilon instead of the tolerance, so it is rejected
    up front rather than passed through.
    """
    names = {f.name for f in fields(candidate)} if is_dataclass(candidate) else set()
    missing = _SCALED_RECORD_FIELDS - names
    if missing:
        raise TypeError(
            "the epsilon scale needs candidate records it can stamp and mark "
            "rejected, such as CandidateSkillRecord: a dataclass with the fields "
            f"{sorted(_SCALED_RECORD_FIELDS)}; got {type(candidate).__name__}, "
            f"missing {sorted(missing)}"
        )


def _as_task_failure(candidate: Any) -> Any:
    """Mark a candidate rejected because it failed the task.

    Only called on records `_require_scalable` accepted, so the result is
    always unapproved, whatever approval the record arrived with.
    """
    metadata = dict(candidate.metadata or {})
    metadata["rejection_reason"] = TASK_FAILURE
    return replace(candidate, is_certified=False, metadata=metadata)


@dataclass
class StepRecord:
    """What happened during one controller step, for logging and assertions."""

    step: int
    weights: np.ndarray
    pds_epsilon: float
    cvar_tail_level: float
    modulators: Dict[str, float] = field(default_factory=dict)
    candidate_count: int = 0
    admitted_count: int = 0
    selected_skill_id: Optional[str] = None
    selected_score: Optional[float] = None
    seed: Optional[int] = None
    # Set only when the controller has an epsilon scale.
    pds_tolerance: Optional[float] = None
    task_failures: int = 0

    def as_row(self) -> str:
        """Compact single-line summary."""
        sel = self.selected_skill_id or "-"
        return (
            f"step={self.step:>3}  eps={self.pds_epsilon:.4f}  "
            f"alpha={self.cvar_tail_level:.4f}  "
            f"admitted={self.admitted_count}/{self.candidate_count}  "
            f"selected={sel}"
        )


def _seed_torch(seed: int) -> None:
    """Seed the global torch RNG that CVaRGate samples from."""
    import torch

    torch.manual_seed(seed)


class MetaMoController:
    """Couples a `MotivationalGovernor` to SubRep's certification pipeline."""

    def __init__(
        self,
        governor: Any,
        pipeline: Any,
        *,
        seed: int = 42,
        epsilon_scale: Optional[EpsilonScale] = None,
    ) -> None:
        self.governor = governor
        self.pipeline = pipeline
        self.seed = int(seed)
        self.epsilon_scale = epsilon_scale
        self.history: List[StepRecord] = []
        # The records the last `certify()` call returned: this step's
        # decisions, including candidates rejected before the pipeline.
        self.last_records: List[Any] = []
        self._step_index = 0

    def certify(
        self,
        *,
        context: np.ndarray,
        candidate_skills: Sequence[Any],
        baseline_stats: Dict[str, Any],
        signal: GovernorSignal,
        seed: Optional[int] = None,
    ) -> List[Any]:
        """Certify candidates under this step's budgets.

        Both budgets reach the gates, by two different routes:

        * `cvar_tail_level` is a call parameter -- the pipeline threads it
          down to `CVaRGate(confidence=...)`.
        * `pds_epsilon` has no call parameter, so it is stamped onto each
          candidate record instead; see `_with_epsilon`.

        The MDN's Dirichlet concentration is fetched internally by the
        pipeline and never travels through this call.

        With an epsilon scale, candidates that fail the task are returned
        rejected (`metadata["rejection_reason"] == TASK_FAILURE`) without
        reaching the pipeline, and the rest are stamped with the tolerance
        epsilon * unit and with their approval cleared, so the pipeline's
        verdict is the only approval that can come back. Every record must be
        one the scale can stamp (see `_require_scalable`); otherwise TypeError
        is raised before anything runs. Records come back in the input order
        either way, and are kept as `last_records`.
        """
        effective_seed = self.seed if seed is None else int(seed)
        _seed_torch(effective_seed)

        scale = self.epsilon_scale
        if scale is None:
            budgeted = [
                _with_epsilon(candidate, signal.pds_epsilon)
                for candidate in candidate_skills
            ]
            records = list(
                self.pipeline.certify_candidate_skills(
                    context=np.asarray(context),
                    candidate_skills=budgeted,
                    baseline_stats=baseline_stats,
                    weights_used=signal.weights,
                    cvar_confidence=signal.cvar_tail_level,
                )
            )
            self.last_records = records
            return records

        for candidate in candidate_skills:
            _require_scalable(candidate)
        tolerance = scale.tolerance(signal.pds_epsilon)
        failing = [scale.fails_task(c, baseline_stats) for c in candidate_skills]
        # A record that fails the gates comes back from the pipeline exactly as
        # it went in (utils/mdn_runtime_pipeline.py), so a stale approval is
        # cleared here, not trusted.
        passing = [
            replace(candidate, epsilon=float(tolerance), is_certified=False)
            for candidate, fails in zip(candidate_skills, failing)
            if not fails
        ]
        certified = iter(
            self.pipeline.certify_candidate_skills(
                context=np.asarray(context),
                candidate_skills=passing,
                baseline_stats=baseline_stats,
                weights_used=signal.weights,
                cvar_confidence=signal.cvar_tail_level,
            )
            if passing
            else []
        )
        records = [
            _as_task_failure(candidate) if fails else next(certified)
            for candidate, fails in zip(candidate_skills, failing)
        ]
        self.last_records = records
        return records

    def select(
        self,
        records: Sequence[Any],
        signal: GovernorSignal,
    ) -> Tuple[Optional[str], Optional[float]]:
        """Pick the best admitted skill under the governor's weights.

        Reuses SubRep's own scalarization rather than reimplementing it
        (library/skill_selector.py:21-46).
        """
        from library.skill_selector import score_skill_entry

        admitted = [r for r in records if getattr(r, "is_certified", False)]
        if not admitted:
            return None, None

        best_record = None
        best_score = None
        for record in admitted:
            score = score_skill_entry(record, signal.weights)
            if (
                best_score is None
                or score > best_score
                or (score == best_score and record.skill_id < best_record.skill_id)
            ):
                best_record = record
                best_score = score
        return best_record.skill_id, float(best_score)

    def step(
        self,
        *,
        context: np.ndarray,
        candidate_skills: Sequence[Any],
        baseline_stats: Dict[str, Any],
        outcome_for: Any,
        seed: Optional[int] = None,
    ) -> StepRecord:
        """Run one full governor -> gates -> selection -> feedback cycle.

        Args:
            outcome_for: Callable mapping the selected record (or None) to a
                `SkillOutcome`. Supplied by the caller so the controller stays
                independent of any particular environment.
        """
        signal = self.governor.signal()
        modulators = (
            self.governor.modulator_summary()
            if hasattr(self.governor, "modulator_summary")
            else {}
        )

        effective_seed = self.seed if seed is None else int(seed)
        records = self.certify(
            context=context,
            candidate_skills=candidate_skills,
            baseline_stats=baseline_stats,
            signal=signal,
            seed=effective_seed,
        )

        admitted = [r for r in records if getattr(r, "is_certified", False)]
        skill_id, score = self.select(records, signal)

        selected_record = next(
            (r for r in admitted if r.skill_id == skill_id), None
        )
        outcome: SkillOutcome = outcome_for(selected_record)
        self.governor.step(outcome)

        scale = self.epsilon_scale
        pds_tolerance = None if scale is None else scale.tolerance(signal.pds_epsilon)
        task_failures = (
            0
            if scale is None
            else sum(1 for c in candidate_skills if scale.fails_task(c, baseline_stats))
        )

        record = StepRecord(
            step=self._step_index,
            weights=np.asarray(signal.weights).copy(),
            pds_epsilon=signal.pds_epsilon,
            cvar_tail_level=signal.cvar_tail_level,
            modulators=dict(modulators),
            candidate_count=len(records),
            admitted_count=len(admitted),
            selected_skill_id=skill_id,
            selected_score=score,
            seed=effective_seed,
            pds_tolerance=pds_tolerance,
            task_failures=task_failures,
        )
        self.history.append(record)
        self._step_index += 1
        return record

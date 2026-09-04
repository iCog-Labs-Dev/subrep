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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .protocol import GovernorSignal, SkillOutcome


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
    ) -> None:
        self.governor = governor
        self.pipeline = pipeline
        self.seed = int(seed)
        self.history: List[StepRecord] = []
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

        Both budgets are passed explicitly. `cvar_confidence` is the scalar
        tail level; the MDN's Dirichlet concentration is fetched internally by
        the pipeline and never travels through this call.
        """
        effective_seed = self.seed if seed is None else int(seed)
        _seed_torch(effective_seed)

        return self.pipeline.certify_candidate_skills(
            context=np.asarray(context),
            candidate_skills=list(candidate_skills),
            baseline_stats=baseline_stats,
            weights_used=signal.weights,
            cvar_confidence=signal.cvar_tail_level,
        )

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
        )
        self.history.append(record)
        self._step_index += 1
        return record

"""Put MetaMo's PDS budget into the environment's delta units.

MetaMo emits epsilon as a dimensionless number in [0, epsilon_max]
(bridge/budget.py). The PDS gate compares it directly with
delta_r + min(delta_n) (certification/pds_test.py), which is in whatever units
the environment's rewards use. When those deltas are in the hundreds, as on
MO-LunarLander, no epsilon MetaMo can emit changes an admission.

`EpsilonScale` fixes the units and adds one guard:

* Fixed unit. The PDS tolerance becomes epsilon * unit, so epsilon reads as a
  fraction of a typical candidate change. The unit is measured once, on
  calibration seeds, and stays fixed. delta_r and delta_n are not rescaled, so
  selection scores and stored certificates keep their meaning; only the
  admission line moves. Normalising per step would make one candidate's
  admission depend on which other candidates are present, and normalising per
  objective would change the selection scores.

* Task-failure guard. A candidate whose own payoff (baseline payoff + delta_r)
  is below `payoff_floor` fails the task and is rejected at every epsilon,
  before certification. Under any tolerance large enough to matter, a failing
  candidate can sit inside the band: on MO-LunarLander, crashes against a
  crashing baseline have margins within 1-40 units of zero. The guard keeps
  crash safety a rule instead of a side effect of the units, and because it
  acts before the pipeline, a failing candidate never enters the certificate
  cache or grows W_x. Guard rejections are reported apart from PDS rejections.

No MetaMo import and no fixed objective count: the scale works on any record
with `delta_r` and any `baseline_stats` with a "baseline_payoff" entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

# Written to a rejected record's metadata["rejection_reason"].
TASK_FAILURE = "task_failure"


@dataclass(frozen=True)
class EpsilonScale:
    """Turns MetaMo's dimensionless epsilon into a PDS tolerance.

    Attributes:
        unit: U > 0, in the environment's delta units. The PDS tolerance is
            epsilon * U.
        payoff_floor: When set, a candidate whose own payoff (baseline payoff +
            delta_r) is below it fails the task and is rejected at every
            epsilon. None turns the guard off.
    """

    unit: float
    payoff_floor: Optional[float] = None

    def __post_init__(self) -> None:
        unit = float(self.unit)
        if not math.isfinite(unit) or unit <= 0.0:
            raise ValueError(f"unit must be finite and positive, got {self.unit}")
        object.__setattr__(self, "unit", unit)
        if self.payoff_floor is not None:
            floor = float(self.payoff_floor)
            if not math.isfinite(floor):
                raise ValueError(f"payoff_floor must be finite or None, got {self.payoff_floor}")
            object.__setattr__(self, "payoff_floor", floor)

    def tolerance(self, epsilon: float) -> float:
        """The PDS tolerance for `epsilon`, in delta units."""
        return float(epsilon) * self.unit

    def fails_task(self, candidate: Any, baseline_stats: Mapping[str, Any]) -> bool:
        """Whether `candidate` fails the task, so must be rejected at every epsilon.

        False when the guard is off (no `payoff_floor`). With the guard on, the
        candidate's own payoff must be measurable: a missing or non-finite
        "baseline_payoff", or a non-finite `delta_r`, raises ValueError rather
        than letting the candidate through, because a payoff that cannot be
        compared with the floor must not read as a pass.
        """
        if self.payoff_floor is None:
            return False
        if "baseline_payoff" not in baseline_stats:
            raise ValueError(
                "the task-failure guard needs baseline_stats['baseline_payoff'] "
                "to measure each candidate's own payoff; none was given"
            )
        baseline_payoff = float(baseline_stats["baseline_payoff"])
        if not math.isfinite(baseline_payoff):
            raise ValueError(
                f"the task-failure guard needs a finite baseline payoff, got {baseline_payoff}"
            )
        delta_r = float(candidate.delta_r)
        if not math.isfinite(delta_r):
            name = getattr(candidate, "skill_id", type(candidate).__name__)
            raise ValueError(
                f"the task-failure guard needs a finite delta_r, got {delta_r} for {name!r}"
            )
        return baseline_payoff + delta_r < self.payoff_floor

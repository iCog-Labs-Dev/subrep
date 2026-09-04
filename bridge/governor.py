"""The MetaMo adapter -- the ONLY module in SubRep that imports MetaMo.

Confining MetaMo imports here is deliberate. MetaMo's own `usecase/` code
imports `metamo.core` / `metamo.state` (usecase/agents/metamo_agent.py:10,
usecase/simulation/runner.py:27) while the repository root actually exposes
`core/`, `category/`, `dynamics/` -- there is no `metamo/` package. Upstream's
import surface has demonstrably churned, so a future rename should be a
one-file fix rather than a repo-wide one.

The import is lazy: this module can be imported, and `FakeGovernor` used,
without a MetaMo checkout present.

MetaMo is never modified. Only its public API is used:
  * MotivationalState / Stimulus / Action   (core/state.py)
  * MetaMoPseudoBimonad.step()              (category/bimonad.py:153)
  * OpenPsiAppraisal, MagusDecision         injected via the constructor at
                                            category/bimonad.py:36
`step()` is pure -- it takes a state and returns a new one, never mutating
`self` -- which is what makes zero-change integration possible.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from . import budget as budget_mod
from . import weights as weights_mod
from ._loader import ensure_metamo_on_path, is_available
from .protocol import GovernorSignal, SkillOutcome
from .stimulus import build_stimulus_values

__all__ = ["MetaMoGovernor", "FakeGovernor", "is_available"]

# Populated on first use by _import_metamo().
_metamo: Optional[dict] = None


def _import_metamo() -> dict:
    """Import MetaMo symbols once, after putting its root on sys.path."""
    global _metamo
    if _metamo is not None:
        return _metamo

    ensure_metamo_on_path()

    from category.bimonad import MetaMoPseudoBimonad  # noqa: E402
    from core.config import (  # noqa: E402
        M_APPROACH,
        M_SECURING,
        M_THRESHOLD,
        NUM_GOALS,
        NUM_MODULATORS,
    )
    from core.state import Action, MotivationalState, Stimulus  # noqa: E402
    from magus.decision import MagusDecision  # noqa: E402
    from openpsi.appraisal import OpenPsiAppraisal  # noqa: E402

    _metamo = {
        "MetaMoPseudoBimonad": MetaMoPseudoBimonad,
        "MotivationalState": MotivationalState,
        "Stimulus": Stimulus,
        "Action": Action,
        "MagusDecision": MagusDecision,
        "OpenPsiAppraisal": OpenPsiAppraisal,
        "NUM_GOALS": NUM_GOALS,
        "NUM_MODULATORS": NUM_MODULATORS,
        "M_APPROACH": M_APPROACH,
        "M_SECURING": M_SECURING,
        "M_THRESHOLD": M_THRESHOLD,
    }
    return _metamo


def default_goal_vector(num_goals: int) -> np.ndarray:
    """Starting goal intensities.

    Mirrors MetaMo's own `_default_goal_vector` (core/engine.py:47-58). It is
    replicated rather than imported because `core.engine` pulls in `llm.*`,
    which needs API clients we do not want as a dependency.
    """
    g = np.zeros(num_goals, dtype=np.float64)
    # Index order per core/config.py:1-3.
    defaults = (0.5, 0.5, 0.8, 0.6, 0.4, 0.3, 0.9, 0.2)
    g[: min(num_goals, len(defaults))] = defaults[:num_goals]
    return g


class MetaMoGovernor:
    """Drives SubRep's weights and risk budgets from MetaMo's state.

    Each `step()` appraises the last skill outcome, advances the motivational
    state through the pseudo-bimonad F = D o Psi, and emits a fresh
    `GovernorSignal`.
    """

    def __init__(
        self,
        *,
        coefficients: Optional[budget_mod.BudgetCoefficients] = None,
        goal_affinity: Optional[np.ndarray] = None,
        modulator_gain: Optional[np.ndarray] = None,
        temperature: float = weights_mod.DEFAULT_TEMPERATURE,
        cvar_samples: int = 1000,
        candidates: Optional[Sequence[Any]] = None,
        initial_goals: Optional[np.ndarray] = None,
        payoff_scale: float = 1.0,
        motive_scale: float = 1.0,
    ) -> None:
        mm = _import_metamo()

        self._mm = mm
        self._coefficients = coefficients or budget_mod.BudgetCoefficients()
        self._goal_affinity = (
            weights_mod.DEFAULT_GOAL_AFFINITY if goal_affinity is None
            else np.asarray(goal_affinity, dtype=np.float64)
        )
        self._modulator_gain = (
            weights_mod.DEFAULT_MODULATOR_GAIN if modulator_gain is None
            else np.asarray(modulator_gain, dtype=np.float64)
        )
        self._temperature = float(temperature)
        self._cvar_samples = int(cvar_samples)

        # Characteristic magnitudes of delta_r / delta_n in the host
        # environment. These matter: the stimulus builder squashes through
        # tanh, so leaving them at 1.0 when the environment reports deltas of
        # order 10 saturates risk to ~1.0 on the first step and pins the
        # modulators at their bounds, flattening the whole coupling.
        if payoff_scale <= 0.0:
            raise ValueError(f"payoff_scale must be positive, got {payoff_scale}")
        if motive_scale <= 0.0:
            raise ValueError(f"motive_scale must be positive, got {motive_scale}")
        self._payoff_scale = float(payoff_scale)
        self._motive_scale = float(motive_scale)

        num_goals = int(mm["NUM_GOALS"])
        num_modulators = int(mm["NUM_MODULATORS"])

        if self._goal_affinity.shape[1] != num_goals:
            raise ValueError(
                f"goal_affinity has {self._goal_affinity.shape[1]} columns but "
                f"MetaMo exposes {num_goals} goals"
            )
        if self._modulator_gain.shape[1] != num_modulators:
            raise ValueError(
                f"modulator_gain has {self._modulator_gain.shape[1]} columns but "
                f"MetaMo exposes {num_modulators} modulators"
            )

        goals = (
            default_goal_vector(num_goals) if initial_goals is None
            else np.asarray(initial_goals, dtype=np.float64).reshape(-1)
        )
        if goals.size != num_goals:
            raise ValueError(
                f"initial_goals must have length {num_goals}, got {goals.size}"
            )

        # Modulators start at the neutral point (core/engine.py:81), which is
        # also the fixed point of the appraisal sigmoid (openpsi/appraisal.py:97).
        modulators = np.full(num_modulators, budget_mod.MODULATOR_NEUTRAL)

        self._bimonad = mm["MetaMoPseudoBimonad"](
            mm["OpenPsiAppraisal"](), mm["MagusDecision"]()
        )
        self._state = mm["MotivationalState"](G=goals, M=modulators)
        self._candidates = (
            list(candidates) if candidates is not None
            else self._default_candidates(num_goals)
        )
        self._last_action: Optional[Any] = None

    # -- introspection -----------------------------------------------------

    @property
    def num_objectives(self) -> int:
        return int(self._goal_affinity.shape[0])

    @property
    def state(self) -> Any:
        """Current MotivationalState. Read-only by convention."""
        return self._state

    @property
    def last_action_id(self) -> Optional[str]:
        return None if self._last_action is None else str(self._last_action.id)

    def modulator_summary(self) -> dict:
        """The three modulators that drive the risk budgets, for logging."""
        mm = self._mm
        m = self._state.M
        return {
            "securing": float(m[mm["M_SECURING"]]),
            "threshold": float(m[mm["M_THRESHOLD"]]),
            "approach": float(m[mm["M_APPROACH"]]),
        }

    # -- the protocol ------------------------------------------------------

    def signal(self) -> GovernorSignal:
        """Current signal, without advancing the motivational state."""
        mods = self.modulator_summary()
        epsilon, tail_level = budget_mod.compute_budgets(
            securing=mods["securing"],
            threshold=mods["threshold"],
            approach=mods["approach"],
            coefficients=self._coefficients,
            n_samples=self._cvar_samples,
        )
        w = weights_mod.w_meta(
            self._state.G,
            self._state.M,
            goal_affinity=self._goal_affinity,
            modulator_gain=self._modulator_gain,
            temperature=self._temperature,
        )
        return GovernorSignal(
            weights=w,
            pds_epsilon=epsilon,
            cvar_tail_level=tail_level,
        )

    def step(self, outcome: SkillOutcome) -> GovernorSignal:
        """Appraise an outcome, advance the state, return the new signal."""
        mm = self._mm

        values = build_stimulus_values(
            outcome,
            weights=self.signal().weights,
            payoff_scale=self._payoff_scale,
            motive_scale=self._motive_scale,
        )
        stimulus = mm["Stimulus"](
            novelty=values.novelty,
            conduciveness=values.conduciveness,
            risk=values.risk,
            effort=values.effort,
        )

        # bimonad.step is pure: it returns a new state and never mutates self
        # or the state passed in (category/bimonad.py:153-171).
        action, next_state = self._bimonad.step(
            self._state, stimulus, self._candidates
        )
        self._state = next_state
        self._last_action = action
        return self.signal()

    # -- candidates --------------------------------------------------------

    def _default_candidates(self, num_goals: int) -> List[Any]:
        """Motivational archetypes for MetaMo's decision monad.

        These are NOT SubRep skills. MetaMo's `step()` requires candidate
        actions in order to evolve the goal vector; SubRep's actual option
        choice happens separately, via the emitted weights. Three broad
        postures are enough to let G drift coherently.
        """
        mm = self._mm
        Action = mm["Action"]

        def make(
            action_id: str,
            emphasis: dict,
            risk_estimate: float,
        ) -> Any:
            correlations = np.full(num_goals, 0.1, dtype=np.float64)
            delta_g = np.zeros(num_goals, dtype=np.float64)
            for idx, value in emphasis.items():
                if idx < num_goals:
                    correlations[idx] = value
                    # Small nudges: MetaMo damps goal updates near the safety
                    # boundary (dynamics/stability.py:111-119) and we want the
                    # contractivity machinery to stay in charge.
                    delta_g[idx] = 0.02 * value
            return Action(
                id=action_id,
                goal_correlations=correlations,
                risk_estimate=risk_estimate,
                delta_g=delta_g,
            )

        # Goal indices per core/config.py:1-3.
        g_ind, g_trans, g_help, g_curio, g_novel, g_self, g_ethic, g_soc = range(8)

        return [
            make("secure", {g_ind: 0.9, g_ethic: 0.7, g_self: 0.4}, 0.05),
            make("explore", {g_curio: 0.9, g_novel: 0.8, g_trans: 0.6}, 0.45),
            make("engage", {g_soc: 0.9, g_help: 0.8, g_ethic: 0.5}, 0.20),
        ]


class FakeGovernor:
    """Scripted governor for tests and for exercising SubRep without MetaMo."""

    def __init__(
        self,
        signals: Sequence[GovernorSignal],
        *,
        repeat_last: bool = True,
    ) -> None:
        if not signals:
            raise ValueError("FakeGovernor requires at least one signal")
        self._signals = list(signals)
        self._repeat_last = repeat_last
        self._index = 0
        self.observed: List[SkillOutcome] = []

    @property
    def num_objectives(self) -> int:
        return self._signals[0].num_objectives

    def signal(self) -> GovernorSignal:
        return self._signals[min(self._index, len(self._signals) - 1)]

    def step(self, outcome: SkillOutcome) -> GovernorSignal:
        self.observed.append(outcome)
        if self._index < len(self._signals) - 1:
            self._index += 1
        elif not self._repeat_last:
            raise IndexError("FakeGovernor exhausted its scripted signals")
        return self.signal()

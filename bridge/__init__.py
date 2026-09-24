"""MetaMo <-> SubRep coupling layer.

All coupling logic lives here: the motivational governor adapter, the
w_meta projection, and the epsilon/alpha budget formulas. Nothing in this
package is pushed upstream into MetaMo, which is consumed read-only.

Only `bridge.governor` imports MetaMo symbols. Everything else is pure and
testable without MetaMo present.
"""

from .protocol import GovernorSignal, MotivationalGovernor, SkillOutcome

__all__ = ["GovernorSignal", "MotivationalGovernor", "SkillOutcome"]

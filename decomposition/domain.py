"""What goal decomposition needs to know about an environment.

The planner, the effect measurements, the verifier, the executor, the runtime
and the feedback log work only through a ``Domain``, so they do not depend on
any environment. Supporting a new environment means writing a recipe book and
one ``Domain`` for it: how to read facts from an observation, which skills exist
and how to run them, which objectives its reward vector holds, and how to judge
how an episode ended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

Facts = Dict[str, bool]
GAMMA = 0.99


@dataclass(frozen=True)
class Domain:
    """An environment as goal decomposition sees it."""

    environment: str
    objectives: Tuple[str, ...]  # the order of the reward vector, for example ("safety", "fuel")
    fact_names: Tuple[str, ...]
    group_facts: Tuple[str, ...]  # measurements are grouped by these facts of their start state
    outcomes: Tuple[str, ...]  # every way an episode can end
    idle_action: int
    skills: Mapping[str, Any]  # name -> skill with ``policy`` and ``is_done``
    settling_fact: str  # a plan that ends with this fact true keeps running until the episode ends
    facts: Callable[[np.ndarray], Facts]
    run_skill: Callable[[Any, Any, np.ndarray], Any]  # (env, skill, observation) -> the skill's run
    outcome: Callable[[Mapping[str, Any], np.ndarray], str]  # (final info, final observation) -> outcome
    final_facts: Callable[[Optional[Mapping[str, Any]], np.ndarray], Optional[Facts]]  # None unless confirmed
    make_env: Callable[[int], Any]
    gamma: float = GAMMA  # the discount of every measurement; its horizon should span the plans it compares

    @property
    def num_objectives(self) -> int:
        return len(self.objectives)

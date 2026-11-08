"""MO-LunarLander as a goal decomposition domain."""

from __future__ import annotations

from collections import ChainMap
from typing import Any, Mapping, Optional

import numpy as np

from decomposition.domain import Domain, Facts
from env.lunar_facts import FACT_NAMES, OUTCOMES, landing_outcome, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_pilots import landing_pilots
from env.lunar_subskills import NO_ENGINE, SUBSKILLS, run_subskill


def confirmed_final_facts(info: Optional[Mapping[str, Any]], observation: np.ndarray) -> Optional[Facts]:
    """The facts of the final state once the environment has confirmed a landing, else None.

    The environment's verdict stands in for leg contact, which flickers when the
    lander comes to rest.
    """
    if not (info and info.get("landing_success")):
        return None
    return {**lander_facts(observation), "landed": True}


LUNAR_LANDER = Domain(
    environment="mo-lunar-lander-v3",
    objectives=("safety", "fuel"),
    fact_names=FACT_NAMES,
    group_facts=("upright", "over_pad", "near_ground", "slow", "falling_fast"),
    outcomes=OUTCOMES,
    idle_action=NO_ENGINE,
    skills=ChainMap(SUBSKILLS, landing_pilots()),
    settling_fact="landed",
    facts=lander_facts,
    run_skill=run_subskill,
    outcome=landing_outcome,
    final_facts=confirmed_final_facts,
    make_env=lambda seed: SubRepEnv(seed=seed),
)

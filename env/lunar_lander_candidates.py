"""Trade-off candidates for MO-LunarLander: PPO that lands but burns extra fuel.

The default and extended candidate sets (data_collector/collect_candidate_sets.py)
mostly either land cleanly or crash, so their PDS margins sit far from zero and
no risk budget can decide them. These candidates fly the PPO pilot, but when
it would coast (action 0) they fire an extra engine with probability p. They
still land, while spending more fuel than the pilot, so their margin
delta_r + min(delta_n) falls close to zero: a tighter epsilon rejects them and
a looser one admits them.

Kept apart from data_collector/collect_candidate_sets.py, so SubRep's data
collection is unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from data_collector.collect_candidate_sets import CandidatePolicy
from pilot.rl_pilot import RLPilot

# The engine actions of mo-lunar-lander: 0 coast, 1 left, 2 main, 3 right.
COAST = 0
LEFT_ENGINE = 1
MAIN_ENGINE = 2
RIGHT_ENGINE = 3

# Offset for the per-episode RNG seed, so this candidate's draws do not
# coincide with any other seeded stream in an episode.
RNG_SEED_OFFSET = 20_000

MAIN_PROBABILITIES = (0.12, 0.15, 0.18, 0.20)
SIDE_PROBABILITIES = (0.15, 0.20)


class PpoWaste:
    """The PPO pilot, firing an extra engine with probability `p` when it would coast.

    `extra` is "main" (the main engine) or "side" (left and right in turn, so
    attitude stays roughly balanced). The RNG is re-seeded from the episode
    seed in `reset`, which `run_episode` calls before every episode
    (env/lunar_lander_rollout.py), so the same seed replays the same flight.
    """

    def __init__(self, pilot: RLPilot, p: float, extra: str) -> None:
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if extra not in ("main", "side"):
            raise ValueError(f"extra must be 'main' or 'side', got {extra!r}")
        self.pilot = pilot
        self.p = float(p)
        self.extra = extra
        self._rng = np.random.default_rng(RNG_SEED_OFFSET)
        self._left_next = True

    def reset(self, seed: Optional[int] = None, **_: object) -> None:
        self._rng = np.random.default_rng(RNG_SEED_OFFSET + int(seed or 0))
        self._left_next = True

    def __call__(self, obs: np.ndarray) -> Tuple[int, float]:
        result = self.pilot.predict(obs, deterministic=True, return_probability=True)
        action = int(result[0] if isinstance(result, tuple) else result)
        if action == COAST and self._rng.random() < self.p:
            if self.extra == "main":
                return MAIN_ENGINE, 1.0
            side = LEFT_ENGINE if self._left_next else RIGHT_ENGINE
            self._left_next = not self._left_next
            return side, 1.0
        return action, 1.0


def build_tradeoff_candidate_policies(
    *,
    pilot_checkpoint: str = "models/pilot_ppo.pt",
    map_location: str = "cpu",
) -> Tuple[CandidatePolicy, ...]:
    """The six trade-off candidates: `ppo_waste_main_{12,15,18,20}` and `ppo_waste_side_{15,20}`."""
    pilot = RLPilot.load(pilot_checkpoint, map_location=map_location)
    return tuple(
        CandidatePolicy(f"ppo_waste_main_{round(p * 100)}", PpoWaste(pilot, p, "main"))
        for p in MAIN_PROBABILITIES
    ) + tuple(
        CandidatePolicy(f"ppo_waste_side_{round(p * 100)}", PpoWaste(pilot, p, "side"))
        for p in SIDE_PROBABILITIES
    )

"""Pin SubRep's selection score to the reference specification's own trace.

Reference specification: SubRep-Minecraft-AIRIS_v2
https://drive.google.com/file/d/1Rpvusi_nIEIheElX7kUKQY88Dw0fazgS/view

The specification's option table gives delta_r and delta_n for each option,
and its execution trace gives the score B = delta_r + w . delta_n under the
weight vectors in force at each step. Recomputing those scores with SubRep's
own `score_skill_entry` must reproduce them exactly -- and only does so when
delta_r is the separate task reward, not sum(delta_n).

These tests touch no environment. They pin the scoring formula permanently,
whatever happens to the stub's calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from library.skill_selector import score_skill_entry
from utils.mdn_contracts import CandidateSkillRecord

# Specified weight vectors, objective order
# [Safety, Reputation, DeadlineSlack, InventoryValue, Sustainability,
#  Infrastructure].
W0 = np.array([0.35, 0.15, 0.20, 0.20, 0.05, 0.05])  # dusk, patrol risk
W1 = np.array([0.38, 0.17, 0.18, 0.17, 0.05, 0.05])  # patrol appears
W3 = np.array([0.25, 0.25, 0.20, 0.20, 0.05, 0.05])  # villagers, trading

# Specified options: (delta_r, delta_n).
IRON_GOLEM_SPAWN = (0.0, (0.40, 0.05, -0.08, 0.0, -0.03, 0.02))        # O5
SWING_GATE_BARRICADE = (0.0, (0.18, -0.01, -0.05, -0.01, 0.0, 0.12))   # O9
DISCOUNT_CHAIN = (0.01, (0.02, 0.20, -0.04, 0.35, 0.0, -0.02))         # O10


def _record(skill_id: str, option, *, bugged: bool = False) -> CandidateSkillRecord:
    delta_r, delta_n = option
    if bugged:
        delta_r = sum(delta_n)  # the old delta_r == sum(delta_n)
    return CandidateSkillRecord(
        skill_id=skill_id,
        delta_r=delta_r,
        delta_n=delta_n,
        is_certified=True,
        gate_type="PDS",
    )


@pytest.mark.parametrize(
    "skill_id, option, weights, specified",
    [
        ("IronGolemSpawn", IRON_GOLEM_SPAWN, W0, 0.1310),
        ("SwingGateBarricade", SWING_GATE_BARRICADE, W1, 0.0620),
        ("DiscountChain", DISCOUNT_CHAIN, W3, 0.126),
    ],
)
def test_specified_scores_reproduce_exactly(skill_id, option, weights, specified):
    score = score_skill_entry(_record(skill_id, option), weights)
    assert score == pytest.approx(specified, abs=1e-9)


def test_specified_first_choice_is_defence_not_trade():
    """At dusk (w0) the specification chooses IronGolemSpawn over trading."""
    golem = score_skill_entry(_record("IronGolemSpawn", IRON_GOLEM_SPAWN), W0)
    trade = score_skill_entry(_record("DiscountChain", DISCOUNT_CHAIN), W0)
    assert golem > trade


def test_summing_the_objectives_flips_that_choice():
    """Documents what bug 1 did: with delta_r = sum(delta_n), the agent trades
    at dusk with a patrol incoming instead of defending."""
    golem = score_skill_entry(
        _record("IronGolemSpawn", IRON_GOLEM_SPAWN, bugged=True), W0
    )
    trade = score_skill_entry(
        _record("DiscountChain", DISCOUNT_CHAIN, bugged=True), W0
    )
    assert golem == pytest.approx(0.491, abs=1e-9)
    assert trade == pytest.approx(0.608, abs=1e-9)
    assert trade > golem

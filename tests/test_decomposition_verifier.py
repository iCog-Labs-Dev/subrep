"""Tests for plan certification and ranking (decomposition/verifier.py)."""

import json

import numpy as np
import pytest
import torch

from certification.certificate_schema import Certificate
from decomposition.effects import (
    BASELINE_ID,
    DEFAULT_TABLE_PATH,
    ENVIRONMENT,
    GAMMA,
    GROUP_FACTS,
    SCHEMA_VERSION,
    EffectTable,
    plan_key,
)
from decomposition.planner import CandidatePlan, propose
from decomposition.recipe_book import LUNAR_LANDER_BOOK, load_recipe_book
from decomposition.verifier import CERTIFICATE_VERSION, MdnRegion, Priorities, best_plan, verify
from env.lunar_facts import FACT_NAMES
from env.lunar_subskills import SUBSKILLS

FACTS = {"upright": True, "over_pad": True, "near_ground": False}
GROUP = "upright=1,over_pad=1,near_ground=0"
TIMESTAMP = "2026-10-17T00:00:00+00:00"
EVEN = Priorities(weights=(0.5, 0.5))


def candidate(option, skills):
    return CandidatePlan("land_on_pad", option, (("land", option),), tuple(skills), {"landed": True})


def gains(delta_r, delta_n):
    def stats(lower):
        return {"mean": lower + 1.0, "std": 1.0, "lower": lower}

    return {"delta_r": stats(delta_r), "delta_safety": stats(delta_n[0]), "delta_fuel": stats(delta_n[1])}


def table_of(plans, min_samples=30):
    """A table measuring each (skills, Δr, Δn) from GROUP; the given gains are the lower bounds."""
    assert GROUP_FACTS == tuple(FACTS)
    entries = [
        {
            "name": plan_key(skills),
            "group": GROUP,
            "samples": 100,
            "completion_rate": 0.98,
            "mean_frames": 250.4,
            "skills": list(skills),
            "interaction": gains(0.5, (0.5, 0.0)),
            **gains(delta_r, delta_n),
        }
        for skills, delta_r, delta_n in plans
    ]
    return EffectTable(
        {
            "schema_version": SCHEMA_VERSION,
            "environment": ENVIRONMENT,
            "baseline_id": BASELINE_ID,
            "gamma": GAMMA,
            "confidence": 0.95,
            "bootstrap_resamples": 10,
            "min_samples": min_samples,
            "group_facts": list(GROUP_FACTS),
            "seeds": {"start": 7, "stop": 107},
            "steps": [],
            "plans": entries,
        }
    )


def region(support_values, alpha=(1.0, 1.0)):
    return MdnRegion(
        context=(0.1, 0.2),
        alpha=alpha,
        support_directions=((1.0, 0.0), (0.0, 1.0)),
        support_values=support_values,
    )


def verify_one(delta_r, delta_n, priorities=EVEN):
    table = table_of([(("a", "b"), delta_r, delta_n)])
    [verdict] = verify([candidate("plan", ["a", "b"])], FACTS, table, priorities, TIMESTAMP)
    return verdict


# Gates


def test_cds_admits_a_plan_that_gains_under_every_weighting():
    verdict = verify_one(10.0, (5.0, -3.0))
    assert verdict.admitted and verdict.reason == "admitted by the CDS gate"
    assert verdict.cds_margin == pytest.approx(7.0)
    certificate = verdict.entry.certificate
    assert certificate.skill_id == "plan:land_on_pad/plan"
    assert certificate.gate_type == "CDS" and certificate.epsilon == 0.0
    assert certificate.delta_r == 10.0 and certificate.delta_n == (5.0, -3.0)
    assert certificate.admission_margin == pytest.approx(7.0)
    assert (certificate.baseline_id, certificate.environment, certificate.gamma) == (BASELINE_ID, ENVIRONMENT, GAMMA)
    assert (certificate.seed, certificate.episode_length, certificate.timestamp) == (7, 250, TIMESTAMP)
    assert certificate.version == CERTIFICATE_VERSION
    assert certificate.weight_region_type == "FULL_SIMPLEX"
    assert verdict.score == pytest.approx(10.0 + 0.5 * 5.0 - 0.5 * 3.0)


def test_pds_admits_a_trade_off_within_its_budget():
    verdict = verify_one(1.0, (5.0, -3.0), Priorities(weights=(0.5, 0.5), pds_epsilon=2.5))
    assert verdict.reason == "admitted by the PDS gate"
    assert verdict.entry.certificate.epsilon == 2.5
    assert verdict.entry.certificate.admission_margin == pytest.approx(0.5)


def test_a_trade_off_beyond_the_budget_is_rejected_with_its_margins():
    verdict = verify_one(1.0, (5.0, -3.0), Priorities(weights=(0.5, 0.5), pds_epsilon=1.0))
    assert not verdict.admitted
    assert verdict.reason == "rejected by the CDS and PDS gates"
    assert verdict.cds_margin == pytest.approx(-2.0)
    assert verdict.pds_margin == pytest.approx(-1.0)
    assert verdict.score is None and verdict.estimate is not None


def test_the_gates_use_the_lower_bounds_not_the_means():
    # The means are one higher than the lower bounds: CDS would pass on the means.
    assert not verify_one(-0.5, (0.0, 0.0)).admitted


def test_plans_without_enough_measurements_are_not_certified():
    table = table_of([(("a", "b"), 10.0, (5.0, 5.0))], min_samples=1000)
    verdicts = verify([candidate("thin", ["a", "b"]), candidate("unknown", ["c"])], FACTS, table, EVEN, TIMESTAMP)
    assert [v.reason for v in verdicts] == ["not measured from a state like this one"] * 2
    assert all(v.estimate is None and not v.admitted for v in verdicts)


def test_an_mdn_region_admits_trade_offs_it_rules_out():
    assert not verify_one(0.0, (30.0, -20.0)).admitted
    safety_first = region(support_values=(1.0, 0.3))
    verdict = verify_one(0.0, (30.0, -20.0), Priorities(weights=(0.7, 0.3), region=safety_first))
    assert verdict.reason == "admitted by the CDS gate"
    assert verdict.cds_margin == pytest.approx(0.7 * 30.0 - 0.3 * 20.0)
    certificate = verdict.entry.certificate
    assert certificate.weight_region_type == "MDN_WX"
    assert certificate.wx_support_values == (1.0, 0.3)
    assert certificate.mdn_alpha == (1.0, 1.0)
    assert verdict.entry.weight_region_type == "MDN_WX"


@pytest.mark.parametrize("alpha, admitted", [((9.0, 1.0), True), ((1.0, 9.0), False)])
def test_the_cvar_check_follows_the_mdn_distribution(alpha, admitted):
    priorities = Priorities(weights=(0.7, 0.3), cvar_tail_level=0.1, region=region((1.0, 0.3), alpha))
    verdict = verify_one(0.0, (30.0, -20.0), priorities)
    assert verdict.cds_margin > 0.0
    assert verdict.admitted is admitted
    assert (verdict.cvar >= 0.0) is admitted
    if not admitted:
        assert verdict.reason == "rejected by the CVaR check"


def test_the_cvar_check_is_reproducible_and_leaves_torch_alone():
    priorities = Priorities(weights=(0.7, 0.3), cvar_tail_level=0.1, region=region((1.0, 0.3), (2.0, 3.0)))
    state = torch.get_rng_state()
    first = verify_one(0.0, (30.0, -20.0), priorities).cvar
    assert torch.equal(torch.get_rng_state(), state)
    assert verify_one(0.0, (30.0, -20.0), priorities).cvar == first


# Ranking


def test_admitted_plans_are_ranked_by_score_then_steps_then_name():
    table = table_of(
        [
            (("best",), 20.0, (1.0, 1.0)),
            (("s1", "s2"), 10.0, (1.0, 1.0)),
            (("s1", "s2", "s3"), 10.0, (1.0, 1.0)),
            (("s4", "s5"), 10.0, (1.0, 1.0)),
            (("bad",), -50.0, (1.0, 1.0)),
        ]
    )
    candidates = [
        candidate("rejected", ["bad"]),
        candidate("beta", ["s1", "s2"]),
        candidate("longer", ["s1", "s2", "s3"]),
        candidate("alpha", ["s4", "s5"]),
        candidate("top", ["best"]),
    ]
    verdicts = verify(candidates, FACTS, table, EVEN, TIMESTAMP)
    assert [v.candidate.option for v in verdicts] == ["top", "alpha", "beta", "longer", "rejected"]
    assert best_plan(verdicts).candidate.option == "top"


@pytest.mark.parametrize("weights, best", [((0.9, 0.1), "safe"), ((0.1, 0.9), "cheap")])
def test_priorities_choose_between_trade_offs(weights, best):
    table = table_of([(("safe",), 20.0, (30.0, -10.0)), (("cheap",), 20.0, (10.0, 10.0))])
    candidates = [candidate("safe", ["safe"]), candidate("cheap", ["cheap"])]
    verdicts = verify(candidates, FACTS, table, Priorities(weights=weights), TIMESTAMP)
    assert all(v.admitted for v in verdicts)
    assert best_plan(verdicts).candidate.option == best


def test_no_admitted_plan_means_no_best_plan():
    assert best_plan([verify_one(-50.0, (0.0, 0.0))]) is None


# Records and priorities


def test_records_are_json_ready_and_certificates_round_trip():
    admitted, rejected = verify_one(10.0, (5.0, -3.0)), verify_one(-50.0, (0.0, 0.0))
    record = json.loads(json.dumps(admitted.record()))
    assert record["skills"] == ["a", "b"] and record["admitted"] is True
    assert record["estimate"]["lower"] == {"delta_r": 10.0, "delta_n": [5.0, -3.0]}
    assert record["estimate"]["interaction"] == {"delta_r": 1.5, "delta_n": [1.5, 1.0]}
    assert Certificate.from_dict(record["certificate"]) == admitted.entry.certificate
    assert json.loads(json.dumps(rejected.record()))["certificate"] is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"weights": (0.6, 0.6)}, "simplex"),
        ({"weights": (-0.1, 1.1)}, "simplex"),
        ({"weights": (0.5, 0.5), "pds_epsilon": -1.0}, "non-negative"),
        ({"weights": (0.5, 0.5), "cvar_tail_level": 0.0, "region": region((1.0, 1.0))}, r"\(0, 1\]"),
        ({"weights": (0.5, 0.5), "cvar_tail_level": 0.1}, "needs an MDN region"),
        ({"weights": (0.5, 0.5), "region": MdnRegion((0.0,), (1.0,), ((1.0,),), (1.0,))}, "same objectives"),
        ({"weights": (0.5, 0.5), "region": region((1.0, 0.3))}, "outside the MDN region"),
    ],
)
def test_priorities_are_checked(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Priorities(**kwargs)


NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"weights": (NAN, NAN)}, "weights must contain only finite"),
        ({"weights": (INF, 0.0)}, "weights must contain only finite"),
        ({"weights": ()}, "non-empty"),
        ({"weights": (0.5, 0.5), "pds_epsilon": NAN}, "pds_epsilon must be a finite"),
        ({"weights": (0.5, 0.5), "pds_epsilon": INF}, "pds_epsilon must be a finite"),
        ({"weights": (0.5, 0.5), "cvar_tail_level": NAN, "region": region((1.0, 1.0))}, "finite"),
        ({"weights": (0.5, 0.5), "cvar_tail_level": INF, "region": region((1.0, 1.0))}, "finite"),
    ],
)
def test_priorities_reject_numbers_that_are_not_finite(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Priorities(**kwargs)


IDENTITY = ((1.0, 0.0), (0.0, 1.0))


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"context": (NAN,)}, "context must contain only finite"),
        ({"alpha": (1.0, INF)}, "alpha must contain only finite"),
        ({"support_values": (NAN, 1.0)}, "support_values must contain only finite"),
        ({"support_values": (1.0, INF)}, "support_values must contain only finite"),
        ({"support_directions": ((1.0, NAN), (0.0, 1.0))}, "support_directions must contain only finite"),
        ({"alpha": (0.0, 1.0)}, "alpha must be positive"),
        ({"support_values": (1.2, 0.3)}, r"in \[0, 1\]"),
        ({"support_values": (0.4, 0.3)}, "region is empty"),
        ({"alpha": (1.0, 1.0, 1.0)}, "same objectives"),
        ({"support_directions": ((1.0, 0.0),)}, "same objectives"),
    ],
)
def test_mdn_regions_reject_invalid_values(fields, message):
    values = {"context": (0.1,), "alpha": (1.0, 1.0), "support_directions": IDENTITY, "support_values": (1.0, 0.3)}
    with pytest.raises(ValueError, match=message):
        MdnRegion(**{**values, **fields})


# With the LunarLander book and the measured table


@pytest.fixture(scope="module")
def lunar_book():
    return load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, SUBSKILLS)


@pytest.fixture(scope="module")
def lunar_table():
    return EffectTable.load(DEFAULT_TABLE_PATH)


def test_a_safety_first_region_certifies_a_landing_plan_from_the_air(lunar_book, lunar_table):
    facts = dict(upright=False, over_pad=False, near_ground=False, slow=False, falling_fast=True, landed=False)
    priorities = Priorities(weights=(0.7, 0.3), region=region((1.0, 0.3)))
    verdicts = verify(propose(lunar_book, "land_on_pad", facts), facts, lunar_table, priorities, TIMESTAMP)
    best = best_plan(verdicts)
    assert best is not None
    certificate = best.entry.certificate
    assert certificate.skill_id in {"plan:land_on_pad/careful_landing", "plan:land_on_pad/fuel_saving_landing"}
    assert best.estimate.samples >= lunar_table.min_samples
    assert np.isclose(best.score, certificate.delta_r + np.dot((0.7, 0.3), certificate.delta_n))


def test_plans_from_unmeasured_states_are_not_certified(lunar_book, lunar_table):
    facts = dict(upright=True, over_pad=True, near_ground=True, slow=True, falling_fast=False, landed=False)
    priorities = Priorities(weights=(0.7, 0.3), pds_epsilon=1000.0)
    verdicts = verify(propose(lunar_book, "land_on_pad", facts), facts, lunar_table, priorities, TIMESTAMP)
    assert verdicts
    assert best_plan(verdicts) is None

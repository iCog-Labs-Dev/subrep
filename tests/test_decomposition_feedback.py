"""Tests for checking certificates against realised gains (decomposition/feedback.py)."""

import json

import numpy as np
import pytest

from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable, Gains, fly_plan, group_key
from decomposition.feedback import REVOKED_REASON, CertificateStatus, FeedbackLog, realised_gains
from decomposition.planner import propose
from decomposition.recipe_book import LUNAR_LANDER_BOOK, load_recipe_book
from decomposition.runtime import GoalStatus, pursue
from decomposition.verifier import MdnRegion, Priorities, best_plan, plan_skill_id, verify
from env.lunar_facts import FACT_NAMES, lander_facts
from env.lunar_lander_wrapper import SubRepEnv
from env.lunar_subskills import SUBSKILLS

SEED = 0
SAFETY_FIRST = Priorities(
    weights=(0.7, 0.3),
    region=MdnRegion(
        context=(0.0,),
        alpha=(4.0, 1.0),
        support_directions=((1.0, 0.0), (0.0, 1.0)),
        support_values=(1.0, 0.3),
    ),
)


@pytest.fixture
def env():
    environment = SubRepEnv(seed=0)
    yield environment
    environment.close()


@pytest.fixture(scope="module")
def book():
    return load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, SUBSKILLS)


@pytest.fixture(scope="module")
def table():
    return EffectTable.load(DEFAULT_TABLE_PATH)


@pytest.fixture
def start(env):
    observation, _ = env.reset(seed=SEED)
    return observation, lander_facts(observation)


@pytest.fixture
def verdicts(book, table, start):
    _, facts = start
    return verify(propose(book, "land_on_pad", facts), facts, table, SAFETY_FIRST)


def shifted(gains, by, count, spread=1.0):
    """``count`` realised gains around ``gains`` moved by ``by``, with a little reproducible noise."""
    noise = np.random.default_rng(0).normal(0.0, spread, size=(count, 3))
    return [
        Gains(gains.delta_r + by + n[0], (gains.delta_n[0] + by + n[1], gains.delta_n[1] + by + n[2])) for n in noise
    ]


def certified_gains(verdict):
    certificate = verdict.entry.certificate
    return Gains(certificate.delta_r, certificate.delta_n)


def test_realised_gains_match_the_paired_measurement_of_the_same_flight(env, book, table, start):
    observation, _ = start
    run = pursue(env, observation, "land_on_pad", book, table, SAFETY_FIRST)
    realised = realised_gains(env, SEED, run.rewards)
    measured = fly_plan(env, SEED, run.attempts[0].chosen.candidate.skills).plan
    assert realised.delta_r == pytest.approx(measured.delta_r)
    assert realised.delta_n == pytest.approx(measured.delta_n)


def test_a_certificate_holds_when_realised_gains_meet_it(verdicts, start):
    _, facts = start
    best, log = best_plan(verdicts), FeedbackLog()
    for realised in shifted(certified_gains(best), by=5.0, count=40):
        log.record(best, facts, realised)
    [entry] = log.report()
    assert entry["status"] == CertificateStatus.HELD.value
    assert entry["plan"] == plan_skill_id(best.candidate) and entry["group"] == group_key(facts)
    assert entry["samples"] == 40
    assert entry["delta_r"]["realised_mean"] > entry["delta_r"]["certified"]
    assert log.revoked() == frozenset()


def test_a_certificate_is_revoked_when_realised_gains_fall_short(verdicts, start):
    _, facts = start
    best, log = best_plan(verdicts), FeedbackLog()
    for realised in shifted(certified_gains(best), by=-10.0, count=40):
        log.record(best, facts, realised)
    [entry] = log.report()
    assert entry["status"] == CertificateStatus.REVOKED.value
    assert entry["delta_r"]["realised_upper"] < entry["delta_r"]["certified"]
    assert log.revoked() == {(plan_skill_id(best.candidate), group_key(facts))}


def test_a_few_bad_outcomes_do_not_revoke_a_certificate(verdicts, start):
    _, facts = start
    best, log = best_plan(verdicts), FeedbackLog()
    for realised in shifted(certified_gains(best), by=-100.0, count=5):
        log.record(best, facts, realised)
    [entry] = log.report()
    assert entry["status"] == CertificateStatus.TOO_FEW_SAMPLES.value
    assert entry["delta_r"]["realised_upper"] is None
    assert log.revoked() == frozenset()


def test_only_admitted_plans_can_be_recorded(book, table, start):
    _, facts = start
    rejected = verify(propose(book, "land_on_pad", facts), facts, table, Priorities(weights=(0.5, 0.5)))[0]
    with pytest.raises(ValueError, match="admitted"):
        FeedbackLog().record(rejected, facts, Gains(0.0, (0.0, 0.0)))


def test_the_runtime_skips_a_revoked_plan(env, book, table, verdicts, start):
    observation, facts = start
    best, log = best_plan(verdicts), FeedbackLog()
    for realised in shifted(certified_gains(best), by=-10.0, count=40):
        log.record(best, facts, realised)

    screened = log.screen(verdicts, facts)
    assert [v.candidate.option for v in screened] == [v.candidate.option for v in verdicts]
    assert screened[0].reason == REVOKED_REASON and not screened[0].admitted

    run = pursue(env, observation, "land_on_pad", book, table, SAFETY_FIRST, feedback=log)
    chosen = run.attempts[0].chosen
    assert chosen.candidate.option != best.candidate.option
    assert run.status is GoalStatus.ACHIEVED


def test_the_report_is_json_ready(verdicts, start):
    _, facts = start
    log = FeedbackLog(min_samples=2)
    for realised in shifted(certified_gains(best_plan(verdicts)), by=0.0, count=3):
        log.record(best_plan(verdicts), facts, realised)
    assert json.loads(json.dumps(log.report()))[0]["samples"] == 3


def test_min_samples_is_checked():
    with pytest.raises(ValueError, match="min_samples"):
        FeedbackLog(min_samples=1)

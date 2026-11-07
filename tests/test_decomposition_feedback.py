"""Tests for checking certificates against realised gains (decomposition/feedback.py)."""

import dataclasses
import json

import numpy as np
import pytest

from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable, Gains, fly_plan, group_key, plan_key, replay
from decomposition.executor import execute
from decomposition.feedback import REVOKED_REASON, CertificateStatus, FeedbackLog, realised_gains
from decomposition.planner import propose
from decomposition.recipe_book import LUNAR_LANDER_BOOK, load_recipe_book
from decomposition.runtime import DEFAULT_MIN_GOAL_RATE, pursue
from decomposition.verifier import MdnRegion, Priorities, best_plan, plan_skill_id, verify
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import FACT_NAMES, lander_facts
from env.lunar_lander_wrapper import SubRepEnv

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
    return load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, LUNAR_LANDER.skills)


@pytest.fixture(scope="module")
def table():
    return EffectTable.load(DEFAULT_TABLE_PATH, LUNAR_LANDER)


@pytest.fixture
def start(env):
    observation, _ = env.reset(seed=SEED)
    return observation, lander_facts(observation)


@pytest.fixture
def verdicts(book, table, start):
    observation, facts = start
    return verify(propose(book, "land_on_pad", facts), observation, table, SAFETY_FIRST, min_goal_rate=DEFAULT_MIN_GOAL_RATE)


def shifted(gains, by, count, spread=1.0):
    """``count`` realised gains around ``gains`` moved by ``by``, with a little reproducible noise."""
    noise = np.random.default_rng(0).normal(0.0, spread, size=(count, 3))
    return [
        Gains(gains.delta_r + by + n[0], (gains.delta_n[0] + by + n[1], gains.delta_n[1] + by + n[2])) for n in noise
    ]


def certified_gains(verdict):
    certificate = verdict.entry.certificate
    return Gains(certificate.delta_r, certificate.delta_n)


def test_realised_gains_match_the_paired_measurement_of_the_same_flight(env):
    skills = ["stabilise", "move_over_pad", "slow_descent", "touch_down"]
    observation, _ = env.reset(seed=SEED)
    execution = execute(LUNAR_LANDER, env, load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, LUNAR_LANDER.skills), skills, observation)
    realised = realised_gains(LUNAR_LANDER, env, SEED, execution.rewards)
    measured = fly_plan(LUNAR_LANDER, env, SEED, skills).plan
    assert realised.delta_r == pytest.approx(measured.delta_r)
    assert realised.delta_n == pytest.approx(measured.delta_n)


def test_realised_gains_from_mid_flight_start_at_that_state(env):
    skills = ["stabilise", "move_over_pad", "slow_descent", "touch_down"]
    flight = fly_plan(LUNAR_LANDER, env, SEED, skills)
    prefix = flight.next_states[0]
    rest = fly_plan(LUNAR_LANDER, env, SEED, skills[1:], prefix)
    start = replay(env, SEED, prefix)
    execution = execute(LUNAR_LANDER, env, load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, LUNAR_LANDER.skills), skills[1:], start)
    realised = realised_gains(LUNAR_LANDER, env, SEED, execution.rewards, prefix)
    assert realised.delta_r == pytest.approx(rest.plan.delta_r)


def test_a_certificate_holds_when_realised_gains_meet_it(verdicts, start):
    _, facts = start
    best, log = best_plan(verdicts), FeedbackLog(LUNAR_LANDER)
    for realised in shifted(certified_gains(best), by=5.0, count=40):
        log.record(best, facts, realised)
    [entry] = log.report()
    assert entry["status"] == CertificateStatus.HELD.value
    assert entry["plan"] == plan_skill_id(best.candidate) and entry["group"] == group_key(LUNAR_LANDER, facts)
    assert entry["samples"] == 40
    assert entry["delta_r"]["realised_mean"] > entry["delta_r"]["certified"]
    assert log.revoked() == frozenset()


def test_a_certificate_is_revoked_when_realised_gains_fall_short(verdicts, start):
    _, facts = start
    best, log = best_plan(verdicts), FeedbackLog(LUNAR_LANDER)
    for realised in shifted(certified_gains(best), by=-10.0, count=40):
        log.record(best, facts, realised)
    [entry] = log.report()
    assert entry["status"] == CertificateStatus.REVOKED.value
    assert entry["delta_r"]["realised_upper"] < entry["delta_r"]["certified"]
    assert log.revoked() == {(plan_skill_id(best.candidate), group_key(LUNAR_LANDER, facts))}


def test_a_few_bad_outcomes_do_not_revoke_a_certificate(verdicts, start):
    _, facts = start
    best, log = best_plan(verdicts), FeedbackLog(LUNAR_LANDER)
    for realised in shifted(certified_gains(best), by=-100.0, count=5):
        log.record(best, facts, realised)
    [entry] = log.report()
    assert entry["status"] == CertificateStatus.TOO_FEW_SAMPLES.value
    assert entry["delta_r"]["realised_upper"] is None
    assert log.revoked() == frozenset()


def test_only_admitted_plans_can_be_recorded(book, table, start):
    observation, facts = start
    rejected = verify(propose(book, "land_on_pad", facts), observation, table, SAFETY_FIRST, min_goal_rate=1.0)[0]
    assert not rejected.admitted
    with pytest.raises(ValueError, match="admitted"):
        FeedbackLog(LUNAR_LANDER).record(rejected, facts, Gains(0.0, (0.0, 0.0)))


def test_the_runtime_skips_a_revoked_plan(env, book, table, verdicts, start):
    observation, facts = start
    best, log = best_plan(verdicts), FeedbackLog(LUNAR_LANDER)
    for realised in shifted(certified_gains(best), by=-10.0, count=40):
        log.record(best, facts, realised)

    screened = log.screen(verdicts, facts)
    assert [v.candidate.option for v in screened] == [v.candidate.option for v in verdicts]
    assert screened[0].reason == REVOKED_REASON and not screened[0].admitted

    run = pursue(LUNAR_LANDER, env, observation, "land_on_pad", book, table, SAFETY_FIRST, feedback=log)
    chosen = run.decisions[0].chosen
    assert chosen is None or chosen.candidate.skills != best.candidate.skills
    assert REVOKED_REASON in {verdict.reason for verdict in run.decisions[0].verdicts}


def recertified(verdict, by):
    """``verdict`` with a certificate claiming ``by`` more on every gain, as a rebuilt table might."""
    certificate = verdict.entry.certificate
    claim = dataclasses.replace(
        certificate,
        delta_r=certificate.delta_r + by,
        delta_n=tuple(value + by for value in certificate.delta_n),
    )
    return dataclasses.replace(verdict, entry=dataclasses.replace(verdict.entry, certificate=claim))


def test_a_revocation_stands_until_a_certificate_claims_no_more_than_the_flights_support(verdicts, start):
    _, facts = start
    best, log = best_plan(verdicts), FeedbackLog(LUNAR_LANDER)
    for realised in shifted(certified_gains(best), by=-10.0, count=40):
        log.record(best, facts, realised)
    [same, overclaiming, supported] = log.screen([best, recertified(best, 5.0), recertified(best, -20.0)], facts)
    assert same.reason == REVOKED_REASON and overclaiming.reason == REVOKED_REASON
    assert supported.admitted and supported.entry.certificate.delta_r == best.entry.certificate.delta_r - 20.0


def test_the_report_keeps_every_certificate_it_judged(verdicts, start):
    _, facts = start
    best, log = best_plan(verdicts), FeedbackLog(LUNAR_LANDER)
    rebuilt = recertified(best, -20.0)
    for index, realised in enumerate(shifted(certified_gains(best), by=-10.0, count=40)):
        log.record(best if index < 20 else rebuilt, facts, realised)
    [entry] = log.report()
    assert entry["status"] == CertificateStatus.HELD.value  # the latest certificate holds
    assert entry["delta_r"]["certified"] == pytest.approx(certified_gains(rebuilt).delta_r)
    assert [c["status"] for c in entry["certificates"]] == [CertificateStatus.REVOKED.value, CertificateStatus.HELD.value]
    assert entry["certificates"][0]["certified"][0] == pytest.approx(certified_gains(best).delta_r)
    assert log.revoked() == frozenset()


def with_claim_lowered(table, skills, group, by):
    """``table`` as if rebuilt with the entry for ``skills`` from ``group`` claiming ``by`` less on every gain."""

    def lowered(entry):
        if (entry["name"], entry["group"]) != (plan_key(skills), group):
            return entry
        return {**entry, **{k: {**v, "lower": v["lower"] - by} for k, v in entry.items() if k.startswith("delta_")}}

    return EffectTable({**table.data, "plans": [lowered(entry) for entry in table.data["plans"]]}, LUNAR_LANDER)


def test_the_runtime_flies_a_revoked_plan_again_once_new_evidence_supports_it(env, book, table, verdicts, start):
    observation, facts = start
    best, log = best_plan(verdicts), FeedbackLog(LUNAR_LANDER)
    for realised in shifted(certified_gains(best), by=-10.0, count=40):
        log.record(best, facts, realised)
    rebuilt = with_claim_lowered(table, best.candidate.skills, group_key(LUNAR_LANDER, facts), by=20.0)

    def verdict_under(current):
        env.reset(seed=SEED)
        run = pursue(LUNAR_LANDER, env, observation, "land_on_pad", book, current, SAFETY_FIRST, feedback=log)
        [verdict] = [v for v in run.decisions[0].verdicts if v.candidate.skills == best.candidate.skills]
        return verdict

    assert verdict_under(table).reason == REVOKED_REASON
    assert verdict_under(rebuilt).admitted


def test_the_report_is_json_ready(verdicts, start):
    _, facts = start
    log = FeedbackLog(LUNAR_LANDER, min_samples=2)
    for realised in shifted(certified_gains(best_plan(verdicts)), by=0.0, count=3):
        log.record(best_plan(verdicts), facts, realised)
    assert json.loads(json.dumps(log.report()))[0]["samples"] == 3


def test_min_samples_is_checked():
    with pytest.raises(ValueError, match="min_samples"):
        FeedbackLog(LUNAR_LANDER, min_samples=1)


def test_the_report_follows_every_new_outcome(verdicts, start):
    _, facts = start
    best, log = best_plan(verdicts), FeedbackLog(LUNAR_LANDER)
    outcomes = shifted(certified_gains(best), by=-10.0, count=31)
    for realised in outcomes[:30]:
        log.record(best, facts, realised)
    assert log.report()[0]["samples"] == 30 and log.revoked()
    log.report()[0]["samples"] = 0  # a caller changing its copy does not change the log
    log.record(best, facts, outcomes[30])
    assert log.report()[0]["samples"] == 31

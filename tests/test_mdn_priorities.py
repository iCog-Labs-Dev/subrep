"""Tests for priorities from a trained MDN (decomposition/mdn_priorities.py, demo/train_lunar_mdn.py)."""

import numpy as np
import pytest
import torch

from decomposition.effects import DEFAULT_TABLE_PATH, EffectTable
from decomposition.mdn_priorities import (
    CONCENTRATION,
    MOTIVE_SHIFT,
    MdnPriorities,
    evaluate_mdn,
    priorities_from_weights,
    train_mdn,
)
from decomposition.recipe_book import LUNAR_LANDER_BOOK, load_recipe_book
from decomposition.runtime import GoalStatus, pursue
from demo.train_lunar_mdn import (
    LANDING_PLAN,
    MISSION_PLAN,
    collect_observations,
    landing_schedule,
    load_landing_mdn,
    load_mission_mdn,
    mission_schedule,
)
from env.lunar_beacon import BEACON_MISSION, beacon_facts
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import FACT_NAMES
from generator.mdn import MotiveDecompositionNetwork

OBJECTIVES = ("safety", "fuel")


def constant_schedule(observation):
    return (0.7, 0.3)


@pytest.fixture(scope="module")
def observations():
    return collect_observations(LUNAR_LANDER, LANDING_PLAN, range(4))


@pytest.fixture(scope="module")
def trained(observations):
    return train_mdn(observations, constant_schedule, OBJECTIVES, epochs=300)


class FixedModel(MotiveDecompositionNetwork):
    """An MDN that gives fixed answers, to test what the source does with them."""

    def __init__(self, alpha, support):
        super().__init__(input_dim=8, num_objectives=len(alpha), num_skills=1)
        self.answers = (torch.tensor(alpha, dtype=torch.float32), torch.tensor(support, dtype=torch.float32))

    def forward_inference(self, context):
        return self.answers


OBSERVATION = np.zeros(8, dtype=np.float32)


# Fixed priorities


def test_fixed_priorities_have_the_region_an_mdn_would_give():
    priorities = priorities_from_weights((0.7, 0.3), cvar_tail_level=0.1)
    assert priorities.weights == (0.7, 0.3)
    assert priorities.region.alpha == pytest.approx((CONCENTRATION * 0.7, CONCENTRATION * 0.3))
    assert priorities.region.support_values == pytest.approx((0.7 + MOTIVE_SHIFT, 0.3 + MOTIVE_SHIFT))
    assert priorities.cvar_tail_level == 0.1


def test_fixed_priorities_cap_the_region_at_one():
    assert priorities_from_weights((0.98, 0.02)).region.support_values == pytest.approx((1.0, 0.02 + MOTIVE_SHIFT))


# A source of priorities


def test_weights_are_the_mean_of_the_dirichlet_and_the_region_comes_with_them():
    source = MdnPriorities(FixedModel([35.0, 15.0], [0.75, 0.35]), OBJECTIVES, cvar_tail_level=0.1)
    priorities = source(OBSERVATION)
    assert priorities.weights == pytest.approx((0.7, 0.3))
    assert priorities.region.support_values == pytest.approx((0.75, 0.35))
    assert priorities.region.alpha == pytest.approx((35.0, 15.0))
    assert priorities.cvar_tail_level == 0.1


def test_weights_outside_the_mdns_own_region_are_refused_not_raised():
    source = MdnPriorities(FixedModel([35.0, 15.0], [0.6, 0.45]), OBJECTIVES)
    assert source(OBSERVATION) is None


def test_the_source_checks_its_objectives():
    with pytest.raises(ValueError, match="objectives"):
        MdnPriorities(FixedModel([1.0, 1.0], [1.0, 1.0]), ("a", "b", "c"))


# Training


def test_training_fits_the_schedule(trained, observations):
    source, metrics = trained
    assert metrics["alpha_log_mse"] < 0.05
    report = evaluate_mdn(source, observations, constant_schedule)
    assert report["contexts"] == len(observations)
    assert report["refused_rate"] == 0.0
    assert report["max_weight_error"] < MOTIVE_SHIFT


def test_a_checkpoint_round_trips(trained, observations, tmp_path):
    source, _ = trained
    path = source.save(tmp_path / "mdn.pth", {"note": 1.0})
    loaded = MdnPriorities.load(path, OBJECTIVES)
    assert loaded(observations[0]).weights == pytest.approx(source(observations[0]).weights)


def test_a_checkpoint_for_other_objectives_is_refused(trained, tmp_path):
    source, _ = trained
    path = source.save(tmp_path / "mdn.pth")
    with pytest.raises(ValueError, match="objectives"):
        MdnPriorities.load(path, ("safety", "time"))


def test_a_checkpoint_of_another_version_is_refused(trained, tmp_path):
    source, _ = trained
    path = source.save(tmp_path / "mdn.pth")
    checkpoint = torch.load(path, weights_only=False)
    torch.save({**checkpoint, "checkpoint_version": "0"}, path)
    with pytest.raises(ValueError, match="version"):
        MdnPriorities.load(path, OBJECTIVES)


# The schedules and the committed networks


def lander(y=1.0, angle=0.0, progress=None):
    numbers = [0.0, y, 0.0, 0.0, angle, 0.0, 0.0, 0.0]
    return np.array(numbers if progress is None else numbers + [0.0, 0.0, progress], dtype=np.float32)


def test_the_landing_schedule_favours_safety_near_the_ground_and_when_tilted():
    assert landing_schedule(lander()) == (0.7, 0.3)
    assert landing_schedule(lander(y=0.1)) == (0.8, 0.2)
    assert landing_schedule(lander(angle=0.5)) == (0.8, 0.2)


def test_the_mission_schedule_moves_from_the_mission_to_getting_down():
    assert mission_schedule(lander(progress=0.0))[2] == max(mission_schedule(lander(progress=0.0)))
    assert mission_schedule(lander(progress=1.0))[0] == max(mission_schedule(lander(progress=1.0)))
    assert mission_schedule(lander(y=0.1, progress=1.0))[0] > mission_schedule(lander(progress=1.0))[0]


@pytest.mark.parametrize("schedule", [landing_schedule, mission_schedule])
def test_every_schedule_is_on_the_simplex(schedule):
    for observation in (lander(progress=0.0), lander(progress=1.0), lander(y=0.1, progress=1.0)):
        inputs = observation if schedule is mission_schedule else observation[:8]
        assert sum(schedule(inputs)) == pytest.approx(1.0)


def test_the_committed_landing_mdn_follows_its_schedule():
    source = load_landing_mdn()
    observations = collect_observations(LUNAR_LANDER, LANDING_PLAN, range(2000, 2005))
    report = evaluate_mdn(source, observations, landing_schedule)
    assert report["refused_rate"] < 0.2
    assert report["mean_weight_error"] < MOTIVE_SHIFT


def test_the_committed_mission_mdn_follows_its_schedule():
    source = load_mission_mdn()
    observations = collect_observations(BEACON_MISSION, MISSION_PLAN, range(2000, 2005))
    report = evaluate_mdn(source, observations, mission_schedule)
    assert report["refused_rate"] < 0.2
    assert report["mean_weight_error"] < MOTIVE_SHIFT
    start = source(observations[0])
    assert start.weights[2] > start.weights[1]
    assert beacon_facts(observations[0])["beacon_scanned"] is False


def test_the_runtime_flies_under_the_committed_mdn():
    book = load_recipe_book(LUNAR_LANDER_BOOK, FACT_NAMES, LUNAR_LANDER.skills)
    table = EffectTable.load(DEFAULT_TABLE_PATH, LUNAR_LANDER)
    env = LUNAR_LANDER.make_env(0)
    try:
        observation, _ = env.reset(seed=8000)
        run = pursue(LUNAR_LANDER, env, observation, "land_on_pad", book, table, load_landing_mdn())
    finally:
        env.close()
    assert run.status in (GoalStatus.ACHIEVED, GoalStatus.FAILED)
    weights = [decision.weights for decision in run.flown if decision.weights is not None]
    assert weights and all(sum(w) == pytest.approx(1.0) for w in weights)

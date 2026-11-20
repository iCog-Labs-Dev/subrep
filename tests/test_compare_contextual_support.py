import numpy as np
import pytest
from utils.compare_contextual_support import coverage, evaluate_context


def test_volume():
    assert coverage([1,1,1]) == pytest.approx(1)
    assert coverage([1/3]*3) == pytest.approx(0)
    assert coverage([.8]*3) == pytest.approx(.88)
    rng = np.random.default_rng(42)
    points = rng.dirichlet(np.ones(3), size=100000)
    caps = np.array([.2,.8,.8])
    assert coverage(caps) == pytest.approx(np.all(points <= caps, axis=1).mean(), abs=.005)
    with pytest.raises(ValueError):
        coverage([.1]*3)


def test_admission_coverage_and_execution_are_separate():
    idle = dict(motives=[0,0,0], undiscounted_motives=[0,0,0], goals=0)
    record = dict(outcomes={'zero_action': idle, 'ppo': dict(motives=[-1,.1,.1], goals=1)})
    execution = dict(outcomes={'zero_action': idle, 'ppo': dict(motives=[-2,.2,.1], undiscounted_motives=[-3,.3,.2], goals=1)})
    full = evaluate_context(record, execution, np.ones(3), np.ones(3), .2)
    learned = evaluate_context(record, execution, np.ones(3), np.array([.2,.8,.8]), .2)
    assert full['passed'] == 0
    assert learned['passed'] == learned['pds_only'] == 1
    assert [s['fallback'] for s in learned['selections']] == [True,True,False,False]
    assert learned['selections'][2]['raw_cost'] == 3
    assert learned['selections'][2]['discounted_cost'] == 2

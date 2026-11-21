import numpy as np
import pytest
from demo.evaluate_safety_readiness import summarize


def test_admission_denominator_and_goal_budget_are_distinct():
    # Equal motives pass CDS even though no goal was reached.
    idle = dict(motives=[0,0,0,0], undiscounted_motives=[0,0,0,0], goals=0)
    bad = dict(motives=[-10,-10,-10,-10], undiscounted_motives=[-10,-10,-10,-10], goals=1)
    records = [dict(context_seed=1, outcomes={'zero_action':idle,'a':idle,'b':bad})]
    r = summarize(records, ['a','b'], np.ones(4), .2, 1.)
    assert r['policy_context_admission_rate'] == .5
    assert r['context_coverage'] == 1.
    assert r['policies'][0]['admitted_goal_and_episode_budget_rate'] == 0.
    assert r['policies'][1]['goal_rate'] == 1.
    assert r['policies'][1]['mean_action_change'] == 10.
    assert all(a['region']=='FULL_SIMPLEX' for a in r['audits'][0]['attempts'])
    with pytest.raises(ValueError): summarize(records*2, ['a'], np.ones(4), .2, 1.)
    with pytest.raises(ValueError): summarize(records, ['a'], [1,1,1,0], .2, 1.)


def test_fourth_objective_changes_admission_and_uses_frozen_scale():
    idle = dict(motives=[0,0,0,0], undiscounted_motives=[0,0,0,0], goals=0)
    policy = dict(motives=[0,1,0,-3], undiscounted_motives=[0,2,0,-6], goals=1)
    records = [dict(context_seed=9, outcomes={'zero_action':idle,'p':policy})]
    failed = summarize(records, ['p'], np.ones(4), .2, 1.)
    passed = summarize(records, ['p'], [1,1,1,10], .2, 1.)
    assert failed['policy_context_admission_rate'] == 0.
    assert passed['policy_context_admission_rate'] == 1.
    assert passed['policies'][0]['mean_action_change'] == 6.
    assert passed['policies'][0]['mean_return'] == 2.
    assert passed['policies'][0]['admitted_goal_and_episode_budget_rate'] == 1.

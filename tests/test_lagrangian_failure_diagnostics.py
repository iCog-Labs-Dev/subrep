import pytest
from demo.diagnose_lagrangian_failures import select_episodes, summarize_steps


def test_cost_timing_streak_and_discount():
    steps = [dict(cost=c, reward=1., goal_met=(i==2))
             for i,c in enumerate([0.,2.,3.,0.,4.])]
    result = summarize_steps(steps, .5)
    assert result['total_cost'] == 9
    assert result['discounted_cost'] == 2
    assert result['longest_cost_streak'] == 2
    assert result['first_cost_step'] == 2
    assert result['second_half_cost'] == 7
    assert result['goals'] == 1


def test_zero_cost_and_selection():
    result = summarize_steps([dict(cost=0., reward=0., goal_met=False)], .99)
    assert result['first_cost_step'] is None
    assert result['longest_cost_streak'] == 0
    records = [dict(seed=i,cost=c) for i,c in enumerate([0,5,0,3])]
    assert [r['seed'] for r in select_episodes(records,1)] == [1,3,0]
    with pytest.raises(ValueError):
        select_episodes(records + records,1)

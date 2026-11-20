import pytest
from utils.diagnose_contextual_goals import classify


def setup(goals=1,cost=0,region='FULL_SIMPLEX',admit=True):
    execution={'outcomes':{'zero_action':{'goals':0,'undiscounted_motives':[0,0,0]},'ppo':{'goals':goals,'undiscounted_motives':[-cost,1,1]},'other':{'goals':0,'undiscounted_motives':[0,0,0]}}}
    cert=dict(weight_region_type=region,wx_support_values=[.1,.9,.1],delta_r=1,delta_n=[0,0,0])
    library={'skills':{'other:a':{'certificate':dict(cert,weight_region_type='FULL_SIMPLEX',delta_r=2)}}}
    if admit:library['skills']['ppo:b']={'certificate':cert}
    return execution,library,dict(scenario='balanced',weight=[1/3]*3,policy='other',fallback=False)


@pytest.mark.parametrize('kwargs,category',[
    ({'goals':0},'no_goal_reaching_candidate'),
    ({'cost':2},'goal_candidates_exceed_cost_limit'),
    ({},'suitable_eligible_candidate_not_selected'),
    ({'region':'MDN_WX'},'suitable_candidates_outside_region'),
    ({'admit':False},'suitable_candidates_rejected')])
def test_categories(kwargs,category):
    e,l,s=setup(**kwargs)
    assert classify(s,e,l,1)['category']==category


def test_success_threshold_and_replay():
    e,l,s=setup(cost=1)
    del l['skills']['other:a'];s['policy']='ppo'
    r=classify(s,e,l,1)
    assert r['category']=='goal_reached' and r['selected_meets_goal_and_cost']
    s['policy']='other'
    with pytest.raises(ValueError,match='ranking'):classify(s,e,l,1)

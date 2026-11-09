import numpy as np
import pytest
from utils.analyze_contextual_baselines import analyze_query


def outcome(task,cost=0):
    return dict(motives=[-cost,task,0],undiscounted_motives=[-cost,task,0],goals=int(task>0))


def test_evidence_selection_execution_scoring_and_region_filter():
    evidence={'outcomes':{'zero_action':outcome(0),'ppo':outcome(3),'ppo_lagrangian':outcome(1)}}
    execution={'outcomes':{'zero_action':outcome(0),'ppo':outcome(-4,8),'ppo_lagrangian':outcome(2)}}
    library={'skills':{
        'ppo:a':{'certificate':{'weight_region_type':'MDN_WX','wx_support_values':[.1,.9,.1]}},
        'ppo_lagrangian:b':{'certificate':{'weight_region_type':'FULL_SIMPLEX'}}}}
    selection=dict(scenario='balanced',weight=[1/3]*3,policy='ppo_lagrangian',fallback=False)
    rows,a=analyze_query(evidence,execution,library,selection,np.ones(3))
    r={x['method']:x for x in rows}
    assert r['without_certification']['safety_cost']==8
    assert r['subrep']['safety_cost']==0
    assert r['random_candidate']['safety_cost']==4
    assert a['outside_region_selection']==1 and a['rejected_selection']==0
    assert r['without_certification']['score'] < 0
    assert r['random_certified']['score']==r['subrep']['score']
    with pytest.raises(ValueError,match='differs'):
        analyze_query(evidence,execution,library,{**selection,'policy':'ppo'},np.ones(3))


def test_empty_library_fallback_and_rejection():
    evidence={'outcomes':{'zero_action':outcome(0),'ppo':outcome(3),'ppo_lagrangian':outcome(1)}}
    rows,a=analyze_query(evidence,evidence,{'skills':{}},dict(scenario='task',weight=[.1,.8,.1],policy='zero_action',fallback=True),np.ones(3))
    r={x['method']:x for x in rows}
    assert r['subrep']['fallback']==r['random_certified']['fallback']==1
    assert r['zero_action']['fallback']==0
    assert r['zero_action']['idle_execution']==1
    assert a['rejected_selection']==1

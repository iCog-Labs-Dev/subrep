import numpy as np
import pytest
from utils.analyze_four_objective_baselines import query, pareto_mask


def outcome(task,cost=0):
    return dict(motives=[-cost,task,-1,-1],undiscounted_motives=[-cost,task,-1,-1],goals=int(task>0))


def test_evidence_selection_execution_measurement_and_random_expectation():
    e={'outcomes':{'zero_action':outcome(0),'ppo':outcome(10),'lag':outcome(1)}}
    x={'outcomes':{'zero_action':outcome(0),'ppo':outcome(-20,9),'lag':outcome(2)}}
    audits=[dict(policy='ppo',admitted=False,region=None),dict(policy='lag',admitted=True,region='FULL_SIMPLEX')]
    selection=dict(weight=[.25]*4,policy='lag',fallback=False)
    rows,ab=query(e,x,audits,selection,np.ones(4),[.8]*4,{'always_ppo':'ppo','always_ppo_lagrangian':'lag'},1)
    r={a['method']:a for a in rows}
    assert r['without_certification']['task_return']==-20  # No execution oracle.
    assert r['subrep']['task_return']==2
    assert r['random_candidate']['task_return']==-9
    assert ab==dict(blocked=1.,rejected=1.,outside_region=0.)
    assert r['subrep']['successful_reuse']==1
    with pytest.raises(ValueError,match='disagree'):
        query(e,x,audits,{**selection,'policy':'ppo'},np.ones(4),[.8]*4,{'always_ppo':'ppo'},1)


def test_region_exclusion_is_not_gate_rejection_and_fallback_not_reuse():
    e={'outcomes':{'zero_action':outcome(0),'ppo':outcome(10),'lag':outcome(1)}}
    audits=[dict(policy='ppo',admitted=True,region='MDN_WX'),dict(policy='lag',admitted=False,region=None)]
    rows,ab=query(e,e,audits,dict(weight=[.8,.1,.05,.05],policy='zero_action',fallback=True),np.ones(4),[.3,.8,.8,.8],{'always_ppo':'ppo'},1)
    assert ab==dict(blocked=1.,rejected=0.,outside_region=1.)
    sub=next(r for r in rows if r['method']=='subrep')
    assert sub['fallback']==1 and sub['successful_reuse']==0


def test_four_dimensional_pareto_and_ties():
    assert pareto_mask([[1,0,0,0],[1,0,0,0],[0,0,0,0],[0,1,0,0]])==[True,True,False,True]

import numpy as np
import pytest
from utils.safety_objectives import action_change
from utils.contextual_safety_benchmark import priority_queries,certify
from data_collector.collect_safety_scaling import export


def test_change_metric():
    lo=np.array([-1.,-2.]);hi=np.array([1.,4.])
    assert action_change(hi,None,lo,hi)==0
    assert action_change(hi,hi,lo,hi)==0
    assert action_change(hi,lo,lo,hi)==1
    assert action_change(np.zeros(2),hi,lo,hi)==pytest.approx(.25)
    with pytest.raises(ValueError):action_change(hi*2,None,lo,hi)


def test_four_weights_and_gate():
    assert len(priority_queries(4))==5
    for _,w in priority_queries(4):assert len(w)==4 and sum(w)==pytest.approx(1)
    result,_=certify(0,np.array([-1,1,1,1]),.2,[.1,.8,.8,.8])
    assert result['region']=='MDN_WX'


def test_paired_export(tmp_path):
    records=[dict(context_seed=1,initial_observation=[1,2],outcomes={'zero_action':{'motives':[0,0,0,0]},'ppo':{'motives':[-1,2,-3,-4]}})]
    export(records,tmp_path,.99)
    a=np.load(tmp_path/'objectives3/context_1.npz');b=np.load(tmp_path/'objectives4/context_1.npz')
    assert np.array_equal(a['candidate_motives'],b['candidate_motives'][:,:3])
    assert b['candidate_motives'].shape==(2,4)


def test_collector_resets_previous_action():
    from types import SimpleNamespace
    from utils.safety_reuse_benchmark import Collector,Config
    class Env:
        def reset(self,seed):return np.zeros(2),{}
        def step(self,action):return np.zeros(2),1.,0.,False,False,{}
    c=Collector.__new__(Collector);c.env=Env();c.config=Config('unused',objectives=4,max_steps=2)
    c.low=-np.ones(2);c.high=np.ones(2)
    c.policies={'constant':lambda obs:np.ones(2)}
    records=c.collect([1,2])
    for r in records:
        v=r['outcomes']['constant']['motives']
        assert len(v)==4 and v[3]==0 and v[2]<0


def test_training_penalty_and_checkpoint_selection():
    from pilot.safety_gymnasium_ppo import SafetyPPOConfig
    from pilot.train_safety_gymnasium_ppo import action_penalty,_checkpoint_rank
    config=SafetyPPOConfig(effort_penalty=.01,smoothness_penalty=.1)
    assert action_penalty(np.ones(2),-np.ones(2),-np.ones(2),np.ones(2),config)==pytest.approx(.11)
    assert action_penalty(np.ones(2),None,-np.ones(2),np.ones(2),config)==pytest.approx(.01)
    e=dict(goal_reached_rate=1,mean_return=2,mean_cost=0,mean_effort=10,mean_action_change=5)
    assert _checkpoint_rank(e,config) > _checkpoint_rank(dict(e,mean_action_change=10),config)

import numpy as np
from utils.analyze_priority_reuse import schedules,query


def test_paths():
    for w in schedules(3).values():
        assert np.all(w>=0) and np.allclose(w.sum(1),1)
        assert np.allclose(w[0],w[-1])
    assert len(schedules(3)['gradual'])==10


def test_reuse_is_not_availability_or_idle():
    execution={'outcomes':{'zero_action':{'motives':[0,0,0],'undiscounted_motives':[0,0,0],'goals':0},'ppo':{'motives':[0,-1,0],'undiscounted_motives':[0,-1,0],'goals':0}}}
    library={'skills':{'ppo:a':{'certificate':{'weight_region_type':'MDN_WX','wx_support_values':[.1,.9,.1],'delta_r':1,'delta_n':[0,1,0]}}}}
    r=query({},execution,library,np.ones(3),[.1,.8,.1])
    assert r['available']==1 and r['successful_reuse']==0 and r['score']<0
    r=query({},execution,library,np.ones(3),[.8,.1,.1])
    assert r['fallback']==1 and r['successful_reuse']==0 and r['score']==0

import numpy as np
from utils.plot_four_objective_extensions import choose, weight_path


def test_path_is_simplex_and_preserves_endpoints():
    w=weight_path(20)
    assert w.shape==(61,4)
    assert np.all(w>=0) and np.allclose(w.sum(1),1)
    for i in range(4):assert np.isclose(w[20*i,i],.8)


def test_region_gating_changes_selection_without_recertification():
    outcomes={'zero_action':{'motives':[0,0,0,0]},'global':{'motives':[0,1,0,0]},'regional':{'motives':[0,2,0,0]}}
    audits=[dict(policy='global',admitted=True,region='FULL_SIMPLEX'),dict(policy='regional',admitted=True,region='MDN_WX')]
    assert choose(outcomes,audits,np.ones(4),[.3,.9,.9,.9],np.array([.8,.1,.05,.05]))==('global',False)
    assert choose(outcomes,audits,np.ones(4),[.3,.9,.9,.9],np.ones(4)/4)==('regional',False)
    assert choose(outcomes,[],np.ones(4),[1]*4,np.ones(4)/4)==('zero_action',True)

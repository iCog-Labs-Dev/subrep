import numpy as np
import pytest
from utils.plot_four_objective_comparison import extract,bootstrap,METHODS,PRIORITIES


def records():
    contexts=[]
    for i in range(2):
        audits={m:[dict(policy='p',admitted=True,region='FULL_SIMPLEX')] for m in METHODS}
        selections={m:[dict(priority=p,weight=[.25]*4 if j==0 else [.8 if k==j-1 else .2/3 for k in range(4)],
            policy='p',fallback=False,goals=i,cost=i*2,task_return=i,effort=3,action_change=.5)
            for j,p in enumerate(PRIORITIES)] for m in METHODS}
        contexts.append(dict(context_seed=i,caps=[1]*4,audits=audits,selections=selections,coverage={'full_plus_mdn':1.}))
    return contexts


def test_denominators_and_context_pairing():
    a=extract(records(),1.)
    assert a['admissions'].tolist()==[[2,0,0]]*3
    assert a['selected'].tolist()==[[10,0,0]]*3
    assert a['values'][:,:,:,0].mean()==.5
    assert a['values'][:,:,:,2].mean()==0
    indices=np.array([[0,0],[1,1],[0,1]])
    mean,lo,hi=bootstrap(a['values'].mean(axis=2),indices)
    assert np.allclose(mean[0],mean[1]) and np.allclose(lo[0],lo[1])
    assert hi[0,0]>.9 and lo[0,0]<.1  # Resample contexts, not ten repeated queries.
    with pytest.raises(ValueError):extract(records()*2,1.)


def test_rejects_unauthorized_regional_selection():
    r=records()
    r[0]['caps']=[.1,.8,.8,.8]
    r[0]['audits']['full_plus_mdn'][0]['region']='MDN_WX'
    with pytest.raises(ValueError,match='outside caps'):extract(r,1.)


def test_plot_outputs(tmp_path):
    import json
    from utils.plot_four_objective_comparison import run
    (tmp_path/'support_comparison.json').write_text(json.dumps({'contexts':records()}))
    (tmp_path/'manifest.json').write_text(json.dumps({'settings':{'cost_limit':1.}}))
    stats=run(tmp_path,samples=100)
    assert stats['contexts']==2
    assert len(list((tmp_path/'figures').glob('*.png')))==4
    assert len(list((tmp_path/'figures').glob('*.pdf')))==4
    assert (tmp_path/'figures/README.md').exists()
    with pytest.raises(FileExistsError):run(tmp_path,samples=100)

import torch
import pytest
from generator.train_mdn_margin_support import priority_reuse_loss


def test_inclusion_gradient_expands_caps_that_exclude_useful_query():
    caps=torch.tensor([[.1,.8,.8,.8]],dtype=torch.double,requires_grad=True)
    d=torch.tensor([[[-1.,1.,1.,1.]]],dtype=torch.double)
    r=torch.zeros((1,1),dtype=torch.double)
    w=torch.tensor([[.2,.8,0.,0.]],dtype=torch.double)
    loss,m=priority_reuse_loss(caps,d,r,w,.0,0.)
    loss.backward()
    assert caps.grad[0,0]<0
    assert m['priority_coverage']==0 and m['improved_query_rate']==0
    assert m['potential_improvement_query_rate']==1
    _,m=priority_reuse_loss(torch.tensor([[.3,.8,.8,.8]],dtype=torch.double),d,r,w,0.,0.)
    assert m['improved_query_rate']==1 and m['reuse_gain']==pytest.approx(.6)


def test_margin_gradient_and_finite_difference():
    caps=torch.tensor([[.7,.8,.8]],dtype=torch.double,requires_grad=True)
    d=torch.tensor([[[-1.,1.,1.]]],dtype=torch.double);r=torch.zeros((1,1),dtype=torch.double)
    w=torch.tensor([[.2,.7,.1]],dtype=torch.double)
    loss,_=priority_reuse_loss(caps,d,r,w,0.,0.)
    loss.backward();assert caps.grad[0,0]>0
    assert torch.autograd.gradcheck(lambda c:priority_reuse_loss(c,d,r,w,0.,.1)[0],(caps,),eps=1e-6)


def test_global_choices_retained_and_no_possible_gain_is_reported():
    caps=torch.tensor([[.1,.9]])
    d=torch.tensor([[[1.,1.],[-2.,-2.]]]);r=torch.zeros((1,2));w=torch.tensor([[.8,.2]])
    loss,m=priority_reuse_loss(caps,d,r,w,.2,.1)
    assert torch.isfinite(loss)
    assert m['potential_improvement_query_rate']==0 and m['reuse_gain']==0
    with pytest.raises(ValueError):priority_reuse_loss(caps,d,r,torch.tensor([[1.,1.]]),.2,.1)


def test_cli_priority_mode_preserves_frozen_heads_and_metadata(tmp_path, monkeypatch):
    import json, sys, numpy as np
    from generator.mdn import MotiveDecompositionNetwork
    from generator.train_mdn_margin_support import main
    torch.manual_seed(7)
    model=MotiveDecompositionNetwork(input_dim=4,num_objectives=4,num_skills=8)
    base=tmp_path/'base.pth'
    normalization={'enabled':True,'mean':[0.]*4,'std':[1.]*4}
    torch.save({'model_state_dict':model.state_dict(),'auxiliary_target_normalization':normalization},base)
    data=tmp_path/'data';data.mkdir()
    for i in range(12):
        np.savez(data/f'{i}.npz',context=np.array([i,0,1,2],dtype=np.float32),
            candidate_skill_ids=['idle','candidate'],candidate_payoffs=[0.,0.],
            candidate_motives=[[0,0,0,0],[-1,1,1,1]],values_are_raw=True)
    priorities=tmp_path/'priorities.json';priorities.write_text('[[0.2,0.7,0.05,0.05]]')
    out=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['train','--data-dir',str(data),'--baseline-candidate-id','idle',
        '--base-checkpoint',str(base),'--output-dir',str(out),'--loss-mode','priority-reuse',
        '--priority-file',str(priorities),'--epochs','3'])
    main()
    saved=torch.load(out/'support.pth',weights_only=False)
    for name,value in model.state_dict().items():
        if not name.startswith('support_head.'):
            assert torch.equal(value,saved['model_state_dict'][name])
    assert saved['auxiliary_target_normalization']==normalization
    report=saved['support_training']
    assert report['selection_rule']=='validation reuse gain, then lower loss'
    assert np.allclose(report['priorities'], [[.2,.7,.05,.05]])
    assert 'reuse_gain' in report['test']

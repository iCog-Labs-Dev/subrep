import numpy as np
import pytest
import torch
from generator.train_mdn_margin_support import worst_case, margin_loss
from generator.mdn import MotiveDecompositionNetwork
from utils.support_geometry import worst_case_over_support_region


def test_matches_exact_solver():
    rng = np.random.default_rng(9)
    for _ in range(60):
        d = rng.normal(size=3)
        caps = rng.uniform(.4,1,size=3)
        actual = worst_case(torch.tensor(d),torch.tensor(caps))
        assert float(actual) == pytest.approx(worst_case_over_support_region(d,caps))


def test_cap_gradients_away_from_kinks():
    caps = torch.tensor([[.43,.72,.61]],dtype=torch.double,requires_grad=True)
    d = torch.tensor([[-3.,.2,2.]],dtype=torch.double)
    assert torch.autograd.gradcheck(lambda s:worst_case(d,s), (caps,))


def test_updates_only_support_head_and_reduces_loss():
    torch.manual_seed(3)
    m = MotiveDecompositionNetwork(input_dim=4,num_objectives=3)
    for p in m.parameters():
        p.requires_grad_(False)
    for p in m.support_head.parameters():
        p.requires_grad_(True)
    before = {n:p.detach().clone() for n,p in m.named_parameters()}
    x = torch.randn(3,4)
    d = torch.tensor([[[-2.,1.,-2.]]]*3)
    opt = torch.optim.SGD(m.support_head.parameters(),lr=.01)
    def loss():
        return margin_loss(m.forward_inference(x)[1],d,0.,.1)[0]
    initial = float(loss().detach())
    opt.zero_grad(); loss().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.support_head.parameters())
    opt.step()
    assert float(loss().detach()) < initial
    for n,p in m.named_parameters():
        if not n.startswith('support_head.'):
            assert torch.equal(p,before[n])


def test_infeasible_caps_and_empty_eligibility():
    with pytest.raises(ValueError):
        worst_case(torch.ones(3),torch.tensor([.1,.1,.1]))
    caps=torch.tensor([[.7,.7,.7]],requires_grad=True)
    loss,metrics=margin_loss(caps,torch.tensor([[[0.,-1.,0.]]]),.2,.1)
    assert metrics['eligible_pairs']==0
    assert metrics['eligible_pass_rate'] is None
    loss.backward()
    assert torch.isfinite(caps.grad).all()


@pytest.mark.parametrize('m', [2, 4, 8])
def test_solver_multiple_dimensions(m):
    d = torch.linspace(-2, 3, m, dtype=torch.double)
    caps = torch.full((m,), .7, dtype=torch.double)
    assert float(worst_case(d, caps)) == pytest.approx(worst_case_over_support_region(d.numpy(), caps.numpy()))


def test_rejects_objective_broadcasting():
    with pytest.raises(ValueError):
        worst_case(torch.ones(1), torch.ones(3))


def test_cli_roundtrip(tmp_path, monkeypatch):
    import json
    import sys
    from generator.train_mdn_margin_support import main
    from utils.mdn_checkpoint_loader import load_mdn_checkpoint
    data = tmp_path / 'data'
    data.mkdir()
    for i in range(12):
        np.savez(data / f'{i:02}.npz', context=np.array([i, 0, 1, 2],dtype=np.float32),
                 candidate_skill_ids=['zero_action','moving'], values_are_raw=True,
                 candidate_motives=[[0,0,0],[-2,1,-2]], env_id='SafetyPointGoal1-v0',
                 gamma=.99, objective_names=['Safety','Task','ControlEfficiency'])
    scales = tmp_path / 'scales.json'
    scales.write_text(json.dumps({'divisors':[1,1,1]}))
    base = tmp_path / 'base.pth'
    model = MotiveDecompositionNetwork(input_dim=4,num_objectives=3)
    torch.save({'model_state_dict':model.state_dict()}, base)
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['train', '--data-dir',str(data),'--scales-file',str(scales),
        '--base-checkpoint',str(base),'--output-dir',str(output),'--epochs','2'])
    main()
    loaded = load_mdn_checkpoint(output / 'support.pth')
    for n,p in model.named_parameters():
        if not n.startswith('support_head.'):
            assert torch.equal(p,dict(loaded.named_parameters())[n])
    summary=json.loads((output/'summary.json').read_text())
    groups=[set(v) for v in summary['splits'].values()]
    assert sum(map(len,groups)) == len(set.union(*groups)) == 12
    assert summary['best_epoch'] in (0,1,2)
    with pytest.raises(FileExistsError):
        main()

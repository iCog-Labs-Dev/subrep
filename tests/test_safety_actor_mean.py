import numpy as np
import pytest
import torch
from pilot.safety_gymnasium_ppo import SafetyPPOPilot
from pilot.train_safety_gymnasium_ppo import initialize_from_checkpoint


def make(kind):
    return SafetyPPOPilot(observation_dim=3, action_dim=1, action_low=[-1], action_high=[1], actor_mean_type=kind)


@pytest.mark.parametrize('kind', ['tanh', 'linear'])
def test_architecture_roundtrip(tmp_path, kind):
    model = make(kind)
    path = tmp_path/'model.pt'
    model.save(path)
    loaded = SafetyPPOPilot.load(path)
    assert loaded.actor_mean_type == kind
    obs = torch.randn(10, 3)
    for a,b in zip(model(obs), loaded(obs)):
        torch.testing.assert_close(a,b)


def test_old_checkpoint_defaults_to_tanh(tmp_path):
    model = make('tanh'); path=tmp_path/'legacy.pt';model.save(path)
    payload=torch.load(path);del payload['actor_mean_type'];torch.save(payload,path)
    loaded=SafetyPPOPilot.load(path)
    assert loaded.actor_mean_type=='tanh'
    torch.testing.assert_close(model(torch.zeros(3))[0], loaded(torch.zeros(3))[0])


def test_linear_removes_output_saturation_and_preserves_raw_logprob():
    model=make('linear')
    with torch.no_grad():
        model.actor_mean.weight.zero_();model.actor_mean.bias.fill_(8.)
    mean,_,_=model(torch.zeros(3));mean.sum().backward()
    torch.testing.assert_close(model.actor_mean.bias.grad, torch.ones(1))
    clipped, raw, old, _=model.act(np.zeros(3,dtype=np.float32),deterministic=True,return_raw=True)
    assert raw[0]==8. and clipped[0]==1.
    new,_,_=model.evaluate_actions(torch.zeros(3), torch.tensor(raw))
    assert new.item()==pytest.approx(old)


def test_cross_architecture_warm_start_rejected(tmp_path):
    path=tmp_path/'model.pt';make('tanh').save(path,metadata={'environment':'test'})
    with pytest.raises(ValueError,match='mismatch'):
        initialize_from_checkpoint(make('linear'),path,'test')

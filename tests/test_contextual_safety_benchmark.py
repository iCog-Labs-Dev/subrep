import json
import numpy as np
import pytest
from types import SimpleNamespace
from utils.contextual_safety_benchmark import certify, context_key, eligible, run_contextual
from utils.safety_reuse_benchmark import Config


def test_sequence():
    for dr, dn, caps, region, gate in [
        (1., [0,0,0], None, 'FULL_SIMPLEX','CDS'),
        (0., [-.1,0,0], None, 'FULL_SIMPLEX','PDS'),
        (0., [-1,1,1], [.2,.8,.8], 'MDN_WX','CDS'),
        (0., [-1,.1,.1], [.2,.8,.8], 'MDN_WX','PDS')]:
        result, attempts = certify(dr, np.array(dn), .2, caps)
        assert (result['region'],result['gate']) == (region,gate)
    result, attempts = certify(0., np.array([-1,-1,-1]), .2)
    assert result is None
    assert [a['status'] for a in attempts] == ['failed','failed','unavailable','unavailable']
    with pytest.raises(ValueError):
        certify(0., np.array([-1,-1,-1]), .2, [.1,.1,.1])


def test_context_and_weights():
    cfg = Config('unused')
    a = dict(context_seed=1, initial_observation=[0.,1.], observation_schema='native_flat_v1')
    b = {**a, 'initial_observation': [1.,1.]}
    full = SimpleNamespace(wx_support_values=None)
    mdn = SimpleNamespace(wx_support_values=(.2,.8,.8))
    library = SimpleNamespace(get_admitted_skills=lambda: [full,mdn])
    assert eligible(library,context_key(a,cfg),context_key(b,cfg),[.8,.1,.1]) == []
    assert eligible(library,'x','x',[.8,.1,.1]) == [full]
    assert eligible(library,'x','x',[.1,.45,.45]) == [full,mdn]


def test_run(tmp_path, monkeypatch):
    import utils.contextual_safety_benchmark as module
    class Collector:
        def __init__(self, config, models):
            self.env = SimpleNamespace(reset=lambda seed: (np.array([float(seed),0.]), {}))
            self.policies = {n: lambda obs: np.zeros(2) for n in ['zero_action','ppo']}
        def collect(self, seeds, *, repeat=0):
            records = []
            for s in seeds:
                if repeat:
                    assert (tmp_path/'run'/f'seed_42/{s}/selection.json').exists()
                motives = [0.,1.,0.] if s % 2 else [-3.,-1.,-2.]
                records.append(dict(context_seed=s, initial_observation=[float(s),0.],
                    observation_schema='native_flat_v1', repeat=repeat,
                    outcomes={n: {'motives': [0.,0.,0.] if n=='zero_action' else motives, 'goals':0}
                              for n in self.policies}))
            return records
        def close(self): pass
    monkeypatch.setattr(module,'Collector',Collector)
    monkeypatch.setattr(module,'load_models',lambda config: ({42:{}}, []))
    run_contextual(Config(str(tmp_path/'run'),training_seeds=(42,),development_contexts=2,contexts=2))
    audits = json.loads((tmp_path/'run/admissions.json').read_text())
    assert len(audits)==2 and sum(a['admitted'] for a in audits)==1
    assert audits[0]['skill_id'] != audits[1]['skill_id']

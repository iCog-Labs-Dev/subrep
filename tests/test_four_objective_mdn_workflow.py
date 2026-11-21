import json
import numpy as np
from demo.run_four_objective_mdn import export_training, training_commands, OBJECTIVES
from utils.mdn_data_adapter import candidate_set_directory_to_prepared_candidate_outcomes


def test_export_keeps_raw_support_values_and_adapter_scales_once(tmp_path):
    records=[dict(context_seed=1,initial_observation=[1,2],outcomes={
        'zero_action':dict(motives=[-1,2,0,0]),'ppo':dict(motives=[-3,6,-8,-10]),
        'lagrangian':dict(motives=[-1,4,-4,-5])})]
    export_training(records,tmp_path/'data',[2,2,4,5])
    with np.load(tmp_path/'data/context_1.npz') as d:
        assert np.array_equal(d['candidate_motives'][1],[-3,6,-8,-10])
        assert bool(d['values_are_raw'])
        assert len(d['candidate_skill_ids'])==3
    outcomes=candidate_set_directory_to_prepared_candidate_outcomes(tmp_path/'data',baseline_candidate_id='zero_action')
    p=next(o for o in outcomes if np.isclose(o.payoff,3))
    assert np.allclose(p.motives,[-1.5,3,-2,-2])
    assert p.metadata['same_context_baseline']['baseline_payoff']==1.


def test_training_commands_keep_matching_payoff_scale_and_support_settings(tmp_path):
    (tmp_path/'scales.json').write_text(json.dumps(dict(objective_names=OBJECTIVES,divisors=[2,7,4,5])))
    mdn,support=training_commands(tmp_path,.2,.1,500,42)
    assert '--baseline-candidate-id' in mdn
    assert support[support.index('--payoff-scale')+1]=='7.0'
    assert support[support.index('--epsilon')+1]=='0.2'
    assert support[support.index('--data-dir')+1]==mdn[mdn.index('--data-dir')+1]


def test_collect_stage_uses_disjoint_contexts_and_only_requested_policies(tmp_path, monkeypatch):
    import sys
    import demo.run_four_objective_mdn as workflow
    checkpoints=[]
    for name in ('ppo','lag1','lag2','lag3'):
        p=tmp_path/f'{name}.pt';p.write_bytes(b'fake');checkpoints.append(p)
    class Model:
        def eval(self): return self
    class FakeCollector:
        def __init__(self, config, models):
            self.policies={'zero_action':None,'random':None,**models}
        def collect(self, seeds):
            return [dict(context_seed=s, initial_observation=[s,1], outcomes={
                n:dict(motives=[0,0,0,0] if n=='zero_action' else [-1,2,-3,-4])
                for n in self.policies}) for s in seeds]
        def close(self): pass
    monkeypatch.setattr(workflow,'Collector',FakeCollector)
    monkeypatch.setattr(workflow.torch,'load',lambda *a,**k: {'metadata':{'environment':'SafetyPointGoal1-v0'}})
    monkeypatch.setattr(workflow.SafetyPPOPilot,'load',lambda *a,**k: Model())
    out=tmp_path/'run'
    argv=['run','--stage','collect','--output',str(out),'--scale-contexts','10','--contexts','10']
    for p in checkpoints: argv+=['--checkpoint',str(p)]
    monkeypatch.setattr(sys,'argv',argv)
    workflow.main()
    m=json.loads((out/'manifest.json').read_text())
    assert not set(m['scale_context_seeds']) & set(m['training_context_seeds'])
    assert len(list((out/'candidate_sets').glob('*.npz')))==10
    with np.load(next((out/'candidate_sets').glob('*.npz'))) as d:
        assert list(d['candidate_skill_ids'])==['zero_action','ppo','lag1','lag2','lag3']
        assert d['candidate_motives'].shape==(5,4)

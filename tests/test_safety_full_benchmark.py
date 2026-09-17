import json
import numpy as np
import pytest
from utils.safety_full_benchmark import make_plan, evaluate, run_benchmark, METHODS
from utils.safety_objectives import FrozenObjectiveScales
from data_collector.collect_safety_gymnasium_rollouts import SafetyGymnasiumRolloutCollector
from tests.test_safety_three_pipeline import factory

SCALES=FrozenObjectiveScales((1,1,1),'test')


def record(values,seed=1):
    return {'candidate_skill_ids':np.array(['zero_action','careful','fast']),
            'candidate_motives':np.array(values,dtype=float),'context_seed':seed}


def test_heldout_outcomes_do_not_change_selection():
    plan=make_plan([record([[0,0,0],[1,1,0],[-5,2,-2]])],SCALES,4,12,0.)
    before=json.dumps(plan)
    rows=evaluate(plan,[record([[0,0,0],[-9,-9,-9],[10,10,10]])]*4,SCALES)
    assert json.dumps(plan)==before
    chosen=[r for r in rows if r['method']=='best_admissible' and r['scenario']=='balanced']
    assert all(r['selected_skill_id']=='careful' and r['execution_threshold_violation'] for r in chosen)
    assert all(r['selection_valid'] for r in chosen)
    assert len(rows)==7*4*5


def test_empty_admission_falls_back_and_is_reproducible():
    data=[record([[0,0,0],[-1,-1,-1],[-2,-2,-2]])]
    p=make_plan(data,SCALES,4,42,0.)
    assert p==make_plan(data,SCALES,4,42,0.)
    rows=[r for r in p['rows'] if r['method'] in ('best_admissible','random_admissible')]
    assert all(r['abstained'] and r['selected_skill_id']=='zero_action' for r in rows)
    assert all(r['reuse_eligibility'] is None for r in rows)


def test_pds_negative_threshold_and_priority_changes():
    p=make_plan([record([[0,0,0],[1,0,-.1],[0,1,-1.1]])],SCALES,4,42,.2)
    assert p['audit'][0]['gate']=='PDS'
    assert p['audit'][0]['epsilon']==.2
    rows=[r for r in p['rows'] if r['method']=='best_admissible']
    assert all(r['selection_valid'] for r in rows)
    gradual=[r['weights'] for r in rows if r['scenario']=='gradual']
    assert gradual[0]!=gradual[1]!=gradual[-1]
    abrupt=[r['weights'] for r in rows if r['scenario']=='abrupt']
    assert abrupt[0]==abrupt[1] and abrupt[1]!=abrupt[2]


def test_complete_runner_separates_seeds_and_writes_reports(tmp_path):
    def collector(**kwargs):
        return SafetyGymnasiumRolloutCollector(env_factory=factory,**kwargs)
    output=tmp_path/'benchmark'
    summary=run_benchmark(output,contexts=2,batches=2,max_steps=3,collector_class=collector)
    assert len(summary)==7*len(METHODS)
    assert all(len(s['seed_results'])==2 for s in summary)
    seed_sets=[]
    for folder in [output/'development',output/'batch_0/evidence',output/'batch_0/evaluation',output/'batch_1/evidence',output/'batch_1/evaluation']:
        seeds=set()
        for path in folder.glob('*.npz'):
            with np.load(path) as r:
                seeds.add(int(r['context_seed']))
                assert r['candidate_motives'].shape[1]==3
        assert len(seeds)==2
        assert all(seeds.isdisjoint(previous) for previous in seed_sets)
        seed_sets.append(seeds)
    assert (output/'report.md').exists()
    assert (output/'batch_0/selection_plan.json').exists()
    with pytest.raises(FileExistsError):
        run_benchmark(output,contexts=2,batches=1,collector_class=collector)


def test_ppo_preset_and_explicit_overrides():
    from demo.run_safety_full_benchmark import parse_args
    args=parse_args(['--output','unused','--preset','ppo-pds'])
    assert args['epsilon']==.2
    assert args['ppo_checkpoint'].endswith('safety_ppo_point_goal_seed42_updates50.pt')
    assert args['seed']==10000
    assert parse_args(['--output','unused','--preset','ppo-pds','--epsilon','.3'])['epsilon']==.3
    assert parse_args(['--output','unused'])['epsilon']==0.
    with pytest.raises(SystemExit):
        parse_args(['--output','unused','--ppo-checkpoint','missing-checkpoint.pt'])

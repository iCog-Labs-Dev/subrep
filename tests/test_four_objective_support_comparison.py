import json
import numpy as np
import pytest
from demo.compare_four_objective_support import region_coverage, check_seeds, evaluate_evidence, summarize, METHODS
from utils.safety_reuse_benchmark import Config
from library.skill_library import SkillLibrary


def test_four_dimensional_volume_and_seed_leakage():
    assert region_coverage([1]*4)==pytest.approx(1)
    assert region_coverage([.25]*4)==pytest.approx(0,abs=1e-12)
    assert region_coverage([.8]*4)==pytest.approx(.968)
    points=np.random.default_rng(1).dirichlet(np.ones(4),100000)
    caps=np.array([.1,.8,.8,.8])
    assert region_coverage(caps)==pytest.approx(np.all(points<=caps,axis=1).mean(),abs=.005)
    with pytest.raises(ValueError):region_coverage([.1]*4)
    m=dict(scale_context_seeds=[1,2],training_context_seeds=[3,4])
    check_seeds(m,[5,6])
    with pytest.raises(ValueError):check_seeds(m,[4,5])


def test_regional_certificate_roundtrip_and_full_simplex_priority(tmp_path):
    # Specialist loses on safety, but improves all other motives.
    record=dict(context_seed=10,initial_observation=[0.,1.],observation_schema='native_flat_v1',outcomes={
        'zero_action':dict(motives=[0,0,0,0]),
        'universal':dict(motives=[0,.01,0,0]),
        'specialist':dict(motives=[-2,.2,1,1])})
    policies={name:lambda obs:None for name in record['outcomes']}
    audits,selections,libs=evaluate_evidence(record,np.ones(4),np.ones(4),np.array([.1,.8,.8,.8]),np.array([.8]*4),
        Config('unused',objectives=4,epsilon=.2),policies)
    assert [a['admitted'] for a in audits['full_simplex']]==[True,False]
    assert [a['region'] for a in audits['full_plus_mdn']]==['FULL_SIMPLEX','MDN_WX']
    assert audits['full_plus_fixed'][1]['region'] is None
    # Safety priority lies outside the learned region: retain global skill.
    assert selections['full_plus_mdn'][1]['policy']=='universal'
    assert selections['full_plus_mdn'][2]['policy']=='specialist'
    path=tmp_path/'library.json';libs['full_plus_mdn'].save(str(path))
    loaded=SkillLibrary();loaded.load(str(path))
    entries=loaded.get_admitted_skills()
    cert=next(e.certificate for e in entries if e.weight_region_type=='MDN_WX')
    assert cert.wx_support_values==pytest.approx([.1,.8,.8,.8])
    assert len(cert.delta_n)==4 and cert.certification_context==(0.,1.)
    assert cert.mdn_alpha==(1.,1.,1.,1.)


def test_summary_includes_fallback_and_keeps_admission_separate():
    context=dict(audits={},selections={},coverage={m:1. for m in METHODS})
    for m in METHODS:
        context['audits'][m]=[dict(admitted=True,region='FULL_SIMPLEX',gate='CDS'),dict(admitted=False,region=None,gate=None)]
        context['selections'][m]=[dict(fallback=True,goals=0,cost=0,task_return=0,effort=0,action_change=0),
            dict(fallback=False,goals=1,cost=5,task_return=2,effort=2,action_change=3)]
    s=summarize([context],1.)['full_simplex']
    assert s['admission_rate']==.5 and s['context_coverage']==1
    assert s['fallback_rate']==.5 and s['goal_rate']==.5
    assert s['mean_cost']==2.5 and s['admitted_goal_budget_rate']==0


def test_fixed_regions_are_not_mislabelled_and_outside_region_falls_back():
    record=dict(context_seed=10,initial_observation=[0.,1.],observation_schema='native_flat_v1',outcomes={
        'zero_action':dict(motives=[0,0,0,0]),'specialist':dict(motives=[-2,.2,1,1])})
    caps=np.array([.1,.8,.8,.8])
    audits,selections,_=evaluate_evidence(record,np.ones(4),np.ones(4),caps,caps,
        Config('unused',objectives=4,epsilon=.2),{'specialist':lambda obs:None})
    assert audits['full_plus_fixed'][0]['region']=='FIXED_REGION'
    assert all(a['region']!='MDN_WX' for a in audits['full_plus_fixed'][0]['attempts'])
    for method in ('full_plus_mdn','full_plus_fixed'):
        assert selections[method][1]['fallback'] is True
        assert selections[method][1]['policy']=='zero_action'
        assert selections[method][2]['policy']=='specialist'

import json
from dataclasses import replace

import numpy as np
import pytest

from library.skill_library import SkillLibrary
from utils.safety_reuse_benchmark import (
    Config, METHODS, control_effort, crossed_interval, evaluate, fit_scales,
    make_plan, run, seed_plan,
)


EFFECTS = {'zero_action': [0,0,0], 'ppo': [-.1,1,-.2],
           'ppo_lagrangian': [-.1,0,-.1], 'bad': [-3,-1,-2]}


def records(seeds, effects=EFFECTS):
    return [{'context_seed': s, 'outcomes': {
        name: {'motives': vector, 'undiscounted_motives': vector,
               'goals': int(name=='ppo'), 'steps': 2, 'stop_reason': 'max_steps'}
        for name, vector in effects.items()}} for s in seeds]


def test_real_library_admission_queries_and_fixed_baselines(tmp_path, monkeypatch):
    calls = []
    original = SkillLibrary.query_admissible
    def query(self, w, **kwargs):
        calls.append(w)
        return original(self, w, **kwargs)
    monkeypatch.setattr(SkillLibrary, 'query_admissible', query)
    config = Config(str(tmp_path), contexts=2)
    policies = {n: lambda obs: obs for n in EFFECTS}
    plan = make_plan(records([1,2]), np.ones(3), policies, config, 7, tmp_path)
    assert len(calls) == 12  # six main scenarios, two episodes
    assert {a['skill_id']: a['gate'] for a in plan['audit']} == {
        'ppo': 'CDS', 'ppo_lagrangian': 'PDS', 'bad': None}
    library = SkillLibrary()
    library.load(str(tmp_path/'library.json'))
    assert library.count() == 2
    assert (tmp_path/'certificates.json').exists()
    for row in plan['rows']:
        if row['method']=='always_ppo':
            assert row['selected_skill']=='ppo'
        if row['method']=='always_ppo_lagrangian':
            assert row['selected_skill']=='ppo_lagrangian'
        if row['scenario']=='empty_library_control' and row['method']=='subrep':
            assert row['abstained'] and not row['certified_execution']
            assert row['selected_skill']=='zero_action'
    frozen = json.dumps(plan)
    heldout = records([100,101], {**EFFECTS, 'ppo': [-50,-10,-50]})
    rows = evaluate(plan, heldout, np.ones(3), config)
    assert json.dumps(plan) == frozen
    executed = [r for r in rows if r['method']=='subrep' and r['scenario']=='balanced']
    assert all(r['certificate_violation'] for r in executed)
    assert all(r['goal_success'] for r in executed)  # actual events, despite negative reward
    fallback = [r for r in rows if r['method']=='subrep' and r['scenario']=='empty_library_control']
    assert all(r['certificate_violation'] is None for r in fallback)


class FakeCollector:
    seen = []
    def __init__(self, config, models):
        self.config = config
        self.policies = {n: lambda obs: obs for n in EFFECTS}
    def collect(self, seeds):
        splits = seed_plan(self.config)
        for batch, split in enumerate(splits['batches']):
            if seeds == split['evaluation']:
                # Selection artifacts must exist before evaluation is touched.
                for training_seed in self.config.training_seeds:
                    path = __import__('pathlib').Path(self.config.output)/f'seed_{training_seed}/batch_{batch}/selection_plan.json'
                    if path.exists():
                        break
                else:
                    raise AssertionError('Evaluation before selection')
        self.seen.append(tuple(seeds))
        return records(seeds)
    def close(self):
        pass


def test_complete_run_and_fallback_metrics(tmp_path):
    cfg = Config(str(tmp_path/'run'), training_seeds=(42,43), development_contexts=2,
                 contexts=2, batches=2, bootstrap_samples=20)
    summary = run(cfg, collector_class=FakeCollector, models={42:{},43:{}})
    assert len(summary)==7*len(METHODS)
    fallback = next(r for r in summary if r['scenario']=='empty_library_control' and r['method']=='subrep')
    assert fallback['abstained']==1
    assert fallback['certified_executions']==0
    assert fallback['certified_only']['certificate_violation'] is None
    ppo = next(r for r in summary if r['scenario']=='balanced' and r['method']=='always_ppo')
    assert ppo['difference_vs_subrep_ci95']==[0.,0.]
    assert ppo['goal_success']==1
    assert (tmp_path/'run/report.md').exists()
    with pytest.raises(FileExistsError):
        run(cfg, collector_class=FakeCollector, models={42:{},43:{}})


def test_splits_scaling_and_paired_bootstrap():
    plan = seed_plan(Config('unused'))
    sets = [set(plan['development'])]+[set(b[k]) for b in plan['batches'] for k in ('evidence','evaluation')]
    assert sum(map(len,sets))==len(set.union(*sets))
    scales = fit_scales([records([1,2])])
    np.testing.assert_allclose(scales, np.mean(np.abs(list(EFFECTS.values())[1:]), axis=0))
    assert crossed_interval(np.full((3,2,4),-2.),30,np.random.default_rng(1))==[-2.,-2.]
    assert control_effort(np.array([1.,-1.]),np.array([-1.,-1.]),np.array([1.,1.]))==1
    with pytest.raises(ValueError):
        control_effort(np.array([2.]),np.array([-1.]),np.array([1.]))


def test_empty_natural_library(tmp_path):
    cfg = Config(str(tmp_path), contexts=2, epsilon=0.)
    bad = {n: [0.,0.,0.] if n=='zero_action' else [-1.,-1.,-1.] for n in EFFECTS}
    plan = make_plan(records([1,2],bad), np.ones(3), {n: lambda obs: obs for n in bad},cfg,1,tmp_path)
    assert not any(a['admitted'] for a in plan['audit'])
    assert all(r['abstained'] for r in plan['rows'] if r['method']=='subrep')


def test_cli_and_config():
    from demo.run_safety_reuse_benchmark import parse_args
    cfg = parse_args(['--output','unused','--training-seeds','42','44','--epsilon','0'])
    assert cfg.training_seeds==(42,44) and cfg.epsilon==0
    with pytest.raises(ValueError):
        replace(cfg, training_seeds=(42,42)).validate()


def test_native_collector_goal_events_and_effort(monkeypatch):
    import sys
    from types import SimpleNamespace
    from utils.safety_reuse_benchmark import Collector
    class Env:
        observation_space = SimpleNamespace(shape=(2,))
        action_space = SimpleNamespace(low=np.array([-1.,-1.]), high=np.array([1.,1.]))
        def reset(self, seed):
            self.steps = 0
            return np.zeros(2), {}
        def step(self, action):
            self.steps += 1
            # Positive reward at every step but only one actual goal event.
            return np.zeros(2), 2., 1., False, self.steps==2, {'goal_met': self.steps==2}
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, 'safety_gymnasium', SimpleNamespace(make=lambda _: Env()))
    collector = Collector(Config('unused', gamma=.5, max_steps=3), {})
    result = collector.collect([10])[0]['outcomes']
    np.testing.assert_allclose(result['zero_action']['motives'], [-1.5,3.,0.])
    np.testing.assert_allclose(result['axis0_positive']['motives'], [-1.5,3.,-.35**2/2*1.5])
    assert result['zero_action']['goals']==1
    assert result['zero_action']['steps']==2
    assert collector.collect([10])[0]['outcomes']==result


def test_checkpoint_seed_and_duplicate_validation(tmp_path, monkeypatch):
    import torch
    from utils.safety_reuse_benchmark import load_models, SafetyPPOPilot
    monkeypatch.setattr(SafetyPPOPilot, 'load', lambda *args, **kwargs: object())
    cfg = Config('unused', training_seeds=(42,43),
                 ppo_template=str(tmp_path/'ppo{seed}.pt'),
                 lagrangian_template=str(tmp_path/'lag{seed}.pt'))
    for seed in cfg.training_seeds:
        for template, lag in [(cfg.ppo_template,False),(cfg.lagrangian_template,True)]:
            torch.save({'state_dict': {'weight': torch.tensor([1.])},
                        'metadata': {'environment':cfg.env_id,'config':{'seed':seed,'use_lagrangian':lag}}},
                       template.format(seed=seed))
    with pytest.raises(ValueError, match='Duplicate trained parameters'):
        load_models(cfg)
    payload = torch.load(cfg.ppo_template.format(seed=42), weights_only=False)
    payload['metadata']['config']['seed'] = 0
    torch.save(payload, cfg.ppo_template.format(seed=42))
    with pytest.raises(ValueError, match='metadata mismatch'):
        load_models(cfg)

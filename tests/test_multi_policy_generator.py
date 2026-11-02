"""Focused tests for new policy-conditioned generator and strict raw_mixed pipeline."""
import numpy as np
import pytest
import torch
from generator.skill_generator import PolicyConditionedSkillGenerator
from generator.policy_registry import POLICY_NAMES, policy_index
from generator.multi_policy_data import load_records, load_or_create_manifest, partition, group_contexts
from generator.multi_policy_pipeline import fit, evaluate, predict, save_model, load_model
from generator.candidate_filter import select_top_k, execute_selected


def make_data(tmp_path, n=16):
    directory = tmp_path/'raw_mixed'
    directory.mkdir()
    for seed in range(n):
        for policy_id, name in enumerate(POLICY_NAMES):
            obs = np.array([seed/100,0,0,0,0,0,0,0], dtype=np.float32)
            np.savez(directory/f'{name}_seed{seed}.npz', obs=obs, context_seed=seed,
                     skill_id=name, payoff=float(policy_id+seed/10),
                     motives=np.array([policy_id/10, -policy_id], dtype=np.float32),
                     terminated=True)
    return directory


def test_policy_conditioning_and_shapes():
    torch.manual_seed(4)
    model = PolicyConditionedSkillGenerator()
    obs = torch.zeros(8)
    p, m = model(obs, 0)
    assert p.shape == (1,) and m.shape == (2,)
    pb, mb = model(torch.zeros(2, 8), torch.tensor([0, 1]))
    assert pb.shape == (2, 1) and mb.shape == (2, 2)
    with pytest.raises(ValueError):
        model(obs, 99)
    with pytest.raises(ValueError):
        model(obs, 1.2)
    (pb.sum()+mb.sum()).backward()
    assert model.policy_embedding.weight.grad is not None
    assert not torch.allclose(pb[0], pb[1]) or not torch.allclose(mb[0], mb[1])


def test_grouped_manifest_and_fresh_schema(tmp_path):
    path = make_data(tmp_path)
    rows = load_records(str(path))
    manifest_path = tmp_path/'manifest.json'
    m = load_or_create_manifest(rows, str(manifest_path), seed=2)
    assert m == load_or_create_manifest(rows, str(manifest_path), seed=99)
    parts = partition(rows, m)
    assert not ({r['context_id'] for r in parts['train']} & {r['context_id'] for r in parts['test']})
    assert len(group_contexts(parts['test'])) > 0
    # Changing data after split creation must not silently reuse the manifest.
    (path/f'{POLICY_NAMES[0]}_seed0.npz').unlink()
    with pytest.raises(ValueError):
        load_or_create_manifest(load_records(str(path)), str(manifest_path))


def test_training_evaluation_and_checkpoint(tmp_path):
    rows = load_records(str(make_data(tmp_path)))
    manifest = load_or_create_manifest(rows, str(tmp_path/'manifest.json'))
    parts = partition(rows, manifest)
    model, _ = fit(parts['train'], parts['val'], epochs=2, patience=2)
    path = tmp_path/'model.pt'
    save_model(model, path, conditioned=True)
    restored, conditioned = load_model(path)
    assert conditioned
    np.testing.assert_allclose(predict(model, parts['test'], True),
                               predict(restored, parts['test'], True))
    report = evaluate(parts['test'], parts['train'], {'multi_policy': (model, True)})
    assert 'fuel_mae' in report['multi_policy']['per_policy'][POLICY_NAMES[0]]
    assert 'per_policy_training_mean' in report


def test_filter_never_certifies_predictions():
    model = PolicyConditionedSkillGenerator()
    selected = select_top_k(model, np.zeros(8), POLICY_NAMES, 3)
    assert len(selected) == 3
    calls = []
    def execute(policy):
        calls.append(policy)
        return -100.0, np.array([-5.0, -10.0]), True
    def certify(measured_payoff, measured_motives):
        assert measured_payoff == -100.0
        return {'cds_admit': False, 'pds_admit': False}
    results = execute_selected(selected, {p:p for p in POLICY_NAMES}, execute, certify)
    assert calls == selected
    assert all(not r['certification']['cds_admit'] for r in results)
    with pytest.raises(ValueError):
        select_top_k(model, np.zeros(8), POLICY_NAMES, 10)


def test_offline_metrics_use_measured_outcomes():
    from generator.evaluate_candidate_filter import assess
    rows = [dict(policy=name, payoff=10., motives=np.array([1., -1.]), terminated=True)
            for name in POLICY_NAMES]
    result = assess({'seed:1':rows}, {'seed:1':list(POLICY_NAMES[:3])},
                    {'baseline_payoff':0., 'baseline_motives':[0., -2.]})
    assert result['executed'] == 3
    assert result['avoided'] == 6
    assert result['certifiable_found'] == 3
    assert result['certifiable_missed'] == 6
    assert result['execution_savings_fraction'] == pytest.approx(2/3)
    assert result['missed_certifiable_fraction'] == pytest.approx(2/3)
    

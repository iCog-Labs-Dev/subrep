"""Multi-policy report coverage without running a real environment."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from generator.evaluate_generator_report import build_multi_policy_report, run_multi_policy_report
from generator.policy_registry import POLICY_NAMES
from generator.multi_policy_pipeline import save_model
from generator.skill_generator import SkillGenerator, PolicyConditionedSkillGenerator


def _rows(contexts):
    return [dict(context_id=f'seed:{seed}', seed=seed, policy=name, policy_id=i,
                 obs=np.array([seed / 100, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                 payoff=float(i), motives=np.array([i / 10, -i], dtype=np.float32))
            for seed in contexts for i, name in enumerate(POLICY_NAMES)]


def test_report_uses_all_policies_and_measured_certification():
    conditioned = PolicyConditionedSkillGenerator()
    baseline = SkillGenerator()
    for model in (conditioned, baseline):
        for param in model.parameters():
            torch.nn.init.constant_(param, 0.0)
    report = build_multi_policy_report(_rows([10]), _rows([1, 2]), conditioned, baseline,
                                       {'baseline_payoff': 0., 'baseline_motives': [0., -10.]})
    assert report['n_contexts'] == 1
    assert report['n_candidate_outcomes'] == len(POLICY_NAMES)
    assert set(report['measured_certification_per_policy']) == set(POLICY_NAMES)
    assert set(report['prediction_metrics']) == {'multi_policy', 'observation_only', 'per_policy_training_mean'}
    assert 'fuel_mae' in report['prediction_metrics']['multi_policy']['overall']
    assert report['measured_certification_per_policy']['ppo_deterministic']['measured_payoff_mean'] == 0.


def test_report_measured_certification_ignores_predictions():
    class WrongPredictor(torch.nn.Module):
        def forward(self, obs, policy_ids):
            return torch.full((len(obs), 1), 100000.), torch.full((len(obs), 2), 100000.)

    class WrongBaseline(torch.nn.Module):
        def forward(self, obs):
            return torch.full((len(obs), 1), 100000.), torch.full((len(obs), 2), 100000.)

    rows = _rows([11])
    for row in rows:
        row['payoff'] = -100.
        row['motives'] = np.array([-100., -100.])
    report = build_multi_policy_report(rows, _rows([1]), WrongPredictor(), WrongBaseline(),
                                       {'baseline_payoff': 0., 'baseline_motives': [0., 0.]})
    assert all(v['joint_certifiable_count'] == 0
               for v in report['measured_certification_per_policy'].values())


def test_cli_report_reuses_manifest_and_only_test_contexts(tmp_path):
    raw = tmp_path / 'raw_mixed'
    raw.mkdir()
    for row in _rows(range(16)):
        np.savez(raw / f"{row['policy']}_{row['seed']}.npz", obs=row['obs'],
                 payoff=row['payoff'], motives=row['motives'],
                 context_seed=row['seed'], skill_id=row['policy'])
    model_path = tmp_path / 'model.pt'
    baseline_path = tmp_path / 'baseline.pt'
    save_model(PolicyConditionedSkillGenerator(), model_path, conditioned=True)
    save_model(SkillGenerator(), baseline_path, conditioned=False)
    stats = tmp_path / 'idle.json'
    stats.write_text(json.dumps({'baseline_payoff': 0., 'baseline_motives': [0., -10.]}))
    manifest = tmp_path / 'split.json'
    out = tmp_path / 'report.json'
    args = SimpleNamespace(data_dir=str(raw), split_manifest=str(manifest), seed=42,
                           model_path=str(model_path), observation_baseline_path=str(baseline_path),
                           baseline_stats=str(stats), baseline_episodes=1, baseline_seed=42,
                           output=str(out))
    result = run_multi_policy_report(args)
    assert out.exists()
    assert json.loads(out.read_text())['n_contexts'] == result['n_contexts']
    assert result['n_contexts'] < 16
    assert result['n_contexts'] > 0
    assert run_multi_policy_report(args)['n_contexts'] == result['n_contexts']

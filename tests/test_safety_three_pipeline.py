import json
import numpy as np
import pytest
from tests.test_safety_gymnasium_adapter import _make_fake_env
from env.safety_gymnasium_wrapper import SafeRLGymnasiumEnv
from data_collector.collect_safety_gymnasium_rollouts import SafetyGymnasiumRolloutCollector
from utils.safety_objectives import FrozenObjectiveScales
from utils.safety_gymnasium_pipeline import run_safety_gymnasium_certification_pipeline, _load_rollout_record


def factory(**kwargs):
    return SafeRLGymnasiumEnv(make_env=_make_fake_env, **kwargs)


def test_three_objectives_survive_collection_certificates_and_reports(tmp_path):
    scales = tmp_path/'scales.json'
    FrozenObjectiveScales((2,4,0.5),'development-test-only').save(scales)
    collector = SafetyGymnasiumRolloutCollector(objectives=3, scales_file=str(scales),
        env_factory=factory, save_dir=str(tmp_path/'rollouts'),max_steps=3)
    try:
        records = collector.collect(1)
    finally:
        collector.close()
    raw = records[0]
    assert raw['candidate_motives'].shape == (7,3)
    assert np.any(raw['candidate_motives'][:,2] < 0)
    path = next((tmp_path/'rollouts').glob('*.npz'))
    loaded = _load_rollout_record(path)
    np.testing.assert_allclose(loaded['candidate_motives'], raw['candidate_motives']/[2,4,.5])
    result = run_safety_gymnasium_certification_pipeline(rollout_dir=tmp_path/'rollouts',expected_objectives=3,
        cert_file=tmp_path/'cert.metta', library_file=tmp_path/'lib.json',
        report_json_path=tmp_path/'report.json', report_md_path=tmp_path/'report.md')
    assert len(result.stats['audit_entries']) == 6
    assert all(len(r['delta_n']) == 3 for r in result.stats['audit_entries'])
    assert result.library.count() > 0
    assert all(len(e.certificate.delta_n) == 3 for e in result.library.get_admitted_skills())
    assert 'ControlEfficiency' in (tmp_path/'report.md').read_text()
    assert json.loads((tmp_path/'report.json').read_text())['objective_metadata']['objective_scales'] == [2,4,.5]
    from utils.safety_gymnasium_certification_ablation import build_safety_gymnasium_certification_ablation
    summary = build_safety_gymnasium_certification_ablation(rollout_dirs=[tmp_path/'rollouts'],
        task_weight=(.1,.8,.1), safety_weight=(.8,.1,.1), summary_json_path=tmp_path/'ablation.json')
    assert summary['total_contexts'] == 1
    assert len(summary['queries']['task_focused']['without_certification_selections'][0]['delta_n']) == 3
    from utils.safety_gymnasium_pareto import collect_pareto_points
    points = collect_pareto_points(rollout_dirs=[tmp_path/'rollouts'], pattern='*.npz',
        baseline_candidate_id='zero_action', pds_epsilon=1, selection_weight=(.1,.8,.1))
    assert len(points['subrep'][0]['raw_objective_returns']) == 3
    from utils.safety_gymnasium_reuse_curve import build_safety_gymnasium_reuse_curve
    reuse = build_safety_gymnasium_reuse_curve(rollout_dirs=[tmp_path/'rollouts'],
        output_path=tmp_path/'reuse.png', summary_json_path=tmp_path/'reuse.json', control_weight=.2)
    assert reuse['curve'][0]['objective_weights'][2] == .2
    assert len(reuse['curve'][0]['selections'][0]['delta_n']) == 3



def test_old_two_objective_data_cannot_be_requested_as_three(tmp_path):
    from tests.test_safety_gymnasium_pipeline import _write_rollout_file
    _write_rollout_file(tmp_path/'old.npz')
    with pytest.raises(ValueError, match='fresh rollouts'):
        run_safety_gymnasium_certification_pipeline(rollout_dir=tmp_path, expected_objectives=3)


def test_three_objective_data_requires_measurement_metadata(tmp_path):
    np.savez(tmp_path/'invalid.npz',candidate_motives=np.zeros((2,3)),candidate_payoffs=np.zeros(2))
    with pytest.raises(ValueError, match='freshly collected'):
        _load_rollout_record(tmp_path/'invalid.npz')

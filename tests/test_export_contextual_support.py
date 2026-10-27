import json

import pytest

from generator.export_contextual_support import export_support
from utils.weight_set_store import WeightSetStore


def fixture_run(tmp_path):
    root = tmp_path / 'benchmark'

    def save(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    save('manifest.json', {'config': {'training_seeds': [42], 'epsilon': .2}})
    save('seed_42/scales.json', [1, 1, 1])
    save('seed_42/evidence.json', [{
        'context_seed': 100, 'initial_observation': [.1, .2],
        'observation_schema': 'native_flat_v1',
        'outcomes': {'zero_action': {'motives': [0, 0, 0]},
                     'candidate': {'motives': [0, 1, 0]}}}])
    save('admissions.json', [{'training_seed': 42, 'context_seed': 100,
        'skill_id': 'candidate:hash:context', 'policy_id': 'candidate:hash',
        'admitted': True, 'delta_r': 1, 'delta_n': [0, 1, 0],
        'attempts': [{'status': 'passed', 'region': 'FULL_SIMPLEX', 'gate': 'CDS'}]}])
    save('seed_42/100/selection.json', [
        {'fallback': False, 'skill_id': 'candidate:hash:context',
         'policy': 'candidate', 'weight': [.8, .1, .1]},
        {'fallback': True, 'policy': 'zero_action', 'weight': [.1, .8, .1]}])
    return root


def test_export_preserves_observed_weights_and_excludes_fallback(tmp_path):
    root = fixture_run(tmp_path)
    summary = export_support(root, tmp_path / 'export')
    assert summary['contexts'] == 1
    assert summary['fallbacks_excluded'] == 1
    assert not summary['ready_for_split']
    store = WeightSetStore.load(tmp_path / 'export/weights.json')
    assert store.total_vertex_count() == 1
    assert store.get_support_values([.1, .2]).tolist() == pytest.approx([.8, .1, .1])


def test_export_rejects_evidence_mismatch(tmp_path):
    root = fixture_run(tmp_path)
    path = root / 'admissions.json'
    audits = json.loads(path.read_text())
    audits[0]['delta_r'] = 5
    path.write_text(json.dumps(audits))
    with pytest.raises(ValueError, match='measured evidence'):
        export_support(root, tmp_path / 'export')


def test_export_rejects_self_generated_mdn_admissions(tmp_path):
    root = fixture_run(tmp_path)
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['mdn_checkpoint'] = 'model.pth'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='without an MDN'):
        export_support(root, tmp_path / 'export')

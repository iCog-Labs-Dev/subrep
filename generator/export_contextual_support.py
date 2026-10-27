"""Export observed full-simplex-certified selections, without running environments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from utils.weight_set_store import WeightSetStore


def export_support(benchmark, output):
    root, destination = Path(benchmark), Path(output)
    if destination.exists():
        raise FileExistsError(f'Use a new output directory: {destination}')
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    manifest = read(root / 'manifest.json')
    config = manifest['config']
    # Avoid learning targets from the model's own admissions in this bootstrap export.
    if manifest.get('mdn_checkpoint'):
        raise ValueError('Export requires a benchmark without an MDN checkpoint')
    audits = read(root / 'admissions.json')
    index = {(a['training_seed'], a['context_seed'], a['skill_id']): a for a in audits}
    store = WeightSetStore(3)
    observations = []
    skipped = 0
    dimensions = set()
    for seed in config['training_seeds']:
        folder = root / f'seed_{seed}'
        scales = np.asarray(read(folder / 'scales.json'), dtype=float)
        if scales.shape != (3,) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('Expected three finite positive motive scales')
        for record in read(folder / 'evidence.json'):
            context = np.asarray(record['initial_observation'], dtype=float)
            if record['observation_schema'] != 'native_flat_v1' or context.ndim != 1:
                raise ValueError('Expected native flat starting observations')
            dimensions.add(len(context))
            for selection in read(folder / str(record['context_seed']) / 'selection.json'):
                if selection['fallback']:
                    skipped += 1
                    continue
                audit = index[(seed, record['context_seed'], selection['skill_id'])]
                passed = next((a for a in audit['attempts'] if a['status'] == 'passed'), None)
                if not audit['admitted'] or not passed or passed['region'] != 'FULL_SIMPLEX':
                    raise ValueError('Selection lacks a full-simplex certificate')
                policy = selection['policy']
                if audit['policy_id'].split(':')[0] != policy or policy == 'zero_action':
                    raise ValueError('Selection policy does not match candidate evidence')
                outcomes = record['outcomes']
                dn = (np.asarray(outcomes[policy]['motives']) -
                      np.asarray(outcomes['zero_action']['motives'])) / scales
                dr = float(dn[1])  # Existing benchmark payoff definition, unchanged.
                if not np.allclose(dn, audit['delta_n']) or not np.isclose(dr, audit['delta_r']):
                    raise ValueError('Certificate improvements do not match measured evidence')
                gates = {'CDS': CDSGate(), 'PDS': PDSGate(config['epsilon'])}
                if not gates[passed['gate']].admit(dr, dn):
                    raise ValueError('Saved certificate fails rechecking on measured evidence')
                store.observe_certified_weight(context, selection['weight'])
                observations.append({'training_seed': seed, 'context_seed': record['context_seed'],
                    'skill_id': selection['skill_id'], 'weight': selection['weight'],
                    'delta_r': dr, 'delta_n': dn.tolist(), 'gate': passed['gate']})
    if len(dimensions) != 1:
        raise ValueError('Inconsistent observation dimensions')
    destination.mkdir(parents=True)
    store.save(destination / 'weights.json')
    targets = [target for _, target in store.get_all_support_targets()]
    summary = {'source': str(root.resolve()), 'source_sha256': sources,
        'contexts': store.context_count(), 'observations': len(observations),
        'fallbacks_excluded': skipped, 'ready_for_split': store.context_count() >= 3,
        'constant_targets': bool(targets) and all(np.allclose(t, targets[0]) for t in targets),
        'semantics': 'Coordinate maxima of observed certified-selection priorities; not safe-region boundaries.',
        'scope': 'Support fitting only; use separate benchmark contexts for final evaluation.',
        'observed_selections': observations}
    (destination / 'manifest.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('observed_selections', 'source_sha256')}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    export_support(args.benchmark, args.output)


if __name__ == '__main__':
    main()

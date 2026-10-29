"""Strict raw_mixed-only loader and context-grouped persistent splitting."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from generator.policy_registry import POLICY_NAMES, policy_index


def load_records(data_dir: str = 'data/raw_mixed') -> list[dict]:
    paths = sorted(Path(data_dir).glob('*.npz'))
    if not paths:
        raise FileNotFoundError(f'No .npz records in {data_dir}; collect fresh raw_mixed data first')
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            needed = {'obs', 'payoff', 'motives', 'skill_id', 'context_seed'}
            if not needed.issubset(z.files):
                raise ValueError(f'{path}: missing {sorted(needed - set(z.files))}; recollect fresh raw_mixed data')
            obs = np.asarray(z['obs'], dtype=np.float32).reshape(-1)
            motives = np.asarray(z['motives'], dtype=np.float32).reshape(-1)
            if obs.shape != (8,) or motives.shape != (2,):
                raise ValueError(f'{path}: expected obs=(8,), motives=(2,)')
            name = str(z['skill_id'].item())
            seed = int(z['context_seed'])
            if not np.isfinite(obs).all() or not np.isfinite(motives).all() or not np.isfinite(float(z['payoff'])):
                raise ValueError(f'{path}: nonfinite outcome')
            records.append(dict(file=path.name, context_id=f'seed:{seed}', seed=seed,
                                policy=name, policy_id=policy_index(name), obs=obs,
                                payoff=float(z['payoff']), motives=motives,
                                terminated=bool(z['terminated']) if 'terminated' in z.files else None))
    seen = set()
    for r in records:
        key = (r['context_id'], r['policy'])
        if key in seen:
            raise ValueError(f'Duplicate context/policy {key}; use a fresh empty collection directory')
        seen.add(key)
    groups = group_contexts(records, complete=False)
    for context, rows in groups.items():
        if not all(np.allclose(rows[0]['obs'], r['obs'], rtol=1e-6, atol=1e-6) for r in rows):
            raise ValueError(f'Inconsistent starting observations for {context}')
    return records


def group_contexts(records: list[dict], complete: bool = True) -> dict[str, list[dict]]:
    groups = {}
    for r in records:
        groups.setdefault(r['context_id'], []).append(r)
    if complete:
        expected = set(POLICY_NAMES)
        for context, rows in groups.items():
            got = {r['policy'] for r in rows}
            if got != expected or len(rows) != len(expected):
                raise ValueError(f'{context}: incomplete candidate set: missing {sorted(expected-got)}')
    return groups


def load_or_create_manifest(records: list[dict], path: str = 'data/mixed_split_manifest.json',
                            seed: int = 42, fractions=(0.75, 0.125, 0.125)) -> dict:
    import random
    groups = group_contexts(records, complete=True)
    keys = sorted(groups)
    signature = hashlib.sha256(json.dumps({k: sorted(r['file'] for r in v) for k, v in groups.items()},
                                          sort_keys=True).encode()).hexdigest()
    p = Path(path)
    if p.exists():
        manifest = json.loads(p.read_text())
        if manifest.get('dataset_signature') != signature or set(manifest['assignment']) != set(keys):
            raise ValueError('Manifest does not match raw_mixed dataset; use a new manifest path for new data')
        return manifest
    if len(keys) < 8:
        raise ValueError('At least 8 complete contexts required for train/val/test')
    if abs(sum(fractions)-1) > 1e-8:
        raise ValueError('Split fractions must sum to 1')
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_train, n_val = round(n*fractions[0]), round(n*fractions[1])
    if min(n_train, n_val, n-n_train-n_val) < 1:
        raise ValueError('Empty train/validation/test split')
    assignment = {key: ('train' if i < n_train else 'val' if i < n_train+n_val else 'test')
                  for i, key in enumerate(keys)}
    manifest = dict(seed=seed, fractions=list(fractions), dataset_signature=signature,
                    assignment=assignment, policy_names=list(POLICY_NAMES))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def partition(records: list[dict], manifest: dict) -> dict[str, list[dict]]:
    buckets = {'train': [], 'val': [], 'test': []}
    for r in records:
        buckets[manifest['assignment'][r['context_id']]].append(r)
    return buckets

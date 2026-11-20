"""Fit frozen scales from a separate raw development collection."""
import argparse
from pathlib import Path
import numpy as np
from utils.safety_objectives import FrozenObjectiveScales


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    deltas = []
    for path in sorted(Path(args.data_dir).glob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            names = data['candidate_skill_ids'].astype(str)
            matches = np.flatnonzero(names == 'zero_action')
            if len(matches) != 1 or not bool(data['values_are_raw']):
                raise ValueError(f'{path}: expected raw data and one idle baseline')
            values = data['candidate_motives']
            deltas.extend(values[names != 'zero_action'] - values[matches[0]])
    FrozenObjectiveScales.fit_development(deltas, source=str(Path(args.data_dir).resolve())).save(args.output)


if __name__ == '__main__':
    main()

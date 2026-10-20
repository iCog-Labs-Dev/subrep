"""Small exact-context benchmark; does not train models."""
import argparse
from utils.safety_reuse_benchmark import Config
from utils.contextual_safety_benchmark import run_contextual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--contexts', type=int, default=4)
    p.add_argument('--development-contexts', type=int, default=10)
    p.add_argument('--max-steps', type=int, default=200)
    p.add_argument('--training-seeds', type=int, nargs='+', default=[42])
    p.add_argument('--seed', type=int, default=90000)
    p.add_argument('--epsilon', type=float, default=.2)
    p.add_argument('--mdn-checkpoint')
    p.add_argument('--mdn-contract')
    args = vars(p.parse_args())
    checkpoint, contract = args.pop('mdn_checkpoint'), args.pop('mdn_contract')
    args['training_seeds'] = tuple(args['training_seeds'])
    run_contextual(Config(**args), checkpoint, contract)


if __name__ == '__main__':
    main()

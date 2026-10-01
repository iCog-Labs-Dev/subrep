"""Run the paired three-objective SubRep SkillLibrary benchmark."""
import argparse
from utils.safety_reuse_benchmark import Config, run


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, help='New output directory; existing directories are rejected')
    p.add_argument('--training-seeds', nargs='+', type=int, default=[42,43,44])
    p.add_argument('--ppo-template', default=Config.ppo_template)
    p.add_argument('--lagrangian-template', default=Config.lagrangian_template)
    p.add_argument('--env-id', default=Config.env_id)
    for name in ('seed', 'development_contexts', 'contexts', 'batches', 'max_steps', 'bootstrap_samples'):
        p.add_argument('--'+name.replace('_','-'), type=int, default=getattr(Config, name))
    p.add_argument('--epsilon', type=float, default=Config.epsilon)
    p.add_argument('--gamma', type=float, default=Config.gamma)
    args = vars(p.parse_args(argv))
    args['training_seeds'] = tuple(args['training_seeds'])
    config = Config(**args)
    config.validate()
    return config


if __name__ == '__main__':
    run(parse_args())

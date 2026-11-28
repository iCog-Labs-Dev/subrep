"""Train the Motive Decomposition Networks that give goal decomposition its priorities.

Each network is fitted to a schedule of weights by situation, a stand-in for
MetaMo, as the village MDN is. The observations come from flying the planned
skills on training seeds, and the fit is checked on seeds the training never saw.

    python -m demo.train_lunar_mdn
    python -m demo.train_lunar_mdn --seeds 20 --epochs 50 --output-dir /tmp/models
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from decomposition.domain import Domain
from decomposition.mdn_priorities import MdnPriorities, Schedule, evaluate_mdn, train_mdn
from env.lunar_beacon import BEACON_MISSION, beacon_facts
from env.lunar_domain import LUNAR_LANDER
from env.lunar_facts import lander_facts

LANDING_CHECKPOINT = Path("models/lunar_mdn.pth")
MISSION_CHECKPOINT = Path("models/lunar_beacon_mdn.pth")
LANDING_PLAN = ("stabilise", "move_over_pad", "slow_descent", "touch_down")
MISSION_PLAN = ("go_to_beacon", "cruise_to_pad", "slow_descent", "touch_down")
TRAINING_SEEDS = range(0, 300)
HELD_OUT_SEEDS = range(1000, 1100)
_KEEP_EVERY = 5  # consecutive frames are nearly the same state


def landing_schedule(observation: np.ndarray) -> Tuple[float, float]:
    """(Safety, Fuel): safety matters more when tilted or close to the ground."""
    facts = lander_facts(observation)
    return (0.8, 0.2) if facts["near_ground"] or not facts["upright"] else (0.7, 0.3)


def mission_schedule(observation: np.ndarray) -> Tuple[float, float, float]:
    """(Safety, Fuel, Mission): the mission matters until the scan is done, then getting down safely."""
    facts = beacon_facts(observation)
    if not facts["beacon_scanned"]:
        return (0.3, 0.1, 0.6)
    return (0.7, 0.2, 0.1) if facts["near_ground"] else (0.6, 0.3, 0.1)


class _Recording:
    """Passes an environment through and keeps the observations it returns."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.observations = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step(self, action: int) -> Any:
        result = self.env.step(action)
        self.observations.append(result[0])
        return result


def collect_observations(domain: Domain, plan: Sequence[str], seeds: Sequence[int]) -> np.ndarray:
    """Observations from flying ``plan`` from every seed, and from doing nothing, which falls fast and tilts."""
    env = _Recording(domain.make_env(0))
    try:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            env.observations.append(observation)
            for name in plan:
                run = domain.run_skill(env, domain.skills[name], observation)
                observation = run.observation
                if run.terminated or run.truncated:
                    break
            observation, _ = env.reset(seed=seed)
            over = False
            while not over:
                _, _, terminated, truncated, _ = env.step(domain.idle_action)
                over = terminated or truncated
    finally:
        env.env.close()
    return np.asarray(env.observations[::_KEEP_EVERY], dtype=np.float32)


def train_one(
    domain: Domain, plan: Sequence[str], schedule: Schedule, seeds: Sequence[int], epochs: int, output: Path
) -> Dict[str, float]:
    observations = collect_observations(domain, plan, seeds)
    source, metrics = train_mdn(observations, schedule, domain.objectives, epochs=epochs)
    held_out = collect_observations(domain, plan, HELD_OUT_SEEDS)
    metrics = {"training_contexts": float(len(observations)), **metrics, **evaluate_mdn(source, held_out, schedule)}
    source.save(output, metrics)
    return metrics


def load_landing_mdn(path: Path = LANDING_CHECKPOINT) -> MdnPriorities:
    return MdnPriorities.load(path, LUNAR_LANDER.objectives, cvar_tail_level=0.1)


def load_mission_mdn(path: Path = MISSION_CHECKPOINT) -> MdnPriorities:
    return MdnPriorities.load(path, BEACON_MISSION.objectives, cvar_tail_level=0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MDNs that give goal decomposition its priorities.")
    parser.add_argument("--seeds", type=int, default=len(TRAINING_SEEDS))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    seeds = range(TRAINING_SEEDS.start, TRAINING_SEEDS.start + args.seeds)
    folder = Path(args.output_dir) if args.output_dir else None
    landing = folder / LANDING_CHECKPOINT.name if folder else LANDING_CHECKPOINT
    mission = folder / MISSION_CHECKPOINT.name if folder else MISSION_CHECKPOINT
    print("landing", train_one(LUNAR_LANDER, LANDING_PLAN, landing_schedule, seeds, args.epochs, landing))
    print("mission", train_one(BEACON_MISSION, MISSION_PLAN, mission_schedule, seeds, args.epochs, mission))


if __name__ == "__main__":
    main()

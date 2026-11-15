"""Collect candidate records on MO-LunarLander for the epsilon-scale study.

Scores three candidate groups against the noop baseline with the repository's
own rollout code (env/lunar_lander_rollout.py: `evaluate_candidates`, gamma 1,
no step cap, the same seed for every candidate):

  task        the extended set (`build_extended_candidate_policies`)
  borderline  PPO with a random action at probability p, and PPO that cuts its
              engines after k steps (defined here; study-only)
  tradeoff    PPO that, when it would coast, fires an extra engine with
              probability p (env/lunar_lander_candidates.py)

Writes one JSON row per (seed, candidate) to runs/epsilon_scale/records.json,
or to the path given as the first argument. Seeds 1-40 take about 4 minutes.

    python experiments/epsilon_scale/collect.py [OUT.json]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data_collector.collect_candidate_sets import (  # noqa: E402
    CandidatePolicy,
    build_extended_candidate_policies,
)
from env.lunar_lander_candidates import build_tradeoff_candidate_policies  # noqa: E402
from env.lunar_lander_rollout import evaluate_candidates  # noqa: E402
from env.lunar_lander_wrapper import SubRepEnv  # noqa: E402
from pilot.rl_pilot import RLPilot  # noqa: E402

CHECKPOINT = str(REPO / "models" / "pilot_ppo.pt")
DEFAULT_OUT = REPO / "runs" / "epsilon_scale" / "records.json"
SEEDS = range(1, 41)


def _ppo_action(pilot: RLPilot, obs: np.ndarray) -> int:
    result = pilot.predict(obs, deterministic=True, return_probability=True)
    return int(result[0] if isinstance(result, tuple) else result)


class PpoRandomAt:
    """PPO, but a uniformly random action with probability p."""

    def __init__(self, pilot: RLPilot, p: float) -> None:
        self.pilot, self.p = pilot, p
        self.rng = np.random.default_rng(0)

    def reset(self, seed=None, **_):
        self.rng = np.random.default_rng(10_000 + int(seed or 0))

    def __call__(self, obs):
        if self.rng.random() < self.p:
            return int(self.rng.integers(0, 4)), 1.0
        return _ppo_action(self.pilot, obs), 1.0


class PpoCutAfter:
    """PPO for k steps, then engines off."""

    def __init__(self, pilot: RLPilot, k: int) -> None:
        self.pilot, self.k, self.t = pilot, k, 0

    def reset(self, **_):
        self.t = 0

    def __call__(self, obs):
        self.t += 1
        return (_ppo_action(self.pilot, obs) if self.t <= self.k else 0), 1.0


def build_sets(env: SubRepEnv) -> Dict[str, Tuple[CandidatePolicy, ...]]:
    task = tuple(build_extended_candidate_policies(env, pilot_checkpoint=CHECKPOINT))
    pilot = RLPilot.load(CHECKPOINT, map_location="cpu")
    borderline = tuple(
        [CandidatePolicy(f"ppo_random_{round(p * 100)}", PpoRandomAt(pilot, p)) for p in (0.1, 0.2, 0.3)]
        + [CandidatePolicy(f"ppo_cut_{k}", PpoCutAfter(pilot, k)) for k in (50, 100, 150)]
    )
    tradeoff = build_tradeoff_candidate_policies(pilot_checkpoint=CHECKPOINT)
    return {"task": task, "borderline": borderline, "tradeoff": tradeoff}


def main(out_path: Path) -> None:
    env = SubRepEnv(seed=0)
    sets = build_sets(env)
    rows = []
    start = time.time()
    for seed in SEEDS:
        for set_name, policies in sets.items():
            baseline_stats, records = evaluate_candidates(env, policies, seed=seed)
            baseline_info = baseline_stats["info"]
            for r in records:
                md = r.metadata
                rows.append({
                    "seed": seed, "set": set_name, "skill_id": r.skill_id,
                    "delta_r": r.delta_r, "delta_n": list(r.delta_n),
                    "landed": md["landing_success"], "crashed": md["crashed"],
                    "truncated": md["truncated"], "steps": md["steps"],
                    "baseline_payoff": float(baseline_stats["baseline_payoff"]),
                    "baseline_motives": [float(x) for x in baseline_stats["baseline_motives"]],
                    "baseline_crashed": baseline_info["crashed"],
                    "baseline_landed": baseline_info["landing_success"],
                })
        print(f"seed {seed:>2} done, {len(rows)} rows, {time.time() - start:.0f}s", flush=True)
    env.close()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows), encoding="utf-8")
    print("saved", out_path)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)

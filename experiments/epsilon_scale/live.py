"""The live loop with the real MetaMo governor, seeds 21-36, epsilon scale on.

Selection and feedback run as the demo runs them (MetaMoController with the
LunarLander epsilon scale, the demo's PDS-only pipeline, the noop baseline
when nothing is admitted), on the task and trade-off candidates. At every step
the same records are also judged under the raw eps, to show what the scale
changes in the live loop: the live eps, the tolerance, which candidates only
the scale admits, and whether the pick differs.

Writes runs/epsilon_scale/live.json, or the path given as the first argument.
The budget gain a1_securing can be overridden with the A1 environment
variable. Needs a MetaMo checkout; takes a few minutes.

    python experiments/epsilon_scale/live.py [OUT.json]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import collect  # noqa: E402
import demo.run_lunar_lander_metamo as demo  # noqa: E402
from bridge.budget import BudgetCoefficients  # noqa: E402
from bridge.controller import MetaMoController  # noqa: E402
from env.lunar_lander_metamo import (  # noqa: E402
    lunar_lander_epsilon_scale,
    make_lunar_lander_governor,
)
from env.lunar_lander_rollout import (  # noqa: E402
    NOOP_BASELINE,
    appraisal_scales,
    episode_outcome,
    evaluate_candidates,
    execute_candidate,
)
from env.lunar_lander_wrapper import SubRepEnv  # noqa: E402

DEFAULT_OUT = REPO / "runs" / "epsilon_scale" / "live.json"
SEEDS = list(range(21, 37))


def main(out_path: Path) -> None:
    env = SubRepEnv(seed=0)
    sets = collect.build_sets(env)
    candidates = sets["task"] + sets["tradeoff"]
    policies = {c.skill_id: c for c in candidates}
    scale = lunar_lander_epsilon_scale()

    first = evaluate_candidates(env, candidates, seed=SEEDS[0])
    payoff_scale, motive_scale = appraisal_scales(first[1])
    a1 = float(os.environ.get("A1", BudgetCoefficients().a1_securing))
    governor = make_lunar_lander_governor(
        payoff_scale=payoff_scale,
        motive_scale=motive_scale,
        cvar_samples=1000,
        coefficients=BudgetCoefficients(a1_securing=a1),
    )
    pipeline = demo.build_pipeline(seed=1, gates="pds", cvar_samples=1000)
    controller = MetaMoController(governor, pipeline, seed=1, epsilon_scale=scale)
    print(f"a1_securing = {a1}  unit = {scale.unit}", flush=True)

    log = []
    for index, seed in enumerate(SEEDS):
        baseline_stats, records = first if index == 0 else evaluate_candidates(env, candidates, seed=seed)
        sig = governor.signal()
        eps = float(sig.pds_epsilon)
        raw_admitted = sorted(r.skill_id for r in records if r.delta_r + min(r.delta_n) >= -eps)
        flown = {}

        def outcome_for(selected, _seed=seed):
            policy = NOOP_BASELINE if selected is None else policies[selected.skill_id]
            delta_r, delta_n, info = execute_candidate(env, policy, seed=_seed)
            flown.update(skill_id=policy.skill_id, landed=bool(info["landing_success"]))
            return episode_outcome(delta_r, delta_n, info, admitted=selected is not None)

        step = controller.step(
            context=np.asarray(baseline_stats["info"]["start_obs"]),
            candidate_skills=records,
            baseline_stats=baseline_stats,
            outcome_for=outcome_for,
            seed=1,
        )
        scaled_admitted = sorted(r.skill_id for r in controller.last_records if r.is_certified)
        log.append({
            "step": index, "seed": seed, "epsilon": eps, "pds_tolerance": step.pds_tolerance,
            "securing": governor.modulator_summary()["securing"],
            "safety_weight": float(sig.weights[0]),
            "task_failures": step.task_failures,
            "scaled_admitted": scaled_admitted, "raw_admitted": raw_admitted,
            "only_scaled": sorted(set(scaled_admitted) - set(raw_admitted)),
            "selected": step.selected_skill_id, "flown": flown,
        })
        print(f"{index:>2} seed {seed} eps {eps:.4f} tol {step.pds_tolerance:6.1f} "
              f"fail {step.task_failures:>2} | admitted {len(scaled_admitted)} (raw {len(raw_admitted)}) "
              f"only-scaled {log[-1]['only_scaled']} | pick {step.selected_skill_id}", flush=True)
    env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    widened = sum(1 for s in log if s["only_scaled"])
    print(f"steps where the scale admits more than raw eps: {widened}/{len(log)}; eps range "
          f"[{min(s['epsilon'] for s in log):.4f}, {max(s['epsilon'] for s in log):.4f}]")
    print("saved", out_path)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)


"""Live policy-conditioned candidate filtering with measured CDS/PDS certification."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from baseline.idle_policy import IdlePolicy
from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from data_collector.collect_candidate_sets import build_extended_candidate_policies
from env.lunar_lander_wrapper import SubRepEnv
from env.skill_executor import SkillExecutor
from generator.candidate_filter import execute_selected, select_top_k
from generator.multi_policy_pipeline import load_model
from generator.policy_registry import POLICY_NAMES


def measure_idle_baseline(episodes: int, seed: int, gamma: float) -> ImprovementCalculator:
    """Measure idle-policy outcomes and initialize baseline comparisons."""
    print(f"Running idle baseline ({episodes} episodes) ...", flush=True)
    env = SubRepEnv(seed=seed)
    stats = IdlePolicy(env=env, gamma=gamma).run_baseline_episodes(
        num_episodes=episodes, seed=seed)
    calculator = ImprovementCalculator(stats)
    print(f"  Baseline payoff  : {calculator.baseline_payoff:.3f}", flush=True)
    print(f"  Baseline motives : {calculator.baseline_motives.tolist()}", flush=True)
    return calculator


def execute_candidate(candidate, env, observation, seed: int, gamma: float, max_steps: int | None):
    """Measure a candidate rollout from the shared starting context."""
    if hasattr(candidate.policy_fn, "reset"):
        candidate.policy_fn.reset(seed=seed)
    initial_obs, _ = env.reset(seed=seed)
    if not np.allclose(initial_obs, observation, rtol=1e-6, atol=1e-6):
        raise RuntimeError("Context reset mismatch")
    executor = SkillExecutor(env=env, policy_fn=candidate.policy_fn,
                             gamma=gamma, max_steps=max_steps)
    return executor.run_episode(initial_obs=initial_obs)


def certify_candidate(payoff, motives, calculator: ImprovementCalculator,
                      cds_gate: CDSGate, pds_gate: PDSGate) -> dict:
    """Apply the unchanged CDS/PDS gates to measured improvements."""
    delta_r, delta_n = calculator.compute_improvements(payoff, motives)
    return {
        "delta_r": float(delta_r),
        "delta_n": delta_n.tolist(),
        "cds_admit": bool(cds_gate.admit(delta_r, delta_n)),
        "pds_admit": bool(pds_gate.admit(delta_r, delta_n)),
    }


def evaluate_contexts(model, env, candidates, calculator, args) -> list[dict]:
    """Select, execute, and certify top-ranked candidates per context."""
    by_name = {candidate.skill_id: candidate for candidate in candidates}
    cds_gate, pds_gate = CDSGate(), PDSGate()
    total = args.contexts * args.top_k
    completed = 0
    results = []

    for index in range(args.contexts):
        seed = args.seed + index + 1
        observation, _ = env.reset(seed=seed)
        selected = select_top_k(model, observation, POLICY_NAMES, args.top_k)
        print(f"\nContext {index + 1}/{args.contexts} (seed={seed})", flush=True)
        print(f"  Selected: {', '.join(selected)}", flush=True)

        def execute(candidate):
            print(f"  Executing {completed + 1}/{total}: {candidate.skill_id} ...", flush=True)
            return execute_candidate(candidate, env, observation, seed, args.gamma, args.max_steps)

        def certify(payoff, motives):
            nonlocal completed
            decision = certify_candidate(payoff, motives, calculator, cds_gate, pds_gate)
            completed += 1
            print(f"    Payoff={payoff:.3f}  Motives={motives.tolist()}  "
                  f"CDS={decision['cds_admit']}  PDS={decision['pds_admit']}", flush=True)
            print(f"    Completed {completed}/{total} executions", flush=True)
            return decision

        outcomes = execute_selected(selected, by_name, execute, certify)
        results.append({"context_seed": seed, "selected": selected, "outcomes": outcomes})
        print(f"  Context {index + 1}/{args.contexts} complete.", flush=True)

    return results


def save_report(results: list[dict], calculator: ImprovementCalculator,
                args, elapsed: float) -> None:
    """Save measured filtering outcomes and print the execution summary."""
    outcomes = [outcome for context in results for outcome in context["outcomes"]]
    executed = len(outcomes)
    possible = args.contexts * len(POLICY_NAMES)
    cds_count = sum(item["certification"]["cds_admit"] for item in outcomes)
    pds_count = sum(item["certification"]["pds_admit"] for item in outcomes)
    joint_count = sum(item["certification"]["cds_admit"] and
                      item["certification"]["pds_admit"] for item in outcomes)

    report = {
        "baseline_source": "measured_idle_execution",
        "baseline_episodes": args.baseline_episodes,
        "baseline_payoff": calculator.baseline_payoff,
        "baseline_motives": calculator.baseline_motives.tolist(),
        "contexts": args.contexts,
        "k": args.top_k,
        "possible_executions": possible,
        "executed": executed,
        "executions_avoided": possible - executed,
        "cds_admitted": cds_count,
        "pds_admitted": pds_count,
        "jointly_certifiable_found": joint_count,
        "elapsed_seconds": round(elapsed, 2),
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("  Live Candidate Filtering Results")
    print("=" * 60)
    print(f"Contexts evaluated  : {args.contexts}")
    print(f"Candidates executed : {executed}/{possible}")
    print(f"Executions avoided  : {possible - executed} ({(possible - executed) / possible:.1%})")
    print(f"CDS admitted        : {cds_count}")
    print(f"PDS admitted        : {pds_count}")
    print(f"Jointly certifiable : {joint_count}")
    print(f"Elapsed time (s)    : {elapsed:.1f}")
    print(f"Full report saved -> {output}")


def main(argv=None):
    """Run live candidate filtering using a policy-conditioned checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/multi_policy_generator.pt")
    parser.add_argument("--pilot-checkpoint", default="models/pilot_ppo.pt")
    parser.add_argument("--baseline-episodes", type=int, default=50)
    parser.add_argument("--contexts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", default="reports/multi_policy/live_filter.json")
    args = parser.parse_args(argv)

    if args.contexts <= 0 or args.baseline_episodes <= 0:
        parser.error("--contexts and --baseline-episodes must be positive")
    if not 1 <= args.top_k <= len(POLICY_NAMES):
        parser.error(f"--top-k must be between 1 and {len(POLICY_NAMES)}")
    if not 0 <= args.gamma <= 1:
        parser.error("--gamma must be between 0 and 1")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    started = time.monotonic()
    print("Loading policy-conditioned model ...", flush=True)
    model, conditioned = load_model(args.model_path)
    if not conditioned:
        raise ValueError("A policy-conditioned checkpoint is required")

    calculator = measure_idle_baseline(args.baseline_episodes, args.seed, args.gamma)
    print(f"Loading candidate policies from {args.pilot_checkpoint} ...", flush=True)
    env = SubRepEnv(seed=args.seed)
    candidates = build_extended_candidate_policies(env, pilot_checkpoint=args.pilot_checkpoint)
    if tuple(candidate.skill_id for candidate in candidates) != tuple(POLICY_NAMES):
        raise ValueError("Runtime candidate registry mismatch")

    print(f"Evaluating {args.contexts} contexts (top-{args.top_k} of {len(POLICY_NAMES)}) ...", flush=True)
    results = evaluate_contexts(model, env, candidates, calculator, args)
    save_report(results, calculator, args, time.monotonic() - started)


if __name__ == "__main__":
    main()

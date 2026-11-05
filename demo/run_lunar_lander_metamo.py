"""End-to-end MetaMo -> SubRep demo on MO-LunarLander.

Runs the full loop on the real environment, one episode per decision step:
MetaMo's state gives the Safety/Fuel weights and the risk budgets (eps, alpha);
SubRep scores every candidate policy against the noop baseline from the same
seed, certifies them under those budgets and selects one; the chosen policy
flies a real episode; and the episode's outcome feeds back into MetaMo.

Run:
    python demo/run_lunar_lander_metamo.py
    python demo/run_lunar_lander_metamo.py --steps 16 --candidates ppo
    python demo/run_lunar_lander_metamo.py --steps 16 --json-out runs/run.json

What to watch:
  * `secur` / `thresh` against `eps` / `alpha`: both budgets tighten as caution
    rises.
  * `Safety` against `Fuel`: the weights SubRep selects with.
  * `adm`, `selected` and `result`: which candidates passed the gates, which
    one flew, and whether it landed.

Current behaviour the summary measures rather than assumes:
  * MetaMo's caution modulators rise on every step and saturate within a few
    steps, pinning eps at 0. The cause is in MetaMo's update step.
  * eps cannot change a LunarLander admission at the current scale: eps is at
    most 0.5, while the objective changes are in the hundreds.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge._loader import find_metamo_root  # noqa: E402
from bridge.budget import BudgetCoefficients  # noqa: E402
from bridge.controller import MetaMoController  # noqa: E402
from bridge.protocol import SkillOutcome  # noqa: E402
from certification.pds_test import PDSGate  # noqa: E402
from data_collector.collect_candidate_sets import (  # noqa: E402
    CandidatePolicy,
    _fixed_action_policy,
    _random_policy,
)
from env.lunar_lander_metamo import (  # noqa: E402
    LUNAR_LANDER_OBJECTIVES,
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
from generator.mdn import MotiveDecompositionNetwork  # noqa: E402
from utils.mdn_runtime_pipeline import (  # noqa: E402
    RuntimeCertificationPipeline,
    RuntimePipelineConfig,
)
from utils.weight_set_store import WeightSetStore  # noqa: E402

PILOT_CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "pilot_ppo.pt"

NUM_OBJECTIVES = len(LUNAR_LANDER_OBJECTIVES)
OBS_DIM = 8
SAFETY = 0
SATURATION_LEVEL = 0.95
EPSILON_MAX = BudgetCoefficients().epsilon_max


# ---------------------------------------------------------------------------
# Building the loop
# ---------------------------------------------------------------------------


def build_candidates(env: SubRepEnv, kind: str) -> Tuple[CandidatePolicy, ...]:
    """The candidate set. `ci` needs no PPO checkpoint; the others do."""
    if kind == "ci":
        return (
            CandidatePolicy("noop", _fixed_action_policy(0)),
            CandidatePolicy("left_engine", _fixed_action_policy(1)),
            CandidatePolicy("main_engine", _fixed_action_policy(2)),
            CandidatePolicy("right_engine", _fixed_action_policy(3)),
            CandidatePolicy("random", _random_policy(env)),
        )
    from data_collector.collect_candidate_sets import (
        build_default_candidate_policies,
        build_extended_candidate_policies,
    )

    build = (
        build_default_candidate_policies
        if kind == "ppo"
        else build_extended_candidate_policies
    )
    return build(env, pilot_checkpoint=str(PILOT_CHECKPOINT))


class RecordingPipeline:
    """Forwards certification unchanged and keeps the records it returns.

    `MetaMoController.step()` reports only counts, so this is how the demo sees
    each candidate's verdict (`is_certified`) as the pipeline decided it,
    including the CVaR gate when it is on.
    """

    def __init__(self, pipeline: RuntimeCertificationPipeline) -> None:
        self._pipeline = pipeline
        self.last_records: List[Any] = []

    def certify_candidate_skills(self, **kwargs: Any) -> List[Any]:
        records = self._pipeline.certify_candidate_skills(**kwargs)
        self.last_records = list(records)
        return records

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)


def build_pipeline(*, seed: int, use_cvar: bool, cvar_samples: int) -> RecordingPipeline:
    """A fresh, seeded pipeline around an untrained MDN.

    The W_x store stays empty (no support training after certification), so
    the PDS gate decides with the full simplex, Δr + min Δn. That is what the
    summary's eps-sensitivity check recomputes.
    """
    import torch

    torch.manual_seed(seed)
    model = MotiveDecompositionNetwork(input_dim=OBS_DIM, num_objectives=NUM_OBJECTIVES)
    model.eval()
    return RecordingPipeline(
        RuntimeCertificationPipeline(
            model=model,
            weight_store=WeightSetStore(num_objectives=NUM_OBJECTIVES),
            config=RuntimePipelineConfig(
                gate_type="PDS",
                use_cvar=use_cvar,
                require_cds_or_cvar=True,
                cvar_samples=cvar_samples,
                train_support_after_certify=False,
            ),
        )
    )


def metamo_commit() -> str:
    """Short commit of the MetaMo checkout the bridge resolved, or "unknown".

    Asks git inside the MetaMo root only when that root is itself a checkout,
    so an empty submodule folder never reports SubRep's own commit.
    """
    root = find_metamo_root()
    if root is None or not (root / ".git").exists():
        return "unknown"
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


# ---------------------------------------------------------------------------
# Per-step bookkeeping
# ---------------------------------------------------------------------------


def pds_margin(delta_r: float, delta_n: Sequence[float]) -> float:
    """Full-simplex PDS margin before eps: Δr + min Δn."""
    return float(delta_r) + float(np.min(np.asarray(delta_n, dtype=np.float64)))


def pds_admitted_count(candidates: Sequence[Dict[str, Any]], epsilon: float) -> int:
    """How many candidates the PDS gate alone admits at `epsilon`."""
    gate = PDSGate(epsilon=epsilon)
    return sum(
        1
        for c in candidates
        if gate.admit(c["delta_r"], np.asarray(c["delta_n"], dtype=np.float64))
    )


def episode_result(info: Dict[str, Any]) -> str:
    """How a flown episode ended."""
    if info.get("landing_success"):
        return "landed"
    if info.get("crashed"):
        return "crashed"
    if info.get("stop_reason") in ("truncated", "max_steps"):
        return "timeout"
    return "ended"


def eps_sensitive(candidates: Sequence[Dict[str, Any]]) -> bool:
    """Whether eps anywhere in [0, eps_max] changes this step's PDS admissions."""
    return pds_admitted_count(candidates, 0.0) != pds_admitted_count(candidates, EPSILON_MAX)


def summarise(steps: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Measured diagnostics over the run."""
    eps = [s["signal"]["epsilon"] for s in steps]
    alpha = [s["signal"]["alpha"] for s in steps]
    securing = [s["modulators"]["securing"] for s in steps]
    saturated_at = next(
        (s["step"] for s in steps if s["modulators"]["securing"] >= SATURATION_LEVEL),
        None,
    )
    return {
        "epsilon_range": [min(eps), max(eps)],
        "alpha_range": [min(alpha), max(alpha)],
        "securing_range": [min(securing), max(securing)],
        "first_weights": steps[0]["signal"]["weights"],
        "last_weights": steps[-1]["signal"]["weights"],
        "admitted_total": sum(s["admitted_count"] for s in steps),
        "scored_total": sum(s["candidate_count"] for s in steps),
        "steps_with_nothing_admitted": sum(1 for s in steps if s["admitted_count"] == 0),
        "selections": dict(sorted(Counter(s["selected"] or "-" for s in steps).items())),
        "results": dict(sorted(Counter(s["result"] for s in steps).items())),
        "securing_saturated_at_step": saturated_at,
        "eps_sensitive_steps": sum(1 for s in steps if eps_sensitive(s["candidates"])),
    }


def _weights_text(weights: Sequence[float]) -> str:
    return "  ".join(
        f"{name}={value:.3f}" for name, value in zip(LUNAR_LANDER_OBJECTIVES, weights)
    )


# ---------------------------------------------------------------------------
# The demo
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end MetaMo -> SubRep loop on MO-LunarLander."
    )
    parser.add_argument("--steps", type=int, default=10, help="decision steps; one episode each")
    parser.add_argument("--seed", type=int, default=1, help="first seed; step k uses seed + k")
    parser.add_argument(
        "--candidates",
        choices=("ci", "ppo", "extended"),
        default=None,
        help="candidate set (default: ppo if the PPO checkpoint exists, else ci)",
    )
    parser.add_argument(
        "--no-selection",
        choices=("noop", "stop"),
        default="noop",
        help="when nothing is admitted: fly the noop baseline, or fly nothing",
    )
    parser.add_argument("--cvar", action="store_true", help="also apply the CVaR gate (OR semantics)")
    parser.add_argument("--cvar-samples", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true", help="print every candidate's verdict")
    parser.add_argument("--json-out", type=Path, default=None, help="save the run as JSON")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    kind = args.candidates or ("ppo" if PILOT_CHECKPOINT.is_file() else "ci")
    if kind != "ci" and not PILOT_CHECKPOINT.is_file():
        print(
            f"error: --candidates {kind} needs the PPO checkpoint at {PILOT_CHECKPOINT}; "
            "use --candidates ci to run without it",
            file=sys.stderr,
        )
        return 2

    seeds = [args.seed + k for k in range(args.steps)]
    env = SubRepEnv(seed=seeds[0])
    try:
        candidates = build_candidates(env, kind)
        policies: Dict[str, CandidatePolicy] = {c.skill_id: c for c in candidates}

        # The appraisal scales come from the first step's candidates and stay
        # fixed: the governor takes them once, at construction.
        first = evaluate_candidates(env, candidates, seed=seeds[0])
        payoff_scale, motive_scale = appraisal_scales(first[1])

        governor = make_lunar_lander_governor(
            payoff_scale=payoff_scale,
            motive_scale=motive_scale,
            cvar_samples=args.cvar_samples,
        )
        pipeline = build_pipeline(
            seed=args.seed, use_cvar=args.cvar, cvar_samples=args.cvar_samples
        )
        controller = MetaMoController(governor, pipeline, seed=args.seed)

        config = {
            "steps": args.steps,
            "seeds": seeds,
            "candidates": kind,
            "candidate_ids": [c.skill_id for c in candidates if c.skill_id != NOOP_BASELINE.skill_id],
            "no_selection": args.no_selection,
            "gates": "PDS + CVaR (OR)" if args.cvar else "PDS",
            "cvar_samples": args.cvar_samples,
            "metamo_commit": metamo_commit(),
            "payoff_scale": payoff_scale,
            "motive_scale": motive_scale,
        }

        print("=" * 100)
        print("MetaMo -> SubRep, MO-LunarLander (2 objectives: Safety, Fuel)")
        print("=" * 100)
        print(f"steps: {args.steps}   seeds: {seeds[0]}..{seeds[-1]}   candidates: {kind} "
              f"({', '.join(config['candidate_ids'])}; baseline: {NOOP_BASELINE.skill_id})")
        print(f"gates: {config['gates']}   nothing admitted: {args.no_selection}   "
              f"MetaMo commit: {config['metamo_commit']}")
        print(f"appraisal scales: payoff={payoff_scale:.3f}  motive={motive_scale:.3f}")
        if not args.cvar:
            print("alpha is reported but unused: no gate reads it without --cvar")

        header = (
            f"{'step':>4} {'seed':>4} {'secur':>6} {'thresh':>6} {'appr':>6} "
            f"{'eps':>7} {'alpha':>6} {'Safety':>6} {'Fuel':>6} {'adm':>5}"
            + (f" {'pds':>4}" if args.cvar else "")
            + f"  {'selected':<22} {'result':<8} {'d_r':>8} {'min d_n':>9}"
        )
        print()
        print(header)
        print("-" * len(header))

        steps: List[Dict[str, Any]] = []
        for index, seed in enumerate(seeds):
            baseline_stats, records = (
                first if index == 0 else evaluate_candidates(env, candidates, seed=seed)
            )
            flown: Dict[str, Any] = {}

            def outcome_for(selected: Optional[Any], _seed: int = seed) -> SkillOutcome:
                if selected is None and args.no_selection == "stop":
                    flown.update(skill_id=None, result="stopped", delta_r=0.0,
                                 delta_n=[0.0] * NUM_OBJECTIVES, steps=0)
                    return SkillOutcome(
                        delta_r=0.0,
                        delta_n=np.zeros(NUM_OBJECTIVES),
                        admitted=False,
                        effort=0.0,
                    )
                policy = NOOP_BASELINE if selected is None else policies[selected.skill_id]
                delta_r, delta_n, info = execute_candidate(env, policy, seed=_seed)
                flown.update(
                    skill_id=policy.skill_id,
                    result=episode_result(info),
                    delta_r=float(delta_r),
                    delta_n=[float(v) for v in delta_n],
                    steps=int(info["steps"]),
                )
                return episode_outcome(delta_r, delta_n, info, admitted=selected is not None)

            record = controller.step(
                context=np.asarray(baseline_stats["info"]["start_obs"]),
                candidate_skills=records,
                baseline_stats=baseline_stats,
                outcome_for=outcome_for,
                seed=args.seed,
            )

            verdicts = {r.skill_id: bool(r.is_certified) for r in pipeline.last_records}
            step_candidates = [
                {
                    "skill_id": r.skill_id,
                    "delta_r": float(r.delta_r),
                    "delta_n": [float(v) for v in r.delta_n],
                    "pds_margin": pds_margin(r.delta_r, r.delta_n),
                    "admitted": verdicts.get(r.skill_id, False),
                    "landing_success": bool(r.metadata.get("landing_success", False)),
                    "crashed": bool(r.metadata.get("crashed", False)),
                }
                for r in records
            ]
            selected = record.selected_skill_id
            step = {
                "step": record.step,
                "seed": seed,
                "signal": {
                    "weights": [float(w) for w in record.weights],
                    "epsilon": float(record.pds_epsilon),
                    "alpha": float(record.cvar_tail_level),
                },
                "modulators": {k: float(v) for k, v in record.modulators.items()},
                "candidate_count": record.candidate_count,
                "admitted_count": record.admitted_count,
                "pds_only_admitted": pds_admitted_count(step_candidates, record.pds_epsilon),
                "candidates": step_candidates,
                "selected": selected,
                "flown_skill_id": flown["skill_id"],
                "result": flown["result"],
                "flown_delta_r": flown["delta_r"],
                "flown_delta_n": flown["delta_n"],
                "flown_steps": flown["steps"],
            }
            steps.append(step)

            if selected is not None:
                shown = selected
            elif flown["skill_id"] is None:
                shown = "STOP"
            else:
                shown = "-"
            mods = step["modulators"]
            weights = step["signal"]["weights"]
            print(
                f"{step['step']:>4} {seed:>4} {mods['securing']:>6.3f} "
                f"{mods['threshold']:>6.3f} {mods['approach']:>6.3f} "
                f"{step['signal']['epsilon']:>7.4f} {step['signal']['alpha']:>6.3f} "
                f"{weights[SAFETY]:>6.3f} {weights[1]:>6.3f} "
                f"{record.admitted_count:>2}/{record.candidate_count:<2}"
                + (f" {step['pds_only_admitted']:>4}" if args.cvar else "")
                + f"  {shown:<22} {flown['result']:<8} "
                f"{flown['delta_r']:>+8.1f} {min(flown['delta_n']):>+9.1f}"
            )
            if args.verbose:
                for c in step_candidates:
                    print(
                        f"{'':>11}{c['skill_id']:<24} d_r={c['delta_r']:>+7.1f}  "
                        f"d_n=[{', '.join(f'{v:+.1f}' for v in c['delta_n'])}]  "
                        f"margin={c['pds_margin']:>+8.1f}  "
                        f"{'admitted' if c['admitted'] else 'rejected'}"
                    )
    finally:
        env.close()

    summary = summarise(steps)
    final_modulators = governor.modulator_summary()

    print("-" * len(header))
    print(f"eps   range: [{summary['epsilon_range'][0]:.4f}, {summary['epsilon_range'][1]:.4f}]   "
          f"alpha range: [{summary['alpha_range'][0]:.4f}, {summary['alpha_range'][1]:.4f}]")
    print(f"weights first: {_weights_text(summary['first_weights'])}   "
          f"last: {_weights_text(summary['last_weights'])}")
    print(f"admitted: {summary['admitted_total']}/{summary['scored_total']} candidate evaluations; "
          f"steps with nothing admitted: {summary['steps_with_nothing_admitted']}/{args.steps}")
    print("selections: " + ", ".join(f"{k}={v}" for k, v in summary["selections"].items()))
    print("flown results: " + ", ".join(f"{k}={v}" for k, v in summary["results"].items()))
    saturated = summary["securing_saturated_at_step"]
    print(
        f"caution: securing at decision time ranged "
        f"[{summary['securing_range'][0]:.3f}, {summary['securing_range'][1]:.3f}]; "
        + (f"first step with securing >= {SATURATION_LEVEL}: {saturated}"
           if saturated is not None
           else f"securing >= {SATURATION_LEVEL}: not reached")
        + f"; after the last step: {final_modulators['securing']:.3f}"
    )
    sensitive = summary["eps_sensitive_steps"]
    if sensitive:
        print(f"eps sensitivity: eps in [0, {EPSILON_MAX}] changes the PDS admissions "
              f"on {sensitive}/{args.steps} steps")
    else:
        print(f"eps sensitivity: eps in [0, {EPSILON_MAX}] did not change any PDS "
              "admission in this run")

    if args.json_out is not None:
        summary["final_modulators"] = {k: float(v) for k, v in final_modulators.items()}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"config": config, "steps": steps, "summary": summary}, indent=2),
            encoding="utf-8",
        )
        print(f"saved: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

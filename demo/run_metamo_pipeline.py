"""End-to-end MetaMo -> SubRep demo on the 6-objective Minecraft stub.

Shows the coupling the paper describes (§4.3): MetaMo emits per-step selection
weights and risk budgets, SubRep certifies and selects under them, and the
executed outcome feeds back into MetaMo's appraisal.

Run:
    python demo/run_metamo_pipeline.py
    python demo/run_metamo_pipeline.py --steps 16 --seed 7

What to watch:
  * `eps` and `alpha` BOTH tighten as `securing` rises. That is the corrected
    sign convention -- under the paper's formulas as written they would move in
    opposite directions. See bridge/budget.py for the full argument.
  * The Safety weight climbs as threat rises and relaxes afterwards.
  * Two runs at the same seed produce identical output, which the unseeded
    CVaR sampler (certification/cvar_test.py:51) does not give you by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline.idle_policy import IdlePolicy  # noqa: E402
from baseline.improvement_calculator import ImprovementCalculator  # noqa: E402
from bridge import weights as weights_mod  # noqa: E402
from bridge.controller import MetaMoController  # noqa: E402
from bridge.governor import MetaMoGovernor  # noqa: E402
from bridge.protocol import SkillOutcome  # noqa: E402
from env.minecraft_stub import SKILL_NAMES, MinecraftStubEnv  # noqa: E402
from generator.mdn import MotiveDecompositionNetwork  # noqa: E402
from utils.mdn_contracts import CandidateSkillRecord  # noqa: E402
from utils.mdn_runtime_pipeline import (  # noqa: E402
    RuntimeCertificationPipeline,
    RuntimePipelineConfig,
)
from utils.weight_set_store import WeightSetStore  # noqa: E402

GAMMA = 0.99

# A survival-oriented starting goal vector, in MetaMo's goal order
# (core/config.py:1-3): Individuation, Transcendence, Help, Curiosity, Novelty,
# Self, Ethical, Social.
#
# MetaMo's own default (core/engine.py:47-58) is tuned for a chat assistant
# (Help=0.8, Ethical=0.9) and makes Reputation dominate from the first step,
# which is the wrong prior for an agent that has to survive the night.
MINECRAFT_INITIAL_GOALS = np.array(
    [0.70, 0.40, 0.30, 0.50, 0.40, 0.60, 0.50, 0.30], dtype=np.float64
)


def rollout_fixed_action(
    env: MinecraftStubEnv,
    action: int,
    *,
    seed: int,
    gamma: float = GAMMA,
) -> Tuple[float, np.ndarray]:
    """Run one episode always taking `action`.

    Uses the same discounting convention as `IdlePolicy.run_baseline_episodes`
    (baseline/idle_policy.py:35-56) so the results are directly comparable.
    """
    obs, _ = env.reset(seed=seed)
    discount = 1.0
    total_payoff = 0.0
    motives: Optional[np.ndarray] = None

    while True:
        obs, reward_vec, terminated, truncated, _ = env.step(action)
        reward_vec = np.asarray(reward_vec, dtype=np.float32)
        if motives is None:
            motives = np.zeros_like(reward_vec)
        total_payoff += discount * float(np.sum(reward_vec))
        motives += discount * reward_vec
        if terminated or truncated:
            break
        discount *= gamma

    return float(total_payoff), np.asarray(motives, dtype=np.float32)


def build_candidates(
    env: MinecraftStubEnv,
    baseline_stats: Dict[str, Any],
    *,
    seed: int,
) -> List[CandidateSkillRecord]:
    """Evaluate each non-idle action as a candidate skill."""
    calculator = ImprovementCalculator(baseline_stats)
    records: List[CandidateSkillRecord] = []

    for action in range(1, len(SKILL_NAMES)):
        payoff, motives = rollout_fixed_action(env, action, seed=seed)
        delta_r, delta_n = calculator.compute_improvements(
            skill_payoff=payoff,
            skill_motives=motives,
        )
        records.append(
            CandidateSkillRecord(
                skill_id=SKILL_NAMES[action],
                delta_r=float(delta_r),
                delta_n=tuple(float(v) for v in delta_n),
                is_certified=False,
                gate_type="PDS",
                metadata={"action": action},
            )
        )
    return records


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cvar-samples", type=int, default=1000)
    args = parser.parse_args(argv)

    env = MinecraftStubEnv(seed=args.seed)
    num_objectives = env.num_objectives

    print("=" * 78)
    print("MetaMo -> SubRep, 6-objective Minecraft stub")
    print("=" * 78)

    # 1. Baseline.
    idle = IdlePolicy(env=env, idle_action=0, gamma=GAMMA)
    baseline_stats = idle.run_baseline_episodes(num_episodes=5, seed=args.seed)
    print(f"\nBaseline payoff: {baseline_stats['baseline_payoff']:.4f}")
    print(f"Baseline motives: {np.round(baseline_stats['baseline_motives'], 3)}")

    # 2. Candidate skills.
    candidates = build_candidates(env, baseline_stats, seed=args.seed)
    print(f"\nCandidate skills ({len(candidates)}):")
    for record in candidates:
        print(
            f"  {record.skill_id:<20} delta_r={record.delta_r:+.3f}  "
            f"min(delta_n)={min(record.delta_n):+.3f}"
        )

    # 3. Certification pipeline. PDS carries the epsilon budget and use_cvar
    #    turns on the CVaR gate alongside it (OR semantics, see
    #    utils/mdn_runtime_pipeline.py:402-404).
    obs, _ = env.reset(seed=args.seed)
    model = MotiveDecompositionNetwork(
        input_dim=int(obs.shape[0]),
        num_objectives=num_objectives,
    )
    model.eval()
    pipeline = RuntimeCertificationPipeline(
        model=model,
        weight_store=WeightSetStore(num_objectives=num_objectives),
        config=RuntimePipelineConfig(
            gate_type="PDS",
            use_cvar=True,
            require_cds_or_cvar=True,
            cvar_samples=args.cvar_samples,
            train_support_after_certify=False,
        ),
    )

    # 4. Governor + controller.
    #
    # Scale the appraisal inputs to this environment's actual magnitudes. The
    # stimulus builder squashes through tanh, so leaving the scales at 1.0
    # while deltas run to |10| saturates risk on the first step and pins the
    # modulators at their bounds, flattening the coupling into a constant.
    payoff_scale = float(np.mean([abs(r.delta_r) for r in candidates])) or 1.0
    motive_scale = float(
        np.mean([np.mean(np.abs(r.delta_n)) for r in candidates])
    ) or 1.0
    print(f"\nAppraisal scales: payoff={payoff_scale:.3f}  motive={motive_scale:.3f}")

    governor = MetaMoGovernor(
        cvar_samples=args.cvar_samples,
        initial_goals=MINECRAFT_INITIAL_GOALS,
        payoff_scale=payoff_scale,
        motive_scale=motive_scale,
    )
    controller = MetaMoController(governor, pipeline, seed=args.seed)

    # Diagnostic: how many candidates the PDS gate alone would admit at a
    # given epsilon. Under OR semantics (use_cvar=True,
    # require_cds_or_cvar=True -> utils/mdn_runtime_pipeline.py:402-404) the
    # CVaR gate can admit what PDS rejects, so this is the only way to see
    # epsilon actually biting.
    from certification.pds_test import PDSGate  # noqa: E402

    def pds_only_admitted(epsilon: float) -> int:
        gate = PDSGate(epsilon=epsilon)
        return sum(
            1
            for r in candidates
            if gate.admit(r.delta_r, np.asarray(r.delta_n, dtype=np.float64))
        )

    # Per-step baseline, for turning a single realized reward vector into an
    # improvement. The episode-level baseline covers the whole episode.
    per_step_baseline = (
        np.asarray(baseline_stats["baseline_motives"], dtype=np.float64)
        / env.episode_length
    )

    # Mutable cell so the outcome closure can drive the environment and
    # publish what it observed back to the print loop.
    live: Dict[str, Any] = {"obs": None, "info": {"threat": 0.0}}

    def outcome_for(record: Optional[Any]) -> SkillOutcome:
        """Execute the selected skill and appraise what actually happened.

        Feeding back the REALIZED single-step reward (rather than the
        precomputed episode-level delta) is what makes the loop live: the stub
        env's threat cycle changes the payoff of the same action over time, so
        MetaMo's appraisal sees varying risk instead of a constant.
        """
        action = 0 if record is None else SKILL_NAMES.index(record.skill_id)
        obs, reward_vec, terminated, truncated, info = env.step(action)

        realized = np.asarray(reward_vec, dtype=np.float64)
        delta_n = realized - per_step_baseline
        delta_r = float(np.sum(delta_n))

        if terminated or truncated:
            obs, info = env.reset(seed=args.seed)

        live["obs"] = obs
        live["info"] = info

        return SkillOutcome(
            delta_r=delta_r,
            delta_n=delta_n,
            admitted=record is not None,
            effort=0.2 if record is None else 0.4,
        )

    # 5. The loop.
    print(f"\n{'-' * 78}")
    print("Per-step coupling (watch eps and alpha move together)")
    print("-" * 78)
    header = (
        f"{'step':>4} {'threat':>7} {'secur':>6} {'thresh':>7} "
        f"{'eps':>7} {'alpha':>7} {'adm':>5} {'pds':>4}  "
        f"{'selected':<20} {'Safety w':>9}"
    )
    print(header)
    print("-" * len(header))

    obs, info = env.reset(seed=args.seed)
    live["obs"] = obs
    live["info"] = info

    for _ in range(args.steps):
        threat_before = float(live["info"].get("threat", 0.0))
        record = controller.step(
            context=live["obs"],
            candidate_skills=candidates,
            baseline_stats=baseline_stats,
            outcome_for=outcome_for,
        )

        info = {"threat": threat_before}
        mods = record.modulators
        print(
            f"{record.step:>4} {info['threat']:>7.3f} "
            f"{mods.get('securing', float('nan')):>6.3f} "
            f"{mods.get('threshold', float('nan')):>7.3f} "
            f"{record.pds_epsilon:>7.4f} {record.cvar_tail_level:>7.4f} "
            f"{record.admitted_count:>2}/{record.candidate_count:<2} "
            f"{pds_only_admitted(record.pds_epsilon):>4} "
            f"{(record.selected_skill_id or '-'):<20} "
            f"{record.weights[0]:>9.3f}"
        )

    # 6. Summary.
    eps_values = [r.pds_epsilon for r in controller.history]
    alpha_values = [r.cvar_tail_level for r in controller.history]
    print("-" * len(header))
    print(
        f"eps   range: [{min(eps_values):.4f}, {max(eps_values):.4f}]   "
        f"alpha range: [{min(alpha_values):.4f}, {max(alpha_values):.4f}]"
    )
    print(f"final weights: {weights_mod.describe(controller.history[-1].weights)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

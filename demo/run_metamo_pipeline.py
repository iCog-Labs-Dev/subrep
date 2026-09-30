"""End-to-end MetaMo -> SubRep demo on the 6-objective Minecraft stub.

Shows the coupling the reference specification describes (§4.3): MetaMo emits
per-step selection
weights and risk budgets, SubRep certifies and selects under them, and the
executed outcome feeds back into MetaMo's appraisal.

Run:
    python demo/run_metamo_pipeline.py
    python demo/run_metamo_pipeline.py --steps 16 --seed 7

What to watch:
  * `eps` and `alpha` BOTH tighten as `securing` rises. That is the corrected
    sign convention -- under the specified formulas as written they would move in
    opposite directions. See bridge/budget.py for the full argument.
  * The Safety weight climbs as threat rises and relaxes afterwards.
  * Two runs at the same seed produce identical output, which the unseeded
    CVaR sampler (certification/cvar_test.py:51) does not give you by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bridge import weights as weights_mod  # noqa: E402
from bridge.controller import MetaMoController  # noqa: E402
from bridge.governor import MetaMoGovernor  # noqa: E402
from bridge.protocol import SkillOutcome  # noqa: E402
from env.minecraft_rollout import (  # noqa: E402
    DEFAULT_HORIZON,
    MINECRAFT_INITIAL_GOALS,
    appraisal_scales,
    evaluate_candidates,
    execute_option,
)
from env.minecraft_stub import SKILL_NAMES, MinecraftStubEnv  # noqa: E402
from generator.mdn import MotiveDecompositionNetwork  # noqa: E402
from utils.mdn_runtime_pipeline import (  # noqa: E402
    RuntimeCertificationPipeline,
    RuntimePipelineConfig,
)
from utils.weight_set_store import WeightSetStore  # noqa: E402

GAMMA = 0.99


def build_pipeline(
    model: MotiveDecompositionNetwork,
    *,
    num_objectives: int,
    cvar_samples: int,
) -> RuntimeCertificationPipeline:
    """A fresh certification pipeline around `model`.

    PDS carries the epsilon budget and use_cvar turns on the CVaR gate
    alongside it (OR semantics, see utils/mdn_runtime_pipeline.py:402-404).
    """
    return RuntimeCertificationPipeline(
        model=model,
        weight_store=WeightSetStore(num_objectives=num_objectives),
        config=RuntimePipelineConfig(
            gate_type="PDS",
            use_cvar=True,
            require_cds_or_cvar=True,
            cvar_samples=cvar_samples,
            train_support_after_certify=False,
        ),
    )


def start_new_episode(
    env: MinecraftStubEnv,
    controller: MetaMoController,
    *,
    seed: int,
    make_pipeline: Callable[[], RuntimeCertificationPipeline],
) -> np.ndarray:
    """Reset the env and give the controller a fresh pipeline.

    The pipeline caches certificates by (rounded context, skill_id) and, on a
    hit, returns the STORED delta and epsilon rather than the current ones
    (utils/mdn_runtime_pipeline.py:224-243). Every episode starts from the same
    observation, so without a fresh pipeline each new episode would open on the
    previous episode's certificates. Within an episode contexts never repeat
    (progress t/T changes), so one pipeline per episode is enough.

    The governor is kept: MetaMo's motivational state carries across episodes.
    """
    obs, _ = env.reset(seed=seed)
    controller.pipeline = make_pipeline()
    return obs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cvar-samples", type=int, default=1000)
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="option duration in env steps, for evaluation AND execution",
    )
    args = parser.parse_args(argv)

    env = MinecraftStubEnv(seed=args.seed)
    num_objectives = env.num_objectives

    print("=" * 78)
    print("MetaMo -> SubRep, 6-objective Minecraft stub")
    print("=" * 78)

    # 1-2. Baseline and candidate skills at the start state, for display and
    #      for sizing the appraisal scales. The loop re-evaluates both at every
    #      decision, from whatever state the episode has reached.
    env.reset(seed=args.seed)
    baseline_stats, candidates = evaluate_candidates(
        env, gamma=GAMMA, horizon=args.horizon
    )
    print(f"\nBaseline payoff: {baseline_stats['baseline_payoff']:.4f}")
    print(f"Baseline motives: {np.round(baseline_stats['baseline_motives'], 3)}")

    print(f"\nCandidate skills ({len(candidates)}):")
    for record in candidates:
        print(
            f"  {record.skill_id:<20} delta_r={record.delta_r:+.3f}  "
            f"min(delta_n)={min(record.delta_n):+.3f}"
        )

    # 3. Certification pipeline (rebuilt at every episode start, see
    #    start_new_episode). Seed before building the untrained MDN: the
    #    controller seeds torch before each certification, but the model's
    #    weights are drawn here, and the CVaR gate samples from them.
    import torch

    torch.manual_seed(args.seed)
    model = MotiveDecompositionNetwork(
        input_dim=int(env.observation().shape[0]),
        num_objectives=num_objectives,
    )
    model.eval()

    def make_pipeline() -> RuntimeCertificationPipeline:
        return build_pipeline(
            model,
            num_objectives=num_objectives,
            cvar_samples=args.cvar_samples,
        )

    # 4. Governor + controller, with appraisal inputs scaled to this
    #    environment's actual magnitudes (see appraisal_scales). The governor
    #    takes its scales once, at construction (bridge/governor.py:135-136),
    #    so they come from the start-state candidates and stay fixed.
    payoff_scale, motive_scale = appraisal_scales(candidates)
    print(f"\nAppraisal scales: payoff={payoff_scale:.3f}  motive={motive_scale:.3f}")

    governor = MetaMoGovernor(
        cvar_samples=args.cvar_samples,
        initial_goals=MINECRAFT_INITIAL_GOALS,
        payoff_scale=payoff_scale,
        motive_scale=motive_scale,
    )
    controller = MetaMoController(governor, make_pipeline(), seed=args.seed)

    # Diagnostic: how many candidates the PDS gate alone would admit at a
    # given epsilon. Under OR semantics (use_cvar=True,
    # require_cds_or_cvar=True -> utils/mdn_runtime_pipeline.py:402-404) the
    # CVaR gate can admit what PDS rejects, so this is the only way to see
    # epsilon actually biting.
    from certification.pds_test import PDSGate  # noqa: E402

    def pds_only_admitted(epsilon: float, records: List[Any]) -> int:
        gate = PDSGate(epsilon=epsilon)
        return sum(
            1
            for r in records
            if gate.admit(r.delta_r, np.asarray(r.delta_n, dtype=np.float64))
        )

    def outcome_for(record: Optional[Any]) -> SkillOutcome:
        """Execute the selected option for `horizon` steps and appraise it.

        The realized improvement is measured against idling from the same
        state over the same horizon (execute_option), so MetaMo's appraisal
        sees what this option actually did here, under the current threat.
        """
        action = 0 if record is None else SKILL_NAMES.index(record.skill_id)
        delta_r, delta_n = execute_option(
            env, action, horizon=args.horizon, gamma=GAMMA
        )
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

    env.reset(seed=args.seed)

    for _ in range(args.steps):
        threat_before = env.threat
        # Per-state evaluation: delta(x, o) is only meaningful against the
        # baseline at the SAME state x. Rollouts run on copies, so the live
        # episode is untouched (env/minecraft_rollout.py:evaluate_option).
        baseline_stats, candidates = evaluate_candidates(
            env, gamma=GAMMA, horizon=args.horizon
        )
        record = controller.step(
            context=env.observation(),
            candidate_skills=candidates,
            baseline_stats=baseline_stats,
            outcome_for=outcome_for,
        )

        mods = record.modulators
        print(
            f"{record.step:>4} {threat_before:>7.3f} "
            f"{mods.get('securing', float('nan')):>6.3f} "
            f"{mods.get('threshold', float('nan')):>7.3f} "
            f"{record.pds_epsilon:>7.4f} {record.cvar_tail_level:>7.4f} "
            f"{record.admitted_count:>2}/{record.candidate_count:<2} "
            f"{pds_only_admitted(record.pds_epsilon, candidates):>4} "
            f"{(record.selected_skill_id or '-'):<20} "
            f"{record.weights[0]:>9.3f}"
        )

        if env.episode_over:
            start_new_episode(
                env, controller, seed=args.seed, make_pipeline=make_pipeline
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

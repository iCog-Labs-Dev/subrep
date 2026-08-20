"""Train a Safety-Gymnasium MDN from same-context rollout files.

The standard MDN trainer expects candidate-set files with one context and many
candidate outcomes. Safety-Gymnasium rollout files already have that shape, but
their correct baseline is the same-context ``zero_action`` candidate rather
than MO-LunarLander's idle baseline. This entrypoint converts every non-baseline
candidate into baseline-relative payoff/motive deltas, then trains an MDN using
zero baseline stats.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_trainer import MDNAuxiliaryTrainer, MDNAuxiliaryTrainerConfig
from generator.mdn_trainer import MDNTrainer, MDNTrainerConfig
from generator.train_mdn import (
    build_auxiliary_records_from_prepared_candidate_outcomes,
    build_records_from_prepared_candidate_outcomes,
)
from generator.train_mdn_candidate_sets import (
    _attach_auxiliary_q_calibration,
    _attach_auxiliary_target_normalization,
    _restore_model_state,
    compute_auxiliary_q_calibration,
    compute_auxiliary_target_normalization,
    normalize_auxiliary_targets_in_records,
)
from utils.mdn_record_builder import PreparedCandidateOutcome


def safety_rollout_directory_to_delta_outcomes(
    directory: str | Path,
    *,
    pattern: str = "*.npz",
    baseline_candidate_id: str = "zero_action",
    gate_type: str = "PDS",
    epsilon: float = 1.0,
) -> tuple[PreparedCandidateOutcome, ...]:
    """Load Safety-Gymnasium rollout files as baseline-relative MDN outcomes."""
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No rollout files matching {pattern!r} found in {directory!s}")

    outcomes: list[PreparedCandidateOutcome] = []
    for path in files:
        data = np.load(path, allow_pickle=True)
        context = np.asarray(data["context"], dtype=np.float32).reshape(-1)
        skill_ids = [str(item) for item in np.asarray(data["candidate_skill_ids"]).reshape(-1)]
        payoffs = np.asarray(data["candidate_payoffs"], dtype=np.float32).reshape(-1)
        motives = np.asarray(data["candidate_motives"], dtype=np.float32)

        if baseline_candidate_id not in skill_ids:
            raise ValueError(f"{path} does not contain baseline candidate {baseline_candidate_id!r}")
        baseline_idx = skill_ids.index(baseline_candidate_id)
        baseline_payoff = float(payoffs[baseline_idx])
        baseline_motives = motives[baseline_idx]

        for index, skill_id in enumerate(skill_ids):
            if skill_id == baseline_candidate_id:
                continue
            delta_payoff = float(payoffs[index]) - baseline_payoff
            delta_motives = motives[index] - baseline_motives
            outcomes.append(
                PreparedCandidateOutcome(
                    context=tuple(float(v) for v in context),
                    skill_id=skill_id,
                    payoff=delta_payoff,
                    motives=tuple(float(v) for v in delta_motives),
                    metadata={
                        "candidate_set_path": str(path),
                        "baseline_candidate_id": baseline_candidate_id,
                    },
                    gate_type=gate_type,
                    epsilon=float(epsilon),
                )
            )
    return tuple(outcomes)


def train_safety_gymnasium_mdn(
    *,
    data_dir: str | Path,
    pattern: str = "*.npz",
    baseline_candidate_id: str = "zero_action",
    seed: int = 42,
    device: str | None = None,
    policy_checkpoint_path: str = "models/safety_mdn_policy_3d_best.pth",
    auxiliary_checkpoint_path: str = "models/safety_mdn_auxiliary_3d_best.pth",
    skill_id_bucket_count: int = 100_000,
    normalize_auxiliary_targets: bool = True,
    q_loss: str = "mse",
    gate_type: str = "PDS",
    epsilon: float = 1.0,
) -> dict[str, object]:
    outcomes = safety_rollout_directory_to_delta_outcomes(
        data_dir,
        pattern=pattern,
        baseline_candidate_id=baseline_candidate_id,
        gate_type=gate_type,
        epsilon=epsilon,
    )
    baseline_stats = {
        "baseline_payoff": 0.0,
        "baseline_motives": np.zeros(len(outcomes[0].motives), dtype=np.float32),
    }

    decision_records = build_records_from_prepared_candidate_outcomes(
        prepared_outcomes=outcomes,
        baseline_stats=baseline_stats,
        seed=seed,
        device=device,
    )
    raw_auxiliary_records = build_auxiliary_records_from_prepared_candidate_outcomes(
        prepared_outcomes=outcomes,
        baseline_stats=baseline_stats,
        gamma=1.0,
        skill_id_bucket_count=skill_id_bucket_count,
    )

    auxiliary_records = raw_auxiliary_records
    target_normalization = None
    if normalize_auxiliary_targets:
        target_normalization = compute_auxiliary_target_normalization(auxiliary_records)
        auxiliary_records = normalize_auxiliary_targets_in_records(
            auxiliary_records,
            target_normalization,
        )

    model = MotiveDecompositionNetwork(
        input_dim=len(outcomes[0].context),
        num_objectives=len(outcomes[0].motives),
        num_skills=skill_id_bucket_count,
    )

    policy_trainer = MDNTrainer(
        model,
        config=MDNTrainerConfig(random_seed=seed, checkpoint_path=policy_checkpoint_path),
        device=device,
    )
    policy_metrics = policy_trainer.train_records(decision_records)

    auxiliary_trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(
            checkpoint_path=auxiliary_checkpoint_path,
            random_seed=seed,
            q_loss=q_loss,
        ),
        device=device,
    )
    auxiliary_metrics = auxiliary_trainer.train_records(auxiliary_records)

    _restore_model_state(model, auxiliary_checkpoint_path, device=device)
    q_calibration = compute_auxiliary_q_calibration(
        model,
        raw_auxiliary_records,
        target_normalization=target_normalization,
        device=device,
    )
    if target_normalization is not None:
        _attach_auxiliary_target_normalization(auxiliary_checkpoint_path, target_normalization)
    _attach_auxiliary_q_calibration(auxiliary_checkpoint_path, q_calibration)

    policy_checkpoint = policy_trainer.save_checkpoint(policy_checkpoint_path)
    if target_normalization is not None:
        _attach_auxiliary_target_normalization(policy_checkpoint, target_normalization)
    _attach_auxiliary_q_calibration(policy_checkpoint, q_calibration)

    return {
        "candidate_outcomes": len(outcomes),
        "contexts": len({outcome.context for outcome in outcomes}),
        "context_dim": len(outcomes[0].context),
        "num_objectives": len(outcomes[0].motives),
        "policy": {**policy_metrics, "checkpoint_path": policy_checkpoint},
        "auxiliary": auxiliary_metrics,
        "auxiliary_target_normalization": target_normalization,
        "auxiliary_q_calibration": q_calibration,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Safety-Gymnasium MDN from rollout files.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--baseline-candidate", type=str, default="zero_action")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--policy-checkpoint", type=str, default="models/safety_mdn_policy_3d_best.pth")
    parser.add_argument("--auxiliary-checkpoint", type=str, default="models/safety_mdn_auxiliary_3d_best.pth")
    parser.add_argument("--q-loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--gate-type", choices=("CDS", "PDS"), default="PDS")
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--no-normalize-auxiliary-targets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_safety_gymnasium_mdn(
        data_dir=args.data_dir,
        pattern=args.pattern,
        baseline_candidate_id=args.baseline_candidate,
        seed=args.seed,
        device=args.device,
        policy_checkpoint_path=args.policy_checkpoint,
        auxiliary_checkpoint_path=args.auxiliary_checkpoint,
        normalize_auxiliary_targets=not args.no_normalize_auxiliary_targets,
        q_loss=args.q_loss,
        gate_type=args.gate_type,
        epsilon=args.epsilon,
    )
    print("Safety-Gymnasium MDN Training Complete")
    print("=====================================")
    print(f"contexts: {result['contexts']}")
    print(f"candidate outcomes: {result['candidate_outcomes']}")
    print(f"context dim: {result['context_dim']}")
    print(f"objectives: {result['num_objectives']}")
    print(f"policy checkpoint: {result['policy']['checkpoint_path']}")
    print(f"auxiliary checkpoint: {args.auxiliary_checkpoint}")


if __name__ == "__main__":
    main()

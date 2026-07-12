"""Train MDN checkpoints from probability-aware runtime decision logs."""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_trainer import AuxiliaryTrainingRecord, MDNAuxiliaryTrainer, MDNAuxiliaryTrainerConfig
from generator.mdn_trainer import MDNTrainer, MDNTrainerConfig
from generator.train_mdn_candidate_sets import (
    _attach_auxiliary_target_normalization,
    _restore_model_state,
    compute_auxiliary_target_normalization,
    normalize_auxiliary_targets_in_records,
)
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord
from utils.mdn_checkpoint_loader import load_mdn_checkpoint
from utils.probability_aware_logs import load_probability_aware_log, probability_aware_log_files


def train_mdn_from_probability_aware_logs(
    *,
    data_dir: str | Path = "data/mdn_probability_aware_logs",
    pattern: str = "*.npz",
    seed: int = 42,
    device: str | None = None,
    policy_checkpoint_path: str | None = None,
    auxiliary_checkpoint_path: str | None = None,
    use_ips: bool = False,
    use_doubly_robust: bool = False,
    skill_id_bucket_count: int = 100_000,
    normalize_auxiliary_targets: bool = True,
    q_loss: str = "mse",
    huber_delta: float = 1.0,
    max_logs: int | None = None,
    dr_baseline_checkpoint_path: str | None = None,
) -> dict[str, Any]:
    """Train policy and auxiliary heads from real probability-aware logs."""
    if use_ips and use_doubly_robust:
        raise ValueError("use_ips and use_doubly_robust cannot both be enabled")
    estimator = estimator_name(use_ips=use_ips, use_doubly_robust=use_doubly_robust)
    policy_checkpoint_path, auxiliary_checkpoint_path = resolve_checkpoint_paths(
        policy_checkpoint_path=policy_checkpoint_path,
        auxiliary_checkpoint_path=auxiliary_checkpoint_path,
        estimator=estimator,
    )
    if use_doubly_robust and dr_baseline_checkpoint_path is None:
        raise ValueError(
            "DR training requires --dr-baseline-checkpoint with a frozen unweighted auxiliary checkpoint"
        )
    dr_baseline_model = None
    if dr_baseline_checkpoint_path is not None:
        dr_baseline_model = load_mdn_checkpoint(dr_baseline_checkpoint_path, map_location=device or "cpu")
    log_files = probability_aware_log_files(data_dir, pattern=pattern)
    if max_logs is not None:
        if max_logs <= 0:
            raise ValueError("max_logs must be positive when provided")
        log_files = log_files[: int(max_logs)]
    print(
        f"[LoggedTrain] estimator={estimator} logs={len(log_files)} data_dir={data_dir} pattern={pattern!r} "
        f"policy_checkpoint={policy_checkpoint_path} auxiliary_checkpoint={auxiliary_checkpoint_path} "
        f"dr_baseline_checkpoint={dr_baseline_checkpoint_path}",
        flush=True,
    )
    logs = [load_probability_aware_log(path) for path in log_files]
    decision_records, raw_auxiliary_records = probability_aware_logs_to_training_records(
        logs,
        skill_id_bucket_count=skill_id_bucket_count,
    )
    auxiliary_records = raw_auxiliary_records
    target_normalization: dict[str, object] | None = None
    if normalize_auxiliary_targets:
        target_normalization = compute_auxiliary_target_normalization(auxiliary_records)
        auxiliary_records = normalize_auxiliary_targets_in_records(auxiliary_records, target_normalization)

    first_record = decision_records[0]
    model = MotiveDecompositionNetwork(
        input_dim=len(first_record.context),
        num_objectives=len(first_record.alpha),
        num_skills=skill_id_bucket_count,
    )

    policy_trainer = MDNTrainer(
        model,
        config=MDNTrainerConfig(
            random_seed=seed,
            checkpoint_path=policy_checkpoint_path,
        ),
        device=device,
    )
    print(f"[LoggedTrain] training policy on {len(decision_records)} decision records", flush=True)
    policy_metrics = policy_trainer.train_records(decision_records)
    print(
        f"[LoggedTrain] policy complete loss={policy_metrics['loss']:.6f} "
        f"utility={policy_metrics['utility']:.6f}",
        flush=True,
    )

    auxiliary_trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(
            checkpoint_path=auxiliary_checkpoint_path,
            random_seed=seed,
            use_ips=use_ips,
            use_doubly_robust=use_doubly_robust,
            q_loss=q_loss,
            huber_delta=huber_delta,
        ),
        device=device,
        dr_baseline_model=dr_baseline_model,
    )
    print(
        f"[LoggedTrain] training auxiliary on {len(auxiliary_records)} records "
        f"mode={estimator}",
        flush=True,
    )
    if use_ips or use_doubly_robust:
        auxiliary_metrics = auxiliary_trainer.train_probability_aware_records(auxiliary_records)
    else:
        auxiliary_metrics = auxiliary_trainer.train_records(auxiliary_records)
    print(
        f"[LoggedTrain] auxiliary complete best_val_loss={auxiliary_metrics['best_val_loss']:.6f}",
        flush=True,
    )

    _restore_model_state(model, auxiliary_checkpoint_path, device=device)
    if target_normalization is not None:
        _attach_auxiliary_target_normalization(auxiliary_checkpoint_path, target_normalization)

    policy_checkpoint = policy_trainer.save_checkpoint(policy_checkpoint_path)
    if target_normalization is not None:
        _attach_auxiliary_target_normalization(policy_checkpoint, target_normalization)

    return {
        "logged_decisions": len(logs),
        "decision_records": len(decision_records),
        "auxiliary_records": len(auxiliary_records),
        "estimator": estimator,
        "policy": {**policy_metrics, "checkpoint_path": policy_checkpoint},
        "auxiliary": auxiliary_metrics,
        "auxiliary_target_normalization": target_normalization,
        "dr_baseline_checkpoint_path": dr_baseline_checkpoint_path,
    }


def estimator_name(*, use_ips: bool, use_doubly_robust: bool) -> str:
    if use_ips and use_doubly_robust:
        raise ValueError("use_ips and use_doubly_robust cannot both be enabled")
    if use_doubly_robust:
        return "dr"
    if use_ips:
        return "ips"
    return "unweighted"


def resolve_checkpoint_paths(
    *,
    policy_checkpoint_path: str | None,
    auxiliary_checkpoint_path: str | None,
    estimator: str,
) -> tuple[str, str]:
    """Use distinct default checkpoint names for each logged-data estimator."""
    estimator = estimator.strip().lower()
    if estimator not in {"unweighted", "ips", "dr"}:
        raise ValueError(f"unsupported estimator {estimator!r}")
    return (
        policy_checkpoint_path or f"models/mdn_policy_{estimator}.pth",
        auxiliary_checkpoint_path or f"models/mdn_auxiliary_{estimator}.pth",
    )


def probability_aware_logs_to_training_records(
    logs: list[dict[str, Any]],
    *,
    skill_id_bucket_count: int = 100_000,
) -> tuple[list[MDNDecisionRecord], list[AuxiliaryTrainingRecord]]:
    """Convert validated runtime logs into policy and auxiliary records."""
    if not logs:
        raise ValueError("logs must contain at least one probability-aware record")
    decision_records: list[MDNDecisionRecord] = []
    auxiliary_records: list[AuxiliaryTrainingRecord] = []
    for log in logs:
        candidate_records = _candidate_records_from_log(log)
        selected_index = int(np.asarray(log["selected_candidate_index"]).reshape(()).item())
        certified_indices = [index for index, candidate in enumerate(candidate_records) if candidate.is_certified]
        if selected_index not in certified_indices:
            raise ValueError("selected candidate must be certified")
        selected_certified_index = certified_indices.index(selected_index)
        selected_candidate = candidate_records[selected_index]
        certified_delta_r = tuple(float(candidate_records[index].delta_r) for index in certified_indices)
        certified_delta_n = tuple(tuple(float(v) for v in candidate_records[index].delta_n) for index in certified_indices)
        actual_motives = tuple(float(v) for v in np.asarray(log["actual_motives"], dtype=np.float32).reshape(-1))
        behavior_probability = float(np.asarray(log["behavior_probability"]).reshape(()).item())

        decision_records.append(
            MDNDecisionRecord(
                context=tuple(float(v) for v in np.asarray(log["context"], dtype=np.float32).reshape(-1)),
                alpha=tuple(float(v) for v in np.asarray(log["alpha"], dtype=np.float32).reshape(-1)),
                support_values=tuple(float(v) for v in np.asarray(log["support_values"], dtype=np.float32).reshape(-1)),
                weights_used=tuple(float(v) for v in np.asarray(log["weights_used"], dtype=np.float32).reshape(-1)),
                candidate_skills=candidate_records,
                selected_skill_id=selected_candidate.skill_id,
                selected_score=float(np.asarray(log["selected_score"]).reshape(()).item()),
                behavior_probability=behavior_probability,
                actual_payoff=float(np.asarray(log["actual_payoff"]).reshape(()).item()),
                actual_motives=actual_motives,
            )
        )

        selected_record = AuxiliaryTrainingRecord(
            context=decision_records[-1].context,
            skill_id=_stable_skill_id(selected_candidate.skill_id, bucket_count=skill_id_bucket_count),
            accept_label=1.0,
            q_target=actual_motives,
            has_q_target=True,
            behavior_probability=behavior_probability,
            candidate_delta_r=certified_delta_r,
            candidate_delta_n=certified_delta_n,
            selected_candidate_index=selected_certified_index,
        )
        auxiliary_records.append(selected_record)

        for index, candidate in enumerate(candidate_records):
            if index == selected_index:
                continue
            auxiliary_records.append(
                AuxiliaryTrainingRecord(
                    context=decision_records[-1].context,
                    skill_id=_stable_skill_id(candidate.skill_id, bucket_count=skill_id_bucket_count),
                    accept_label=float(candidate.is_certified),
                    q_target=tuple(0.0 for _ in actual_motives),
                    has_q_target=False,
                    behavior_probability=behavior_probability,
                    candidate_delta_r=certified_delta_r,
                    candidate_delta_n=certified_delta_n,
                    selected_candidate_index=selected_certified_index,
                )
            )
    return decision_records, auxiliary_records


def _candidate_records_from_log(log: dict[str, Any]) -> tuple[CandidateSkillRecord, ...]:
    skill_ids = np.asarray(log["candidate_skill_ids"]).reshape(-1)
    delta_r = np.asarray(log["candidate_delta_r"], dtype=np.float32).reshape(-1)
    delta_n = np.asarray(log["candidate_delta_n"], dtype=np.float32)
    labels = np.asarray(log["candidate_accept_labels"]).reshape(-1).astype(bool)
    gate_types = np.asarray(log.get("candidate_gate_types", np.asarray(["CDS"] * len(skill_ids)))).reshape(-1)
    margins = np.asarray(
        log.get("candidate_admission_margins", np.full(len(skill_ids), np.nan, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    records: list[CandidateSkillRecord] = []
    for index, skill_id in enumerate(skill_ids):
        margin = float(margins[index])
        records.append(
            CandidateSkillRecord(
                skill_id=str(skill_id),
                delta_r=float(delta_r[index]),
                delta_n=tuple(float(v) for v in delta_n[index]),
                is_certified=bool(labels[index]),
                gate_type=str(gate_types[index]),
                admission_margin=None if np.isnan(margin) else margin,
            )
        )
    return tuple(records)


def _stable_skill_id(skill_id: str, *, bucket_count: int) -> int:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    suffix = skill_id.split("_")[-1]
    if suffix.isdigit():
        return int(suffix) % int(bucket_count)
    return zlib.crc32(skill_id.encode()) % int(bucket_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MDN from probability-aware runtime logs.")
    parser.add_argument("--data-dir", type=str, default="data/mdn_probability_aware_logs")
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--policy-checkpoint", type=str, default=None)
    parser.add_argument("--auxiliary-checkpoint", type=str, default=None)
    parser.add_argument("--use-ips", action="store_true")
    parser.add_argument("--use-doubly-robust", action="store_true")
    parser.add_argument("--skill-id-bucket-count", type=int, default=100_000)
    parser.add_argument("--no-normalize-auxiliary-targets", action="store_true")
    parser.add_argument("--q-loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--max-logs", type=int, default=None, help="Use only the first N sorted runtime logs")
    parser.add_argument(
        "--dr-baseline-checkpoint",
        type=str,
        default=None,
        help="Frozen unweighted auxiliary checkpoint used as the DR Q baseline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_mdn_from_probability_aware_logs(
        data_dir=args.data_dir,
        pattern=args.pattern,
        seed=args.seed,
        device=args.device,
        policy_checkpoint_path=args.policy_checkpoint,
        auxiliary_checkpoint_path=args.auxiliary_checkpoint,
        use_ips=args.use_ips,
        use_doubly_robust=args.use_doubly_robust,
        skill_id_bucket_count=args.skill_id_bucket_count,
        normalize_auxiliary_targets=not args.no_normalize_auxiliary_targets,
        q_loss=args.q_loss,
        huber_delta=args.huber_delta,
        max_logs=args.max_logs,
        dr_baseline_checkpoint_path=args.dr_baseline_checkpoint,
    )
    print(result)


if __name__ == "__main__":
    main()

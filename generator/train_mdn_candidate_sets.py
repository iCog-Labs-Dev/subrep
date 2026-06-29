"""Train MDN policy and auxiliary heads from candidate-set data files."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baseline.idle_policy import IdlePolicy
from env.lunar_lander_wrapper import SubRepEnv
from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_trainer import AuxiliaryTrainingRecord, MDNAuxiliaryTrainer, MDNAuxiliaryTrainerConfig
from generator.mdn_trainer import MDNTrainer, MDNTrainerConfig
from generator.train_mdn import (
    build_auxiliary_records_from_prepared_candidate_outcomes,
    build_records_from_prepared_candidate_outcomes,
)
from utils.mdn_data_adapter import candidate_set_directory_to_prepared_candidate_outcomes


def compute_idle_baseline_stats(
    *,
    episodes: int = 20,
    seed: int = 42,
    gamma: float = 0.99,
) -> dict[str, Any]:
    env = SubRepEnv(seed=seed)
    return IdlePolicy(env=env, gamma=gamma).run_baseline_episodes(
        num_episodes=episodes,
        seed=seed,
    )


def train_mdn_from_candidate_set_directory(
    *,
    data_dir: str | Path = "data/mdn_candidate_sets",
    pattern: str = "*.npz",
    baseline_stats: dict[str, Any] | None = None,
    baseline_episodes: int = 20,
    seed: int = 42,
    gamma: float = 0.99,
    device: str | None = None,
    policy_checkpoint_path: str = "models/mdn_policy_best.pth",
    auxiliary_checkpoint_path: str = "models/mdn_auxiliary_best.pth",
    use_ips: bool = False,
    use_doubly_robust: bool = False,
    skill_id_bucket_count: int = 100_000,
    normalize_auxiliary_targets: bool = True,
    q_loss: str = "mse",
    huber_delta: float = 1.0,
    calibrate_auxiliary_q: bool = False,
) -> dict[str, object]:
    """Load candidate-set files and train one shared MDN through policy + auxiliary phases."""
    outcomes = candidate_set_directory_to_prepared_candidate_outcomes(
        data_dir,
        pattern=pattern,
    )
    if baseline_stats is None:
        baseline_stats = compute_idle_baseline_stats(
            episodes=baseline_episodes,
            seed=seed,
            gamma=gamma,
        )

    decision_records = build_records_from_prepared_candidate_outcomes(
        prepared_outcomes=outcomes,
        baseline_stats=baseline_stats,
        seed=seed,
        device=device,
    )
    raw_auxiliary_records = build_auxiliary_records_from_prepared_candidate_outcomes(
        prepared_outcomes=outcomes,
        baseline_stats=baseline_stats,
        gamma=gamma,
        skill_id_bucket_count=skill_id_bucket_count,
    )
    auxiliary_records = raw_auxiliary_records
    target_normalization: dict[str, object] | None = None
    if normalize_auxiliary_targets:
        target_normalization = compute_auxiliary_target_normalization(auxiliary_records)
        auxiliary_records = normalize_auxiliary_targets_in_records(
            auxiliary_records,
            target_normalization,
        )

    first_context_dim = len(outcomes[0].context)
    num_objectives = len(outcomes[0].motives)
    model = MotiveDecompositionNetwork(
        input_dim=first_context_dim,
        num_objectives=num_objectives,
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
    policy_metrics = policy_trainer.train_records(decision_records)

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
    )
    if use_ips or use_doubly_robust:
        auxiliary_metrics = auxiliary_trainer.train_probability_aware_records(auxiliary_records)
    else:
        auxiliary_metrics = auxiliary_trainer.train_records(auxiliary_records)

    _restore_model_state(model, auxiliary_checkpoint_path, device=device)
    q_calibration = None
    if calibrate_auxiliary_q:
        q_calibration = compute_auxiliary_q_calibration(
            model,
            raw_auxiliary_records,
            target_normalization=target_normalization,
            device=device,
        )
    if target_normalization is not None:
        _attach_auxiliary_target_normalization(
            auxiliary_checkpoint_path,
            target_normalization,
        )
    if q_calibration is not None:
        _attach_auxiliary_q_calibration(auxiliary_checkpoint_path, q_calibration)

    # Save the shared model after both phases so either checkpoint can restore
    # the same trained alpha, support, gate, and Q heads.
    policy_checkpoint = policy_trainer.save_checkpoint(policy_checkpoint_path)
    if target_normalization is not None:
        _attach_auxiliary_target_normalization(
            policy_checkpoint,
            target_normalization,
        )
    if q_calibration is not None:
        _attach_auxiliary_q_calibration(policy_checkpoint, q_calibration)

    return {
        "candidate_outcomes": len(outcomes),
        "baseline_payoff": float(baseline_stats["baseline_payoff"]),
        "policy": {**policy_metrics, "checkpoint_path": policy_checkpoint},
        "auxiliary": auxiliary_metrics,
        "auxiliary_target_normalization": target_normalization,
        "auxiliary_q_calibration": q_calibration,
    }


def compute_auxiliary_target_normalization(
    records: list[AuxiliaryTrainingRecord],
    *,
    min_std: float = 1e-6,
) -> dict[str, object]:
    """Compute per-objective normalization stats for auxiliary Q targets."""
    if not records:
        raise ValueError("records must contain at least one auxiliary training record")
    targets = np.asarray([record.q_target for record in records], dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] == 0:
        raise ValueError(f"q_target array must have shape (N, M), got {targets.shape}")
    if not np.all(np.isfinite(targets)):
        raise ValueError("q_target values must be finite")

    mean = targets.mean(axis=0)
    std = targets.std(axis=0)
    std = np.where(std < float(min_std), 1.0, std)
    return {
        "enabled": True,
        "mean": tuple(float(value) for value in mean),
        "std": tuple(float(value) for value in std),
        "count": int(targets.shape[0]),
    }


def normalize_auxiliary_targets_in_records(
    records: list[AuxiliaryTrainingRecord],
    target_normalization: dict[str, object],
) -> list[AuxiliaryTrainingRecord]:
    mean = np.asarray(target_normalization["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(target_normalization["std"], dtype=np.float32).reshape(-1)
    normalized_records: list[AuxiliaryTrainingRecord] = []
    for record in records:
        target = np.asarray(record.q_target, dtype=np.float32).reshape(-1)
        if target.shape != mean.shape:
            raise ValueError(
                f"q_target shape {target.shape} does not match normalization shape {mean.shape}"
            )
        normalized = (target - mean) / std
        normalized_records.append(
            replace(record, q_target=tuple(float(value) for value in normalized))
        )
    return normalized_records


def compute_auxiliary_q_calibration(
    model: MotiveDecompositionNetwork,
    records: list[AuxiliaryTrainingRecord],
    *,
    target_normalization: dict[str, object] | None,
    device: str | None,
    batch_size: int = 512,
    min_variance: float = 1e-8,
) -> dict[str, object]:
    """Fit per-objective affine calibration from Q predictions to raw targets."""
    if not records:
        raise ValueError("records must contain at least one auxiliary training record")

    target = np.asarray([record.q_target for record in records], dtype=np.float32)
    context = np.asarray([record.context for record in records], dtype=np.float32)
    skill_id = np.asarray([record.skill_id for record in records], dtype=np.int64)
    device_obj = torch.device(device or next(model.parameters()).device)
    predictions: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            end = start + batch_size
            context_tensor = torch.tensor(context[start:end], dtype=torch.float32, device=device_obj)
            skill_tensor = torch.tensor(skill_id[start:end], dtype=torch.long, device=device_obj)
            _, q_hat = model.forward_auxiliary(context_tensor, skill_tensor)
            batch_prediction = q_hat.detach().cpu().numpy().astype(np.float32)
            predictions.append(batch_prediction)

    prediction = np.concatenate(predictions, axis=0)
    if target_normalization is not None and target_normalization.get("enabled", True):
        mean = np.asarray(target_normalization["mean"], dtype=np.float32).reshape(1, -1)
        std = np.asarray(target_normalization["std"], dtype=np.float32).reshape(1, -1)
        prediction = prediction * std + mean

    if prediction.shape != target.shape:
        raise ValueError(f"prediction shape {prediction.shape} does not match target shape {target.shape}")

    slope: list[float] = []
    intercept: list[float] = []
    for objective_index in range(target.shape[1]):
        x = prediction[:, objective_index].astype(np.float64)
        y = target[:, objective_index].astype(np.float64)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_variance = float(np.mean((x - x_mean) ** 2))
        if x_variance < min_variance:
            objective_slope = 1.0
            objective_intercept = y_mean - x_mean
        else:
            covariance = float(np.mean((x - x_mean) * (y - y_mean)))
            objective_slope = covariance / x_variance
            objective_intercept = y_mean - objective_slope * x_mean
        slope.append(float(objective_slope))
        intercept.append(float(objective_intercept))

    slope_array = np.asarray(slope, dtype=np.float64).reshape(1, -1)
    intercept_array = np.asarray(intercept, dtype=np.float64).reshape(1, -1)
    calibrated_prediction = prediction.astype(np.float64) * slope_array + intercept_array
    before_error = prediction.astype(np.float64) - target.astype(np.float64)
    after_error = calibrated_prediction - target.astype(np.float64)

    return {
        "enabled": True,
        "type": "affine",
        "slope": tuple(slope),
        "intercept": tuple(intercept),
        "count": int(target.shape[0]),
        "train_mse_before": float(np.mean(before_error ** 2)),
        "train_mse_after": float(np.mean(after_error ** 2)),
        "train_mae_before": float(np.mean(np.abs(before_error))),
        "train_mae_after": float(np.mean(np.abs(after_error))),
    }


def _attach_auxiliary_target_normalization(
    checkpoint_path: str | Path,
    target_normalization: dict[str, object],
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        checkpoint = {"model_state_dict": checkpoint}
    checkpoint["auxiliary_target_normalization"] = target_normalization
    torch.save(checkpoint, checkpoint_path)


def _attach_auxiliary_q_calibration(
    checkpoint_path: str | Path,
    q_calibration: dict[str, object],
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        checkpoint = {"model_state_dict": checkpoint}
    checkpoint["auxiliary_q_calibration"] = q_calibration
    torch.save(checkpoint, checkpoint_path)


def _restore_model_state(
    model: MotiveDecompositionNetwork,
    checkpoint_path: str | Path,
    *,
    device: str | None,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MDN from candidate-set data files.")
    parser.add_argument("--data-dir", type=str, default="data/mdn_candidate_sets")
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--policy-checkpoint", type=str, default="models/mdn_policy_best.pth")
    parser.add_argument("--auxiliary-checkpoint", type=str, default="models/mdn_auxiliary_best.pth")
    parser.add_argument("--use-ips", action="store_true")
    parser.add_argument("--use-doubly-robust", action="store_true")
    parser.add_argument("--no-normalize-auxiliary-targets", action="store_true")
    parser.add_argument("--q-loss", choices=("mse", "huber"), default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--calibrate-auxiliary-q", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_mdn_from_candidate_set_directory(
        data_dir=args.data_dir,
        pattern=args.pattern,
        baseline_episodes=args.baseline_episodes,
        seed=args.seed,
        gamma=args.gamma,
        device=args.device,
        policy_checkpoint_path=args.policy_checkpoint,
        auxiliary_checkpoint_path=args.auxiliary_checkpoint,
        use_ips=args.use_ips,
        use_doubly_robust=args.use_doubly_robust,
        normalize_auxiliary_targets=not args.no_normalize_auxiliary_targets,
        q_loss=args.q_loss,
        huber_delta=args.huber_delta,
        calibrate_auxiliary_q=args.calibrate_auxiliary_q,
    )

    print("MDN Candidate-Set Training Complete")
    print("===================================")
    print(f"candidate outcomes: {result['candidate_outcomes']}")
    print(f"baseline payoff:    {result['baseline_payoff']:.4f}")
    print(f"policy checkpoint:  {result['policy']['checkpoint_path']}")
    print(f"aux checkpoint:     {result['auxiliary']['checkpoint_path']}")
    print(f"policy loss:        {float(result['policy']['loss']):.4f}")
    print(f"aux best val loss:  {float(result['auxiliary']['best_val_loss']):.4f}")
    print(f"aux q loss:         {args.q_loss}")
    normalization = result.get("auxiliary_target_normalization")
    if normalization:
        print(f"aux target mean:    {normalization['mean']}")
        print(f"aux target std:     {normalization['std']}")
    calibration = result.get("auxiliary_q_calibration")
    if calibration:
        print(f"aux q calibration:  slope={calibration['slope']} intercept={calibration['intercept']}")
        print(f"aux q train mse:    {calibration['train_mse_before']:.4f} -> {calibration['train_mse_after']:.4f}")


if __name__ == "__main__":
    main()

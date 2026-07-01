"""Evaluate a trained MDN checkpoint on held-out candidate-set data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baseline.idle_policy import IdlePolicy
from env.lunar_lander_wrapper import SubRepEnv
from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_checkpoint_loader import load_mdn_checkpoint as _load_mdn_checkpoint
from utils.mdn_data_adapter import candidate_set_directory_to_prepared_candidate_outcomes
from utils.mdn_record_builder import build_candidate_skill_records, group_candidate_outcomes_by_context
from utils.mdn_selection import alpha_to_mean_weights, score_candidate, select_best_candidate
from generator.train_mdn import _stable_skill_id


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


def load_mdn_checkpoint(path: str | Path, *, map_location: str = "cpu") -> MotiveDecompositionNetwork:
    return _load_mdn_checkpoint(path, map_location=map_location)


def load_auxiliary_target_normalization(path: str | Path, *, map_location: str = "cpu") -> dict[str, object] | None:
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        return None
    normalization = checkpoint.get("auxiliary_target_normalization")
    if not normalization:
        return None
    mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(normalization["std"], dtype=np.float32).reshape(-1)
    if mean.shape != std.shape or mean.shape[0] == 0:
        raise ValueError("Invalid auxiliary_target_normalization in checkpoint")
    return {
        "enabled": bool(normalization.get("enabled", True)),
        "mean": mean,
        "std": std,
        "count": int(normalization.get("count", 0)),
    }


def load_auxiliary_q_calibration(path: str | Path, *, map_location: str = "cpu") -> dict[str, object] | None:
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        return None
    calibration = checkpoint.get("auxiliary_q_calibration")
    if not calibration:
        return None
    slope = np.asarray(calibration["slope"], dtype=np.float32).reshape(-1)
    intercept = np.asarray(calibration["intercept"], dtype=np.float32).reshape(-1)
    if slope.shape != intercept.shape or slope.shape[0] == 0:
        raise ValueError("Invalid auxiliary_q_calibration in checkpoint")
    return {
        "enabled": bool(calibration.get("enabled", True)),
        "type": str(calibration.get("type", "affine")),
        "slope": slope,
        "intercept": intercept,
        "count": int(calibration.get("count", 0)),
    }


def evaluate_mdn_candidate_sets(
    *,
    checkpoint_path: str | Path = "models/mdn_policy_best.pth",
    data_dir: str | Path = "data/mdn_candidate_sets_eval",
    pattern: str = "*.npz",
    baseline_stats: dict[str, Any] | None = None,
    baseline_episodes: int = 20,
    seed: int = 100,
    gamma: float = 0.99,
    device: str = "cpu",
    gate_threshold: float = 0.5,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, float]:
    model = load_mdn_checkpoint(checkpoint_path, map_location=device)
    target_normalization = load_auxiliary_target_normalization(checkpoint_path, map_location=device)
    q_calibration = load_auxiliary_q_calibration(checkpoint_path, map_location=device)
    outcomes = candidate_set_directory_to_prepared_candidate_outcomes(data_dir, pattern=pattern)
    grouped = group_candidate_outcomes_by_context(outcomes)
    if baseline_stats is None:
        baseline_stats = compute_idle_baseline_stats(
            episodes=baseline_episodes,
            seed=seed,
            gamma=gamma,
        )

    selected_scores: list[float] = []
    random_expected_scores: list[float] = []
    ppo_scores: list[float] = []
    lift_vs_ppo_scores: list[float] = []
    predicted_weight_regrets: list[float] = []
    balanced_selected_scores: list[float] = []
    balanced_oracle_scores: list[float] = []
    balanced_top1_matches_per_context: list[float] = []
    certified_counts: list[int] = []
    alpha_weights: list[np.ndarray] = []
    gate_true: list[int] = []
    gate_pred: list[int] = []
    gate_probabilities: list[float] = []
    q_errors: list[np.ndarray] = []
    balanced_top1_matches = 0
    skipped_no_certified = 0

    for context, group in grouped.items():
        candidates = build_candidate_skill_records(
            skill_outcomes=group,
            baseline_stats=baseline_stats,
        )
        certified = [candidate for candidate in candidates if candidate.is_certified]
        context_tensor = torch.tensor(context, dtype=torch.float32, device=torch.device(device))
        for candidate, outcome in zip(candidates, group):
            skill_id = _stable_skill_id(candidate.skill_id, bucket_count=model.num_skills)
            skill_tensor = torch.tensor(skill_id, dtype=torch.long, device=torch.device(device))
            with torch.no_grad():
                gate_logit, q_hat = model.forward_auxiliary(context_tensor, skill_tensor)
            gate_probability = float(torch.sigmoid(gate_logit).item())
            gate_true.append(1 if candidate.is_certified else 0)
            gate_pred.append(1 if gate_probability >= gate_threshold else 0)
            gate_probabilities.append(gate_probability)
            target_motives = np.asarray(outcome.motives, dtype=np.float32).reshape(-1)
            q_prediction = q_hat.detach().cpu().numpy().reshape(-1)
            if target_normalization is not None and target_normalization["enabled"]:
                mean = np.asarray(target_normalization["mean"], dtype=np.float32).reshape(-1)
                std = np.asarray(target_normalization["std"], dtype=np.float32).reshape(-1)
                if q_prediction.shape != mean.shape:
                    raise ValueError(
                        f"q prediction shape {q_prediction.shape} does not match normalization shape {mean.shape}"
                    )
                q_prediction = q_prediction * std + mean
            if q_calibration is not None and q_calibration["enabled"]:
                if q_calibration["type"] != "affine":
                    raise ValueError(f"Unsupported auxiliary_q_calibration type {q_calibration['type']!r}")
                slope = np.asarray(q_calibration["slope"], dtype=np.float32).reshape(-1)
                intercept = np.asarray(q_calibration["intercept"], dtype=np.float32).reshape(-1)
                if q_prediction.shape != slope.shape:
                    raise ValueError(
                        f"q prediction shape {q_prediction.shape} does not match calibration shape {slope.shape}"
                    )
                q_prediction = q_prediction * slope + intercept
            q_errors.append(q_prediction - target_motives)

        if not certified:
            skipped_no_certified += 1
            continue

        with torch.no_grad():
            alpha, _ = model.forward_inference(context_tensor)
        weights = alpha_to_mean_weights(alpha.detach().cpu().numpy())
        alpha_weights.append(weights.reshape(-1))

        selected_id, selected_score = select_best_candidate(candidates, weights)
        _, predicted_oracle_score = select_best_candidate(candidates, weights)
        selected_scores.append(float(selected_score))
        predicted_weight_regrets.append(float(predicted_oracle_score - selected_score))
        random_expected_scores.append(float(np.mean([score_candidate(candidate, weights) for candidate in certified])))
        certified_counts.append(len(certified))

        ppo_candidate = next((candidate for candidate in certified if candidate.skill_id == "ppo_deterministic"), None)
        if ppo_candidate is not None:
            ppo_score = score_candidate(ppo_candidate, weights)
            ppo_scores.append(ppo_score)
            lift_vs_ppo_scores.append(float(selected_score - ppo_score))

        balanced_weights = np.full_like(weights.reshape(-1), 1.0 / len(weights.reshape(-1)), dtype=np.float32)
        selected_candidate = next(candidate for candidate in certified if candidate.skill_id == selected_id)
        balanced_selected_scores.append(score_candidate(selected_candidate, balanced_weights))
        balanced_oracle_id, balanced_oracle_score = select_best_candidate(candidates, balanced_weights)
        balanced_match = 1.0 if selected_id == balanced_oracle_id else 0.0
        balanced_top1_matches_per_context.append(balanced_match)
        if balanced_match:
            balanced_top1_matches += 1
        balanced_oracle_scores.append(float(balanced_oracle_score))

    if not selected_scores:
        raise ValueError("No evaluable contexts had certified candidates")

    alpha_array = np.stack(alpha_weights, axis=0)
    selected = np.asarray(selected_scores, dtype=np.float64)
    random_expected = np.asarray(random_expected_scores, dtype=np.float64)
    balanced_selected = np.asarray(balanced_selected_scores, dtype=np.float64)
    balanced_oracle = np.asarray(balanced_oracle_scores, dtype=np.float64)
    selected_minus_random = selected - random_expected
    lift_vs_ppo = np.asarray(lift_vs_ppo_scores, dtype=np.float64)
    predicted_weight_regret = np.asarray(predicted_weight_regrets, dtype=np.float64)
    balanced_regret = balanced_oracle - balanced_selected
    balanced_top1 = np.asarray(balanced_top1_matches_per_context, dtype=np.float64)
    gate_true_array = np.asarray(gate_true, dtype=np.int32)
    gate_pred_array = np.asarray(gate_pred, dtype=np.int32)
    true_positive = int(np.sum((gate_true_array == 1) & (gate_pred_array == 1)))
    false_positive = int(np.sum((gate_true_array == 0) & (gate_pred_array == 1)))
    true_negative = int(np.sum((gate_true_array == 0) & (gate_pred_array == 0)))
    false_negative = int(np.sum((gate_true_array == 1) & (gate_pred_array == 0)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    q_error_array = np.stack(q_errors, axis=0).astype(np.float64)
    q_squared_error = q_error_array ** 2
    q_absolute_error = np.abs(q_error_array)

    metrics = {
        "candidate_outcomes": float(len(outcomes)),
        "contexts_total": float(len(grouped)),
        "contexts_evaluated": float(len(selected_scores)),
        "contexts_skipped_no_certified": float(skipped_no_certified),
        "avg_certified_candidates": float(np.mean(certified_counts)),
        "mean_selected_score": float(np.mean(selected)),
        "mean_random_certified_score": float(np.mean(random_expected)),
        "mean_score_lift_vs_random": float(np.mean(selected_minus_random)),
        "mean_ppo_deterministic_score": float(np.mean(ppo_scores)) if ppo_scores else float("nan"),
        "mean_score_lift_vs_ppo": float(np.mean(lift_vs_ppo)) if len(lift_vs_ppo) else float("nan"),
        "ppo_deterministic_contexts": float(len(ppo_scores)),
        "mean_balanced_selected_score": float(np.mean(balanced_selected)),
        "mean_balanced_oracle_score": float(np.mean(balanced_oracle)),
        "mean_balanced_regret": float(np.mean(balanced_regret)),
        "mean_predicted_weight_regret": float(np.mean(predicted_weight_regret)),
        "balanced_top1_accuracy": float(balanced_top1_matches / len(selected_scores)),
        "gate_accuracy": float(np.mean(gate_true_array == gate_pred_array)),
        "gate_precision": float(precision),
        "gate_recall": float(recall),
        "gate_f1": float(f1),
        "gate_true_positive": float(true_positive),
        "gate_false_positive": float(false_positive),
        "gate_true_negative": float(true_negative),
        "gate_false_negative": float(false_negative),
        "mean_gate_probability": float(np.mean(gate_probabilities)),
        "q_motive_mse": float(np.mean(q_squared_error)),
        "q_motive_mae": float(np.mean(q_absolute_error)),
        "q_target_normalization_enabled": float(
            target_normalization is not None and bool(target_normalization["enabled"])
        ),
        "q_calibration_enabled": float(q_calibration is not None and bool(q_calibration["enabled"])),
        "mean_alpha_weight_0": float(np.mean(alpha_array[:, 0])),
        "mean_alpha_weight_1": float(np.mean(alpha_array[:, 1])) if alpha_array.shape[1] > 1 else float("nan"),
        "std_alpha_weight_0": float(np.std(alpha_array[:, 0])),
        "std_alpha_weight_1": float(np.std(alpha_array[:, 1])) if alpha_array.shape[1] > 1 else float("nan"),
    }
    for objective_index in range(q_error_array.shape[1]):
        metrics[f"q_motive_mse_{objective_index}"] = float(np.mean(q_squared_error[:, objective_index]))
        metrics[f"q_motive_mae_{objective_index}"] = float(np.mean(q_absolute_error[:, objective_index]))

    if bootstrap_samples > 0:
        metrics.update(
            _bootstrap_interval_metrics(
                {
                    "score_lift_vs_random": selected_minus_random,
                    "score_lift_vs_ppo": lift_vs_ppo,
                    "balanced_regret": balanced_regret,
                    "predicted_weight_regret": predicted_weight_regret,
                    "balanced_top1_accuracy": balanced_top1,
                },
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        )

    return metrics


def _bootstrap_interval_metrics(
    series_by_name: dict[str, np.ndarray],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    intervals: dict[str, float] = {}
    for name, values in series_by_name.items():
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            intervals[f"{name}_ci95_low"] = float("nan")
            intervals[f"{name}_ci95_high"] = float("nan")
            continue
        bootstrap_means = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            sample_indices = rng.integers(0, values.size, size=values.size)
            bootstrap_means[index] = float(np.mean(values[sample_indices]))
        low, high = np.percentile(bootstrap_means, [2.5, 97.5])
        intervals[f"{name}_ci95_low"] = float(low)
        intervals[f"{name}_ci95_high"] = float(high)
    return intervals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained MDN on held-out candidate-set data.")
    parser.add_argument("--checkpoint", type=str, default="models/mdn_policy_best.pth")
    parser.add_argument("--data-dir", type=str, default="data/mdn_candidate_sets_eval")
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_mdn_candidate_sets(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        pattern=args.pattern,
        baseline_episodes=args.baseline_episodes,
        seed=args.seed,
        gamma=args.gamma,
        device=args.device,
        gate_threshold=args.gate_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    print("MDN Held-Out Candidate-Set Evaluation")
    print("====================================")
    for key, value in metrics.items():
        if key.endswith("contexts") or key.startswith("contexts") or key == "candidate_outcomes":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

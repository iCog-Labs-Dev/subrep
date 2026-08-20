"""Evaluate a trained Safety-Gymnasium MDN on held-out rollout files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from generator.evaluate_mdn_candidate_sets import (
    load_auxiliary_q_calibration,
    load_auxiliary_target_normalization,
    load_mdn_checkpoint,
)
from generator.train_mdn import _stable_skill_id
from generator.train_safety_gymnasium_mdn import safety_rollout_directory_to_delta_outcomes
from utils.mdn_record_builder import build_candidate_skill_records, group_candidate_outcomes_by_context
from utils.mdn_selection import alpha_to_mean_weights, score_candidate, select_best_candidate


def evaluate_safety_gymnasium_mdn(
    *,
    checkpoint_path: str | Path,
    data_dir: str | Path,
    pattern: str = "*.npz",
    baseline_candidate_id: str = "zero_action",
    seed: int = 100,
    device: str = "cpu",
    gate_threshold: float = 0.5,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, float]:
    """Evaluate MDN selection and auxiliary heads on Safety-Gymnasium rollouts.

    Safety-Gymnasium rollout files contain same-context candidate outcomes. The
    baseline is the same-context ``zero_action`` candidate, so this evaluator
    converts every non-baseline candidate into baseline-relative payoff/motive
    deltas before evaluating the MDN.
    """
    del seed  # Kept for CLI symmetry with the MO-LunarLander evaluator.
    model = load_mdn_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    target_normalization = load_auxiliary_target_normalization(checkpoint_path, map_location=device)
    q_calibration = load_auxiliary_q_calibration(checkpoint_path, map_location=device)
    outcomes = safety_rollout_directory_to_delta_outcomes(
        data_dir,
        pattern=pattern,
        baseline_candidate_id=baseline_candidate_id,
    )
    grouped = group_candidate_outcomes_by_context(outcomes)
    baseline_stats = {
        "baseline_payoff": 0.0,
        "baseline_motives": np.zeros(len(outcomes[0].motives), dtype=np.float32),
    }
    device_obj = torch.device(device)

    selected_scores: list[float] = []
    random_expected_scores: list[float] = []
    ppo_scores: list[float] = []
    ppo_lagrangian_scores: list[float] = []
    lift_vs_ppo_scores: list[float] = []
    lift_vs_ppo_lagrangian_scores: list[float] = []
    balanced_selected_scores: list[float] = []
    balanced_oracle_scores: list[float] = []
    balanced_top1_matches: list[float] = []
    predicted_weight_regrets: list[float] = []
    certified_counts: list[int] = []
    alpha_weights: list[np.ndarray] = []
    support_values_seen: list[np.ndarray] = []
    gate_true: list[int] = []
    gate_pred: list[int] = []
    gate_probabilities: list[float] = []
    q_errors: list[np.ndarray] = []
    skipped_no_certified = 0

    for context, group in grouped.items():
        candidates = build_candidate_skill_records(
            skill_outcomes=group,
            baseline_stats=baseline_stats,
        )
        certified = [candidate for candidate in candidates if candidate.is_certified]
        context_tensor = torch.tensor(context, dtype=torch.float32, device=device_obj)

        for candidate, outcome in zip(candidates, group):
            skill_id = _stable_skill_id(candidate.skill_id, bucket_count=model.num_skills)
            skill_tensor = torch.tensor(skill_id, dtype=torch.long, device=device_obj)
            with torch.no_grad():
                gate_logit, q_hat = model.forward_auxiliary(context_tensor, skill_tensor)
            gate_probability = float(torch.sigmoid(gate_logit).item())
            gate_true.append(1 if candidate.is_certified else 0)
            gate_pred.append(1 if gate_probability >= gate_threshold else 0)
            gate_probabilities.append(gate_probability)

            target_motives = np.asarray(outcome.motives, dtype=np.float32).reshape(-1)
            q_prediction = q_hat.detach().cpu().numpy().reshape(-1)
            q_prediction = _denormalize_and_calibrate_q(
                q_prediction,
                target_normalization=target_normalization,
                q_calibration=q_calibration,
            )
            q_errors.append(q_prediction - target_motives)

        if not certified:
            skipped_no_certified += 1
            continue

        with torch.no_grad():
            alpha, support_values = model.forward_inference(context_tensor)
        weights = alpha_to_mean_weights(alpha.detach().cpu().numpy())
        support_values_np = support_values.detach().cpu().numpy().reshape(-1)
        alpha_weights.append(weights.reshape(-1))
        support_values_seen.append(support_values_np)

        selected_id, selected_score = select_best_candidate(candidates, weights)
        _, predicted_oracle_score = select_best_candidate(candidates, weights)
        selected_scores.append(float(selected_score))
        predicted_weight_regrets.append(float(predicted_oracle_score - selected_score))
        random_expected_scores.append(
            float(np.mean([score_candidate(candidate, weights) for candidate in certified]))
        )
        certified_counts.append(len(certified))

        ppo_candidate = next(
            (
                candidate
                for candidate in certified
                if _is_ppo_candidate(candidate.skill_id)
            ),
            None,
        )
        if ppo_candidate is not None:
            ppo_score = float(score_candidate(ppo_candidate, weights))
            ppo_scores.append(ppo_score)
            lift_vs_ppo_scores.append(float(selected_score - ppo_score))

        ppo_lagrangian_candidate = next(
            (
                candidate
                for candidate in certified
                if _is_ppo_lagrangian_candidate(candidate.skill_id)
            ),
            None,
        )
        if ppo_lagrangian_candidate is not None:
            ppo_lagrangian_score = float(score_candidate(ppo_lagrangian_candidate, weights))
            ppo_lagrangian_scores.append(ppo_lagrangian_score)
            lift_vs_ppo_lagrangian_scores.append(float(selected_score - ppo_lagrangian_score))

        balanced_weights = np.full_like(weights.reshape(-1), 1.0 / len(weights.reshape(-1)))
        selected_candidate = next(candidate for candidate in certified if candidate.skill_id == selected_id)
        balanced_selected_scores.append(float(score_candidate(selected_candidate, balanced_weights)))
        balanced_oracle_id, balanced_oracle_score = select_best_candidate(candidates, balanced_weights)
        balanced_top1_matches.append(1.0 if selected_id == balanced_oracle_id else 0.0)
        balanced_oracle_scores.append(float(balanced_oracle_score))

    if not selected_scores:
        raise ValueError("No evaluable contexts had certified candidates")

    selected = np.asarray(selected_scores, dtype=np.float64)
    random_expected = np.asarray(random_expected_scores, dtype=np.float64)
    balanced_selected = np.asarray(balanced_selected_scores, dtype=np.float64)
    balanced_oracle = np.asarray(balanced_oracle_scores, dtype=np.float64)
    selected_minus_random = selected - random_expected
    lift_vs_ppo = np.asarray(lift_vs_ppo_scores, dtype=np.float64)
    lift_vs_ppo_lagrangian = np.asarray(lift_vs_ppo_lagrangian_scores, dtype=np.float64)
    predicted_weight_regret = np.asarray(predicted_weight_regrets, dtype=np.float64)
    balanced_regret = balanced_oracle - balanced_selected
    balanced_top1 = np.asarray(balanced_top1_matches, dtype=np.float64)
    gate_true_array = np.asarray(gate_true, dtype=np.int32)
    gate_pred_array = np.asarray(gate_pred, dtype=np.int32)
    q_error_array = np.stack(q_errors, axis=0).astype(np.float64)
    q_squared_error = q_error_array ** 2
    q_absolute_error = np.abs(q_error_array)
    alpha_array = np.stack(alpha_weights, axis=0)
    support_array = np.stack(support_values_seen, axis=0)

    true_positive = int(np.sum((gate_true_array == 1) & (gate_pred_array == 1)))
    false_positive = int(np.sum((gate_true_array == 0) & (gate_pred_array == 1)))
    true_negative = int(np.sum((gate_true_array == 0) & (gate_pred_array == 0)))
    false_negative = int(np.sum((gate_true_array == 1) & (gate_pred_array == 0)))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics = {
        "candidate_outcomes": float(len(outcomes)),
        "contexts_total": float(len(grouped)),
        "contexts_evaluated": float(len(selected_scores)),
        "contexts_skipped_no_certified": float(skipped_no_certified),
        "num_objectives": float(q_error_array.shape[1]),
        "avg_certified_candidates": float(np.mean(certified_counts)),
        "mean_selected_score": float(np.mean(selected)),
        "mean_random_certified_score": float(np.mean(random_expected)),
        "mean_score_lift_vs_random": float(np.mean(selected_minus_random)),
        "mean_ppo_score": float(np.mean(ppo_scores)) if ppo_scores else float("nan"),
        "mean_score_lift_vs_ppo": float(np.mean(lift_vs_ppo)) if len(lift_vs_ppo) else float("nan"),
        "ppo_contexts": float(len(ppo_scores)),
        "mean_ppo_lagrangian_score": (
            float(np.mean(ppo_lagrangian_scores)) if ppo_lagrangian_scores else float("nan")
        ),
        "mean_score_lift_vs_ppo_lagrangian": (
            float(np.mean(lift_vs_ppo_lagrangian)) if len(lift_vs_ppo_lagrangian) else float("nan")
        ),
        "ppo_lagrangian_contexts": float(len(ppo_lagrangian_scores)),
        "mean_balanced_selected_score": float(np.mean(balanced_selected)),
        "mean_balanced_oracle_score": float(np.mean(balanced_oracle)),
        "mean_balanced_regret": float(np.mean(balanced_regret)),
        "mean_predicted_weight_regret": float(np.mean(predicted_weight_regret)),
        "balanced_top1_accuracy": float(np.mean(balanced_top1)),
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
        "support_min": float(np.min(support_array)),
        "support_max": float(np.max(support_array)),
        "support_sum_min": float(np.min(np.sum(support_array, axis=1))),
    }
    for objective_index in range(alpha_array.shape[1]):
        metrics[f"mean_alpha_weight_{objective_index}"] = float(np.mean(alpha_array[:, objective_index]))
        metrics[f"std_alpha_weight_{objective_index}"] = float(np.std(alpha_array[:, objective_index]))
        metrics[f"q_motive_mse_{objective_index}"] = float(np.mean(q_squared_error[:, objective_index]))
        metrics[f"q_motive_mae_{objective_index}"] = float(np.mean(q_absolute_error[:, objective_index]))

    if bootstrap_samples > 0:
        metrics.update(
            _bootstrap_interval_metrics(
                {
                    "score_lift_vs_random": selected_minus_random,
                    "score_lift_vs_ppo": lift_vs_ppo,
                    "score_lift_vs_ppo_lagrangian": lift_vs_ppo_lagrangian,
                    "balanced_regret": balanced_regret,
                    "balanced_top1_accuracy": balanced_top1,
                },
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        )

    return metrics


def _denormalize_and_calibrate_q(
    q_prediction: np.ndarray,
    *,
    target_normalization: dict[str, object] | None,
    q_calibration: dict[str, object] | None,
) -> np.ndarray:
    prediction = np.asarray(q_prediction, dtype=np.float32).reshape(-1)
    if target_normalization is not None and target_normalization["enabled"]:
        mean = np.asarray(target_normalization["mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(target_normalization["std"], dtype=np.float32).reshape(-1)
        if prediction.shape != mean.shape:
            raise ValueError(
                f"q prediction shape {prediction.shape} does not match normalization shape {mean.shape}"
            )
        prediction = prediction * std + mean
    if q_calibration is not None and q_calibration["enabled"]:
        if q_calibration["type"] != "affine":
            raise ValueError(f"Unsupported auxiliary_q_calibration type {q_calibration['type']!r}")
        slope = np.asarray(q_calibration["slope"], dtype=np.float32).reshape(-1)
        intercept = np.asarray(q_calibration["intercept"], dtype=np.float32).reshape(-1)
        if prediction.shape != slope.shape:
            raise ValueError(
                f"q prediction shape {prediction.shape} does not match calibration shape {slope.shape}"
            )
        prediction = prediction * slope + intercept
    return prediction


def _is_ppo_candidate(skill_id: str | None) -> bool:
    if skill_id is None:
        return False
    normalized = skill_id.lower()
    return "ppo" in normalized and "lagrangian" not in normalized


def _is_ppo_lagrangian_candidate(skill_id: str | None) -> bool:
    if skill_id is None:
        return False
    normalized = skill_id.lower()
    return "ppo" in normalized and "lagrangian" in normalized


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
    parser = argparse.ArgumentParser(
        description="Evaluate trained Safety-Gymnasium MDN on held-out rollout files."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--baseline-candidate", type=str, default="zero_action")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gate-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_safety_gymnasium_mdn(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        pattern=args.pattern,
        baseline_candidate_id=args.baseline_candidate,
        seed=args.seed,
        device=args.device,
        gate_threshold=args.gate_threshold,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    print("Safety-Gymnasium MDN Held-Out Evaluation")
    print("=======================================")
    for key, value in metrics.items():
        if key in {"candidate_outcomes", "contexts_total", "contexts_evaluated", "contexts_skipped_no_certified"}:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

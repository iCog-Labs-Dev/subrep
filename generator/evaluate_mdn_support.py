"""Held-out evaluation for context-conditioned MDN support bounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from generator.evaluate_mdn_candidate_sets import compute_idle_baseline_stats
from utils.mdn_checkpoint_loader import load_mdn_checkpoint
from utils.mdn_data_adapter import candidate_set_directory_to_prepared_candidate_outcomes
from utils.mdn_record_builder import PreparedCandidateOutcome
from utils.mdn_support_data import (
    SupportEvaluationCandidate,
    SupportEvaluationCandidateSet,
    runtime_logs_to_support_data,
)
from utils.mdn_stub import StubMDN
from utils.weight_set_store import WeightSetStore


def evaluate_support_models(
    model: torch.nn.Module,
    test_store: WeightSetStore,
    *,
    constant_baseline_support: np.ndarray | None = None,
    candidate_outcomes: Iterable[PreparedCandidateOutcome] | None = None,
    candidate_sets: Iterable[SupportEvaluationCandidateSet] | None = None,
    baseline_stats: dict[str, Any] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare learned, StubMDN, and full-simplex support on held-out contexts.

    The target is the coordinate-cap outer bound derived from the held-out
    store's observed certified weights. It is an empirical target, not an
    independently known safe-region boundary.
    """
    target_rows = test_store.get_all_support_targets()
    if not target_rows:
        raise ValueError("Support evaluation requires at least one held-out context")

    contexts = np.stack([row[0] for row in target_rows], axis=0).astype(np.float32)
    targets = np.stack([row[1] for row in target_rows], axis=0).astype(np.float32)
    num_objectives = int(targets.shape[1])
    if getattr(model, "num_objectives", num_objectives) != num_objectives:
        raise ValueError("Model objective count does not match held-out support targets")
    if getattr(model, "input_dim", contexts.shape[1]) != contexts.shape[1]:
        raise ValueError("Model input dimension does not match held-out support contexts")

    learned_support = _predict_support(model, contexts, device=device)
    stub = StubMDN(
        input_dim=contexts.shape[1],
        num_objectives=num_objectives,
        fixed_alpha=[1.0] * num_objectives,
        fixed_support_values=[1.0] * num_objectives,
    ).to(device)
    stub_support = _predict_support(stub, contexts, device=device)
    full_simplex_support = np.ones_like(targets, dtype=np.float32)
    predictions_by_arm = {
        "learned": learned_support,
        "stub_mdn": stub_support,
        "full_simplex": full_simplex_support,
    }
    if constant_baseline_support is not None:
        constant_support = np.asarray(constant_baseline_support, dtype=np.float32).reshape(-1)
        if constant_support.shape != (num_objectives,):
            raise ValueError(
                "constant_baseline_support must match the target objective count, "
                f"got {constant_support.shape}"
            )
        evaluate_support_predictions(
            np.broadcast_to(constant_support, (1, num_objectives)),
            np.broadcast_to(constant_support, (1, num_objectives)),
        )
        predictions_by_arm["train_mean_constant"] = np.broadcast_to(
            constant_support, targets.shape
        ).copy()

    arms = {
        name: evaluate_support_predictions(predictions, targets)
        for name, predictions in predictions_by_arm.items()
    }

    outcomes = tuple(candidate_outcomes or ())
    normalized_candidate_sets = tuple(candidate_sets or ())
    if outcomes and normalized_candidate_sets:
        raise ValueError("Provide candidate_outcomes or candidate_sets, not both")
    if outcomes and baseline_stats is None:
        raise ValueError("baseline_stats are required when candidate outcomes are provided")
    if outcomes:
        normalized_candidate_sets = _candidate_sets_from_outcomes(
            outcomes,
            baseline_stats=baseline_stats or {},
        )
    if normalized_candidate_sets:
        downstream = _evaluate_downstream(
            test_store,
            target_rows,
            predictions_by_arm,
            normalized_candidate_sets,
        )
    else:
        downstream = {
            name: _unavailable_downstream_metrics(test_store.context_count())
            for name in predictions_by_arm
        }
    for name in arms:
        arms[name].update(downstream[name])

    return {
        "target_semantics": (
            "coordinate-wise maxima of held-out observed certified-selection weights; "
            "an empirical cap outer approximation, not independent safe-region ground truth"
        ),
        "held_out_contexts": int(test_store.context_count()),
        "held_out_vertices": int(test_store.total_vertex_count()),
        "candidate_outcomes_supplied": int(
            sum(len(candidate_set.candidates) for candidate_set in normalized_candidate_sets)
        ),
        "stub_full_simplex_support_equivalent": bool(
            np.array_equal(stub_support, full_simplex_support)
        ),
        "arms": arms,
        "comparisons": _arm_comparisons(arms),
    }


def evaluate_support_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Compute held-out support regression and geometry metrics."""
    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[1] == 0:
        raise ValueError(f"predictions must have shape (N, M), got {predicted.shape}")
    if predicted.shape != target.shape:
        raise ValueError(
            f"predictions shape {predicted.shape} must match targets shape {target.shape}"
        )
    if predicted.shape[0] == 0:
        raise ValueError("support evaluation requires at least one prediction")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(target)):
        raise ValueError("support predictions and targets must be finite")

    error = predicted - target
    violation = (
        np.any((predicted < 0.0) | (predicted > 1.0), axis=1)
        | (np.sum(predicted, axis=1) < 1.0 - tolerance)
    )
    return {
        "support_region_feasibility_violation_rate": float(np.mean(violation)),
        "support_value_mse": float(np.mean(error ** 2)),
        "support_value_rmse": float(np.sqrt(np.mean(error ** 2))),
        "support_value_mae": float(np.mean(np.abs(error))),
        "support_coordinate_underestimate_rate": float(np.mean(error < -tolerance)),
        "support_coordinate_overestimate_rate": float(np.mean(error > tolerance)),
        "mean_predicted_support_sum": float(np.mean(np.sum(predicted, axis=1))),
        "mean_target_support_sum": float(np.mean(np.sum(target, axis=1))),
        "mean_prediction_context_variance": float(np.mean(np.var(predicted, axis=0))),
        "mean_target_context_variance": float(np.mean(np.var(target, axis=0))),
    }


def write_support_evaluation_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a compact Markdown report alongside the machine-readable JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MDN Support-Head Held-Out Evaluation",
        "",
        f"- Held-out contexts: {report['held_out_contexts']}",
        f"- Held-out observed weights: {report['held_out_vertices']}",
        f"- Candidate outcomes supplied: {report['candidate_outcomes_supplied']}",
        f"- Stub support equals FULL_SIMPLEX: {report['stub_full_simplex_support_equivalent']}",
        f"- Target semantics: {report['target_semantics']}",
        "",
        "## Results",
        "",
        "| Arm | Support MSE | Feasibility violations | CDS agreement | PDS agreement | False admission | False rejection | Shift coverage | Reuse success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["arms"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_metric(metrics["support_value_mse"]),
                    _format_metric(metrics["support_region_feasibility_violation_rate"]),
                    _format_metric(metrics["cds_agreement_rate"]),
                    _format_metric(metrics["pds_agreement_rate"]),
                    _format_metric(metrics["false_admission_rate"]),
                    _format_metric(metrics["false_rejection_rate"]),
                    _format_metric(metrics["reuse_motive_shift_context_coverage"]),
                    _format_metric(metrics["reuse_success_rate"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Lower support error and lower false-admission/false-rejection rates are better. "
            "Reuse success is the fraction of held-out observed motive shifts where the arm "
            "returns a candidate that is admitted under the held-out target region. Exact "
            "oracle top-1 agreement is reported in the JSON artifact.",
            "Reuse metrics require at least two distinct observed weights at a context; "
            "the JSON artifact reports motive-shift context coverage.",
            "",
            "Metrics are `nan` when no matching candidate contexts or no relevant class "
            "denominator exists. Such values are missing evidence, not zero error.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json_report(report: dict[str, Any], path: str | Path) -> None:
    """Write strict JSON, representing unavailable non-finite metrics as null."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _predict_support(
    model: torch.nn.Module,
    contexts: np.ndarray,
    *,
    device: str,
) -> np.ndarray:
    model.to(device)
    model.eval()
    context_tensor = torch.tensor(contexts, dtype=torch.float32, device=torch.device(device))
    with torch.no_grad():
        _, support = model.forward_inference(context_tensor)
    return support.detach().cpu().numpy().astype(np.float32)


def _evaluate_downstream(
    store: WeightSetStore,
    target_rows: list[tuple[np.ndarray, np.ndarray]],
    predictions_by_arm: dict[str, np.ndarray],
    candidate_sets: tuple[SupportEvaluationCandidateSet, ...],
) -> dict[str, dict[str, float]]:
    target_by_key = {
        store.context_key(context): target for context, target in target_rows
    }
    prediction_by_arm_key = {
        name: {
            store.context_key(context): predictions[index]
            for index, (context, _) in enumerate(target_rows)
        }
        for name, predictions in predictions_by_arm.items()
    }
    vertices_by_key = {
        store.context_key(context): vertices
        for context, vertices in store.get_all_context_vertices()
    }
    grouped: dict[tuple[float, ...], list[SupportEvaluationCandidateSet]] = {}
    for candidate_set in candidate_sets:
        key = store.context_key(np.asarray(candidate_set.context, dtype=np.float32))
        grouped.setdefault(key, []).append(candidate_set)

    accumulators = {name: _new_downstream_accumulator() for name in predictions_by_arm}
    matched_context_keys: set[tuple[float, ...]] = set()
    motive_shift_context_keys: set[tuple[float, ...]] = set()
    for key, target_support in target_by_key.items():
        context_groups = tuple(grouped.get(key, ()))
        if context_groups:
            matched_context_keys.add(key)
        motive_shifts = np.unique(vertices_by_key[key], axis=0)
        has_motive_shift = len(motive_shifts) >= 2
        if context_groups and has_motive_shift:
            motive_shift_context_keys.add(key)
        for candidate_set in context_groups:
            candidates = list(candidate_set.candidates)
            target_admission = [
                _candidate_admitted(candidate, target_support) for candidate in candidates
            ]
            for arm_name, arm_predictions in prediction_by_arm_key.items():
                predicted_admission = [
                    _candidate_admitted(candidate, arm_predictions[key])
                    for candidate in candidates
                ]
                _accumulate_classification(
                    accumulators[arm_name], candidates, target_admission, predicted_admission
                )
                if has_motive_shift:
                    _accumulate_reuse(
                        accumulators[arm_name],
                        candidates,
                        target_admission,
                        predicted_admission,
                        motive_shifts,
                    )

    return {
        name: _finalize_downstream_metrics(
            accumulator,
            matched_contexts=len(matched_context_keys),
            motive_shift_contexts=len(motive_shift_context_keys),
            held_out_contexts=store.context_count(),
        )
        for name, accumulator in accumulators.items()
    }


def _candidate_sets_from_outcomes(
    outcomes: tuple[PreparedCandidateOutcome, ...],
    *,
    baseline_stats: dict[str, Any],
) -> tuple[SupportEvaluationCandidateSet, ...]:
    calculator = ImprovementCalculator(baseline_stats)
    grouped: dict[tuple[tuple[float, ...], str], list[SupportEvaluationCandidate]] = {}
    for outcome in outcomes:
        delta_r, delta_n = calculator.compute_improvements(outcome.payoff, outcome.motives)
        source = str(outcome.metadata.get("candidate_set_path", ""))
        key = (outcome.context, source)
        grouped.setdefault(key, []).append(
            SupportEvaluationCandidate(
                skill_id=outcome.skill_id,
                delta_r=delta_r,
                delta_n=tuple(float(value) for value in delta_n),
                gate_type=outcome.gate_type,
                epsilon=outcome.epsilon,
            )
        )
    return tuple(
        SupportEvaluationCandidateSet(
            context=context,
            source=source,
            candidates=tuple(candidates),
        )
        for (context, source), candidates in grouped.items()
    )


def _candidate_admitted(
    candidate: SupportEvaluationCandidate,
    support_values: np.ndarray,
) -> bool:
    if candidate.gate_type == "CDS":
        gate = CDSGate()
    else:
        epsilon = 0.1 if candidate.epsilon is None else float(candidate.epsilon)
        gate = PDSGate(epsilon=epsilon)
    return gate.admit(
        candidate.delta_r,
        np.asarray(candidate.delta_n, dtype=np.float32),
        support_values=support_values,
    )


def _new_downstream_accumulator() -> dict[str, float]:
    return {
        "total": 0.0,
        "correct": 0.0,
        "positive": 0.0,
        "negative": 0.0,
        "false_positive": 0.0,
        "false_negative": 0.0,
        "cds_total": 0.0,
        "cds_correct": 0.0,
        "pds_total": 0.0,
        "pds_correct": 0.0,
        "reuse_opportunities": 0.0,
        "reuse_successes": 0.0,
        "reuse_top1_matches": 0.0,
        "reuse_regret_sum": 0.0,
        "reuse_regret_count": 0.0,
    }


def _accumulate_classification(
    accumulator: dict[str, float],
    candidates: list[SupportEvaluationCandidate],
    target: list[bool],
    predicted: list[bool],
) -> None:
    for candidate, truth, estimate in zip(candidates, target, predicted):
        accumulator["total"] += 1.0
        accumulator["correct"] += float(truth == estimate)
        accumulator["positive"] += float(truth)
        accumulator["negative"] += float(not truth)
        accumulator["false_positive"] += float(estimate and not truth)
        accumulator["false_negative"] += float(truth and not estimate)
        gate_prefix = "cds" if candidate.gate_type == "CDS" else "pds"
        accumulator[f"{gate_prefix}_total"] += 1.0
        accumulator[f"{gate_prefix}_correct"] += float(truth == estimate)


def _accumulate_reuse(
    accumulator: dict[str, float],
    candidates: list[SupportEvaluationCandidate],
    target_admission: list[bool],
    predicted_admission: list[bool],
    motive_shifts: np.ndarray,
) -> None:
    target_indices = [index for index, admitted in enumerate(target_admission) if admitted]
    if not target_indices:
        return
    predicted_indices = [
        index for index, admitted in enumerate(predicted_admission) if admitted
    ]
    for weights in motive_shifts:
        accumulator["reuse_opportunities"] += 1.0
        target_index, target_score = _select_index(candidates, target_indices, weights)
        if not predicted_indices:
            continue
        predicted_index, predicted_score = _select_index(
            candidates, predicted_indices, weights
        )
        if not target_admission[predicted_index]:
            continue
        accumulator["reuse_successes"] += 1.0
        accumulator["reuse_top1_matches"] += float(predicted_index == target_index)
        accumulator["reuse_regret_sum"] += float(target_score - predicted_score)
        accumulator["reuse_regret_count"] += 1.0


def _select_index(
    candidates: list[SupportEvaluationCandidate],
    indices: list[int],
    weights: np.ndarray,
) -> tuple[int, float]:
    scored = [
        (
            index,
            float(
                candidates[index].delta_r
                + np.dot(weights, candidates[index].delta_n)
            ),
        )
        for index in indices
    ]
    best_index, best_score = min(
        scored,
        key=lambda item: (-item[1], candidates[item[0]].skill_id),
    )
    return best_index, best_score


def _finalize_downstream_metrics(
    accumulator: dict[str, float],
    *,
    matched_contexts: int,
    motive_shift_contexts: int,
    held_out_contexts: int,
) -> dict[str, float]:
    return {
        "candidate_contexts_matched": float(matched_contexts),
        "candidate_context_coverage": _safe_rate(float(matched_contexts), float(held_out_contexts)),
        "reuse_motive_shift_contexts": float(motive_shift_contexts),
        "reuse_motive_shift_context_coverage": _safe_rate(
            float(motive_shift_contexts), float(held_out_contexts)
        ),
        "candidate_admission_decisions": accumulator["total"],
        "admission_agreement_rate": _safe_rate(accumulator["correct"], accumulator["total"]),
        "cds_agreement_rate": _safe_rate(accumulator["cds_correct"], accumulator["cds_total"]),
        "pds_agreement_rate": _safe_rate(accumulator["pds_correct"], accumulator["pds_total"]),
        "false_admission_rate": _safe_rate(accumulator["false_positive"], accumulator["negative"]),
        "false_rejection_rate": _safe_rate(accumulator["false_negative"], accumulator["positive"]),
        "false_admissions": accumulator["false_positive"],
        "false_rejections": accumulator["false_negative"],
        "reuse_shift_opportunities": accumulator["reuse_opportunities"],
        "reuse_success_rate": _safe_rate(
            accumulator["reuse_successes"], accumulator["reuse_opportunities"]
        ),
        "reuse_oracle_top1_agreement_rate": _safe_rate(
            accumulator["reuse_top1_matches"], accumulator["reuse_opportunities"]
        ),
        "reuse_mean_score_regret_on_success": _safe_rate(
            accumulator["reuse_regret_sum"], accumulator["reuse_regret_count"]
        ),
    }


def _unavailable_downstream_metrics(held_out_contexts: int) -> dict[str, float]:
    accumulator = _new_downstream_accumulator()
    return _finalize_downstream_metrics(
        accumulator,
        matched_contexts=0,
        motive_shift_contexts=0,
        held_out_contexts=held_out_contexts,
    )


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def _arm_comparisons(arms: dict[str, dict[str, float]]) -> dict[str, float]:
    learned = arms["learned"]
    full_simplex = arms["full_simplex"]
    comparisons = {
        "learned_support_mse_improvement_vs_full_simplex": float(
            full_simplex["support_value_mse"] - learned["support_value_mse"]
        ),
        "learned_reuse_success_lift_vs_full_simplex": float(
            learned["reuse_success_rate"] - full_simplex["reuse_success_rate"]
        ),
        "learned_false_admission_change_vs_full_simplex": float(
            learned["false_admission_rate"] - full_simplex["false_admission_rate"]
        ),
        "learned_false_rejection_change_vs_full_simplex": float(
            learned["false_rejection_rate"] - full_simplex["false_rejection_rate"]
        ),
    }
    if "train_mean_constant" in arms:
        constant = arms["train_mean_constant"]
        comparisons["learned_support_mse_improvement_vs_train_mean_constant"] = float(
            constant["support_value_mse"] - learned["support_value_mse"]
        )
        comparisons["learned_support_mse_skill_score_vs_train_mean_constant"] = (
            float(1.0 - learned["support_value_mse"] / constant["support_value_mse"])
            if constant["support_value_mse"] > 0.0
            else float("nan")
        )
    return comparisons


def _format_metric(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an MDN support head on held-out targets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-store", required=True)
    parser.add_argument(
        "--training-store",
        help="Optional training split used only to construct a context-independent mean baseline.",
    )
    parser.add_argument("--candidate-data-dir")
    parser.add_argument(
        "--runtime-log-dir",
        help="Validated probability-aware logs used for held-out candidate metrics.",
    )
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--baseline-episodes", type=int, default=20)
    parser.add_argument("--baseline-seed", type=int, default=900_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-output", default="data/mdn_support_evaluation/test_metrics.json")
    parser.add_argument("--markdown-output", default="data/mdn_support_evaluation/test_report.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_mdn_checkpoint(args.checkpoint, map_location=args.device)
    store = WeightSetStore.load(args.test_store)
    constant_baseline_support = None
    if args.training_store:
        training_store = WeightSetStore.load(args.training_store)
        training_targets = training_store.get_all_support_targets()
        if not training_targets:
            raise ValueError("training-store contains no support targets")
        constant_baseline_support = np.mean(
            np.stack([target for _, target in training_targets], axis=0), axis=0
        )
    outcomes = None
    candidate_sets = None
    if args.runtime_log_dir:
        _, candidate_sets = runtime_logs_to_support_data(
            args.runtime_log_dir,
            pattern=args.pattern,
        )
    baseline_stats = None
    if args.candidate_data_dir:
        if candidate_sets is not None:
            raise ValueError("Provide --candidate-data-dir or --runtime-log-dir, not both")
        outcomes = candidate_set_directory_to_prepared_candidate_outcomes(
            args.candidate_data_dir, pattern=args.pattern
        )
        baseline_stats = compute_idle_baseline_stats(
            episodes=args.baseline_episodes,
            seed=args.baseline_seed,
            gamma=args.gamma,
        )
    report = evaluate_support_models(
        model,
        store,
        constant_baseline_support=constant_baseline_support,
        candidate_outcomes=outcomes,
        candidate_sets=candidate_sets,
        baseline_stats=baseline_stats,
        device=args.device,
    )
    json_path = Path(args.json_output)
    write_json_report(report, json_path)
    write_support_evaluation_report(report, args.markdown_output)
    print(f"Support evaluation JSON: {json_path}")
    print(f"Support evaluation report: {args.markdown_output}")


if __name__ == "__main__":
    main()

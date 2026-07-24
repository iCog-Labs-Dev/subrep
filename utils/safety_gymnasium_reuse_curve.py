"""Zero-shot reuse curve utilities for Safety-Gymnasium benchmark artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/subrep_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate


def build_safety_gymnasium_reuse_curve(
    *,
    rollout_dirs: Sequence[str | Path],
    output_path: str | Path = "demo/artifacts/safety_gymnasium_zero_shot_reuse_curve.png",
    summary_json_path: str | Path = "demo/artifacts/safety_gymnasium_zero_shot_reuse_curve.json",
    pattern: str = "*.npz",
    baseline_candidate_id: str = "zero_action",
    pds_epsilon: float = 1.0,
    shift_points: Sequence[float] | None = None,
    baseline_retraining_steps: int = 51_200,
) -> dict:
    """Build a zero-shot reuse curve from rollout files.

    A shifted context counts as successful when SubRep can select a certified
    candidate and that candidate scores at least as well as the same-context
    zero-action baseline under the shifted motive weight.
    """
    shift_points = (
        tuple(float(v) for v in shift_points)
        if shift_points is not None
        else tuple(float(v) for v in np.linspace(0.0, 1.0, 11))
    )
    contexts = _load_certified_contexts(
        rollout_dirs=rollout_dirs,
        pattern=pattern,
        baseline_candidate_id=baseline_candidate_id,
        pds_epsilon=pds_epsilon,
    )
    if not contexts:
        raise ValueError("No contexts available for zero-shot reuse curve")

    points = []
    for safety_weight in shift_points:
        weight = np.asarray([safety_weight, 1.0 - safety_weight], dtype=np.float64)
        successes = 0
        selected_scores: list[float] = []
        certified_counts: list[int] = []
        for context in contexts:
            certified = context["certified_candidates"]
            certified_counts.append(len(certified))
            if not certified:
                continue
            selected = max(
                certified,
                key=lambda candidate: (
                    _score_candidate(candidate, weight),
                    candidate["candidate_policy"],
                ),
            )
            score = _score_candidate(selected, weight)
            selected_scores.append(score)
            if score >= 0.0:
                successes += 1

        points.append(
            {
                "safety_weight": float(safety_weight),
                "task_weight": float(1.0 - safety_weight),
                "success_rate": float(successes / len(contexts)),
                "successful_contexts": int(successes),
                "total_contexts": int(len(contexts)),
                "mean_selected_score": _mean(selected_scores),
                "avg_certified_candidates": _mean(certified_counts),
                "subrep_retraining_steps": 0,
                "baseline_retraining_steps": int(baseline_retraining_steps),
            }
        )

    summary = {
        "rollout_dirs": [str(path) for path in rollout_dirs],
        "pds_epsilon": float(pds_epsilon),
        "definition": (
            "success = certified candidate exists and selected shifted-weight "
            "score is >= zero-action baseline score"
        ),
        "baseline_retraining_steps": int(baseline_retraining_steps),
        "total_contexts": len(contexts),
        "curve": points,
        "mean_success_rate": _mean(point["success_rate"] for point in points),
        "min_success_rate": min(point["success_rate"] for point in points),
        "max_success_rate": max(point["success_rate"] for point in points),
    }

    save_reuse_curve_plot(summary, output_path)
    json_path = Path(summary_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_reuse_curve_plot(summary: dict, output_path: str | Path) -> None:
    curve = summary["curve"]
    x = [point["safety_weight"] for point in curve]
    success = [100.0 * point["success_rate"] for point in curve]
    baseline_steps = [point["baseline_retraining_steps"] for point in curve]
    subrep_steps = [point["subrep_retraining_steps"] for point in curve]

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    axes[0].plot(x, success, marker="o", color="#1f77b4", linewidth=2.0)
    axes[0].set_ylabel("Successful Reuse (%)")
    axes[0].set_ylim(0, 105)
    axes[0].set_title("SubRep Zero-Shot Reuse Under Motive Weight Shifts")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, subrep_steps, marker="o", color="#1f77b4", label="SubRep reuse")
    axes[1].plot(
        x,
        baseline_steps,
        marker="s",
        linestyle="--",
        color="#d62728",
        label="Baseline retraining",
    )
    axes[1].set_xlabel("Safety Weight After Shift")
    axes[1].set_ylabel("Environment Steps")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _load_certified_contexts(
    *,
    rollout_dirs: Sequence[str | Path],
    pattern: str,
    baseline_candidate_id: str,
    pds_epsilon: float,
) -> list[dict]:
    files: list[Path] = []
    for rollout_dir in rollout_dirs:
        files.extend(sorted(Path(rollout_dir).glob(pattern)))
    if not files:
        raise FileNotFoundError("No rollout files found for zero-shot reuse curve")

    contexts = []
    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=pds_epsilon)
    for path in files:
        record = _load_record(path)
        ids = record["candidate_skill_ids"]
        if baseline_candidate_id not in ids:
            raise ValueError(f"{path} does not contain baseline {baseline_candidate_id!r}")

        baseline_idx = ids.index(baseline_candidate_id)
        calculator = ImprovementCalculator(
            {
                "baseline_payoff": float(record["candidate_payoffs"][baseline_idx]),
                "baseline_motives": record["candidate_motives"][baseline_idx],
            }
        )
        certified = []
        for idx, candidate_policy in enumerate(ids):
            if candidate_policy == baseline_candidate_id:
                continue
            delta_r, delta_n = calculator.compute_improvements(
                float(record["candidate_payoffs"][idx]),
                record["candidate_motives"][idx],
            )
            if cds_gate.admit(delta_r, delta_n) or pds_gate.admit(delta_r, delta_n):
                certified.append(
                    {
                        "candidate_policy": candidate_policy,
                        "delta_r": float(delta_r),
                        "delta_n": tuple(float(v) for v in delta_n),
                    }
                )
        contexts.append(
            {
                "context_seed": int(record["context_seed"]),
                "certified_candidates": certified,
            }
        )
    return contexts


def _load_record(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {
        "context_seed": int(np.asarray(data["context_seed"]).item()),
        "candidate_skill_ids": [_scalar_to_string(item) for item in data["candidate_skill_ids"]],
        "candidate_payoffs": np.asarray(data["candidate_payoffs"], dtype=np.float32),
        "candidate_motives": np.asarray(data["candidate_motives"], dtype=np.float32),
    }


def _score_candidate(candidate: dict, weight: np.ndarray) -> float:
    return float(candidate["delta_r"]) + float(
        np.dot(weight, np.asarray(candidate["delta_n"], dtype=np.float64))
    )


def _scalar_to_string(value) -> str:
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _mean(values) -> float:
    collected = [float(value) for value in values]
    if not collected:
        return 0.0
    return float(np.mean(collected))

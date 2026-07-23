"""Pareto plot utilities for Safety-Gymnasium SubRep benchmark artifacts."""

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


DEFAULT_SELECTION_WEIGHT = (0.10, 0.90)
METHOD_LABELS = {
    "subrep": "SubRep Certified",
    "ppo": "PPO",
    "ppo_lagrangian": "PPO-Lagrangian",
}


def build_safety_gymnasium_pareto_report(
    *,
    rollout_dirs: Sequence[str | Path],
    output_path: str | Path = "demo/artifacts/safety_gymnasium_pareto_frontier.png",
    summary_json_path: str | Path = "demo/artifacts/safety_gymnasium_pareto_frontier.json",
    pattern: str = "*.npz",
    baseline_candidate_id: str = "zero_action",
    pds_epsilon: float = 1.0,
    selection_weight: Sequence[float] = DEFAULT_SELECTION_WEIGHT,
) -> dict:
    """Build Pareto summary and plot from Safety-Gymnasium rollout files."""
    points = collect_pareto_points(
        rollout_dirs=rollout_dirs,
        pattern=pattern,
        baseline_candidate_id=baseline_candidate_id,
        pds_epsilon=pds_epsilon,
        selection_weight=selection_weight,
    )
    summary = summarize_pareto_points(points)
    summary.update(
        {
            "rollout_dirs": [str(path) for path in rollout_dirs],
            "selection_weight": [float(v) for v in selection_weight],
            "pds_epsilon": float(pds_epsilon),
        }
    )
    save_pareto_plot(points, summary, output_path)

    json_path = Path(summary_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def collect_pareto_points(
    *,
    rollout_dirs: Sequence[str | Path],
    pattern: str,
    baseline_candidate_id: str,
    pds_epsilon: float,
    selection_weight: Sequence[float],
) -> dict[str, list[dict]]:
    files: list[Path] = []
    for rollout_dir in rollout_dirs:
        files.extend(sorted(Path(rollout_dir).glob(pattern)))
    if not files:
        raise FileNotFoundError("No rollout files found for Pareto report")

    points: dict[str, list[dict]] = {
        "subrep": [],
        "ppo": [],
        "ppo_lagrangian": [],
    }
    for path in files:
        record = _load_record(path)
        ids = record["candidate_skill_ids"]
        if baseline_candidate_id not in ids:
            raise ValueError(f"{path} does not contain baseline {baseline_candidate_id!r}")

        selected = _select_subrep_candidate(
            record,
            baseline_candidate_id=baseline_candidate_id,
            pds_epsilon=pds_epsilon,
            selection_weight=selection_weight,
        )
        if selected is not None:
            points["subrep"].append(selected)

        ppo = _candidate_point(record, "ppo_deterministic")
        if ppo is not None:
            points["ppo"].append(ppo)

        ppo_lagrangian = _candidate_point(record, "ppo_lagrangian_deterministic")
        if ppo_lagrangian is not None:
            points["ppo_lagrangian"].append(ppo_lagrangian)

    return points


def summarize_pareto_points(points: dict[str, list[dict]]) -> dict:
    method_summaries = {}
    mean_points = []
    for method, rows in points.items():
        costs = [row["safety_cost"] for row in rows]
        returns = [row["task_return"] for row in rows]
        summary = {
            "label": METHOD_LABELS[method],
            "available": bool(rows),
            "count": len(rows),
            "mean_safety_cost": _mean(costs),
            "std_safety_cost": _std(costs),
            "mean_task_return": _mean(returns),
            "std_task_return": _std(returns),
            "zero_cost_rate": _mean(float(cost <= 1e-8) for cost in costs),
        }
        method_summaries[method] = summary
        if rows:
            mean_points.append(
                {
                    "method": method,
                    "label": METHOD_LABELS[method],
                    "safety_cost": summary["mean_safety_cost"],
                    "task_return": summary["mean_task_return"],
                }
            )

    return {
        "methods": method_summaries,
        "pareto_frontier": _pareto_frontier(mean_points),
    }


def save_pareto_plot(points: dict[str, list[dict]], summary: dict, output_path: str | Path) -> None:
    colors = {
        "subrep": "#1f77b4",
        "ppo": "#ff7f0e",
        "ppo_lagrangian": "#2ca02c",
    }
    markers = {
        "subrep": "o",
        "ppo": "s",
        "ppo_lagrangian": "^",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for method, rows in points.items():
        label = METHOD_LABELS[method]
        if not rows:
            continue
        costs = [row["safety_cost"] for row in rows]
        returns = [row["task_return"] for row in rows]
        ax.scatter(
            costs,
            returns,
            s=16,
            alpha=0.25,
            color=colors[method],
            marker=markers[method],
            label=f"{label} contexts",
        )
        method_summary = summary["methods"][method]
        ax.scatter(
            [method_summary["mean_safety_cost"]],
            [method_summary["mean_task_return"]],
            s=130,
            color=colors[method],
            marker=markers[method],
            edgecolor="black",
            linewidth=1.0,
            label=f"{label} mean",
        )

    frontier = summary.get("pareto_frontier", [])
    if len(frontier) >= 2:
        ax.plot(
            [point["safety_cost"] for point in frontier],
            [point["task_return"] for point in frontier],
            linestyle="--",
            color="#333333",
            linewidth=1.5,
            label="Mean-method Pareto frontier",
        )

    ax.set_xlabel("Safety Cost / Constraint Violation (lower is better)")
    ax.set_ylabel("Task Return (higher is better)")
    ax.set_title("SubRep SafeRL Pareto Plot: SafetyPointGoal1-v0")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    unavailable = [
        METHOD_LABELS[method]
        for method, method_summary in summary["methods"].items()
        if not method_summary["available"]
    ]
    if unavailable:
        ax.text(
            0.01,
            0.01,
            "Unavailable in current rollouts: " + ", ".join(unavailable),
            transform=ax.transAxes,
            fontsize=8,
            color="#555555",
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _select_subrep_candidate(
    record: dict,
    *,
    baseline_candidate_id: str,
    pds_epsilon: float,
    selection_weight: Sequence[float],
) -> dict | None:
    ids = record["candidate_skill_ids"]
    baseline_idx = ids.index(baseline_candidate_id)
    calculator = ImprovementCalculator(
        {
            "baseline_payoff": float(record["candidate_payoffs"][baseline_idx]),
            "baseline_motives": record["candidate_motives"][baseline_idx],
        }
    )
    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=pds_epsilon)
    weight = np.asarray(selection_weight, dtype=np.float64)

    best: dict | None = None
    best_score = -np.inf
    for idx, candidate_id in enumerate(ids):
        if candidate_id == baseline_candidate_id:
            continue
        delta_r, delta_n = calculator.compute_improvements(
            float(record["candidate_payoffs"][idx]),
            record["candidate_motives"][idx],
        )
        if not (cds_gate.admit(delta_r, delta_n) or pds_gate.admit(delta_r, delta_n)):
            continue
        score = float(delta_r) + float(np.dot(weight, np.asarray(delta_n, dtype=np.float64)))
        if score > best_score:
            best_score = score
            best = _point_from_index(record, idx, "subrep")
            best["selected_candidate_policy"] = candidate_id
            best["subrep_score"] = score
    return best


def _candidate_point(record: dict, candidate_id: str) -> dict | None:
    if candidate_id not in record["candidate_skill_ids"]:
        return None
    return _point_from_index(record, record["candidate_skill_ids"].index(candidate_id), candidate_id)


def _point_from_index(record: dict, idx: int, method: str) -> dict:
    return {
        "method": method,
        "context_seed": int(record["context_seed"]),
        "candidate_policy": record["candidate_skill_ids"][idx],
        "safety_cost": float(record["candidate_safety_costs"][idx]),
        "task_return": float(record["candidate_task_returns"][idx]),
    }


def _load_record(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    motives = np.asarray(data["candidate_motives"], dtype=np.float32)
    return {
        "context_seed": int(np.asarray(data["context_seed"]).item()),
        "candidate_skill_ids": [_scalar_to_string(item) for item in data["candidate_skill_ids"]],
        "candidate_payoffs": np.asarray(data["candidate_payoffs"], dtype=np.float32),
        "candidate_motives": motives,
        "candidate_safety_costs": np.asarray(data["candidate_safety_costs"], dtype=np.float32),
        "candidate_task_returns": np.asarray(data["candidate_task_returns"], dtype=np.float32),
    }


def _pareto_frontier(points: list[dict]) -> list[dict]:
    frontier = []
    for point in sorted(points, key=lambda item: (item["safety_cost"], -item["task_return"])):
        dominated = any(
            other["safety_cost"] <= point["safety_cost"]
            and other["task_return"] >= point["task_return"]
            and (
                other["safety_cost"] < point["safety_cost"]
                or other["task_return"] > point["task_return"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return frontier


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


def _std(values) -> float:
    collected = [float(value) for value in values]
    if len(collected) <= 1:
        return 0.0
    return float(np.std(collected, ddof=1))

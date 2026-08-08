"""Certification ablation utilities for Safety-Gymnasium benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate


DEFAULT_TASK_WEIGHT = (0.10, 0.90)
DEFAULT_SAFETY_WEIGHT = (0.90, 0.10)


def build_safety_gymnasium_certification_ablation(
    *,
    rollout_dirs: Sequence[str | Path],
    summary_json_path: str | Path = "demo/artifacts/safety_gymnasium_certification_ablation.json",
    pattern: str = "*.npz",
    baseline_candidate_id: str = "zero_action",
    pds_epsilon: float = 1.0,
    task_weight: Sequence[float] = DEFAULT_TASK_WEIGHT,
    safety_weight: Sequence[float] = DEFAULT_SAFETY_WEIGHT,
) -> dict:
    """Compare SubRep selection with CDS/PDS against selection without certification."""
    contexts = _load_contexts(
        rollout_dirs=rollout_dirs,
        pattern=pattern,
        baseline_candidate_id=baseline_candidate_id,
        pds_epsilon=pds_epsilon,
    )
    if not contexts:
        raise ValueError("No contexts available for certification ablation")

    summary = {
        "rollout_dirs": [str(path) for path in rollout_dirs],
        "baseline_candidate": baseline_candidate_id,
        "pds_epsilon": float(pds_epsilon),
        "total_contexts": len(contexts),
        "queries": {
            "task_focused": _summarize_query(contexts, task_weight),
            "safety_focused": _summarize_query(contexts, safety_weight),
        },
    }

    json_path = Path(summary_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_contexts(
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
        raise FileNotFoundError("No rollout files found for certification ablation")

    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=pds_epsilon)
    contexts = []
    for path in files:
        record = _load_record(path)
        ids = record["candidate_skill_ids"]
        if baseline_candidate_id not in ids:
            raise ValueError(f"{path} does not contain baseline {baseline_candidate_id!r}")

        baseline_idx = ids.index(baseline_candidate_id)
        baseline_safety_cost = float(record["candidate_safety_costs"][baseline_idx])
        calculator = ImprovementCalculator(
            {
                "baseline_payoff": float(record["candidate_payoffs"][baseline_idx]),
                "baseline_motives": record["candidate_motives"][baseline_idx],
            }
        )

        candidates = []
        for idx, candidate_policy in enumerate(ids):
            if candidate_policy == baseline_candidate_id:
                continue

            delta_r, delta_n = calculator.compute_improvements(
                float(record["candidate_payoffs"][idx]),
                record["candidate_motives"][idx],
            )
            admitted = cds_gate.admit(delta_r, delta_n) or pds_gate.admit(delta_r, delta_n)
            safety_cost = float(record["candidate_safety_costs"][idx])
            candidates.append(
                {
                    "candidate_policy": candidate_policy,
                    "admitted": bool(admitted),
                    "delta_r": float(delta_r),
                    "delta_n": tuple(float(v) for v in delta_n),
                    "safety_cost": safety_cost,
                    "task_return": float(record["candidate_task_returns"][idx]),
                    "higher_cost_than_baseline": safety_cost > baseline_safety_cost,
                }
            )

        contexts.append(
            {
                "context_seed": int(record["context_seed"]),
                "candidates": candidates,
            }
        )
    return contexts


def _summarize_query(contexts: list[dict], weight: Sequence[float]) -> dict:
    weight_array = np.asarray(weight, dtype=np.float64)
    certified_selected = []
    uncertified_selected = []
    blocked_without_cert_selection = 0
    contexts_with_certified = 0

    for context in contexts:
        candidates = context["candidates"]
        certified = [candidate for candidate in candidates if candidate["admitted"]]
        if certified:
            contexts_with_certified += 1
            certified_selected.append(_select_best(certified, weight_array))
        if candidates:
            selected_without_cert = _select_best(candidates, weight_array)
            uncertified_selected.append(selected_without_cert)
            if not selected_without_cert["admitted"]:
                blocked_without_cert_selection += 1

    certified_cost = _mean(row["safety_cost"] for row in certified_selected)
    uncertified_cost = _mean(row["safety_cost"] for row in uncertified_selected)
    certified_score = _mean(row["score"] for row in certified_selected)
    uncertified_score = _mean(row["score"] for row in uncertified_selected)

    return {
        "weight": [float(v) for v in weight],
        "contexts_evaluated": len(contexts),
        "contexts_with_certified_candidates": contexts_with_certified,
        "with_certification_mean_score": certified_score,
        "without_certification_mean_score": uncertified_score,
        "with_certification_mean_safety_cost": certified_cost,
        "without_certification_mean_safety_cost": uncertified_cost,
        "safety_cost_reduction_from_certification": uncertified_cost - certified_cost,
        "with_certification_higher_cost_selection_rate": _mean(
            row["higher_cost_than_baseline"] for row in certified_selected
        ),
        "without_certification_higher_cost_selection_rate": _mean(
            row["higher_cost_than_baseline"] for row in uncertified_selected
        ),
        "blocked_without_certification_selection_count": blocked_without_cert_selection,
        "blocked_without_certification_selection_rate": _mean(
            not row["admitted"] for row in uncertified_selected
        ),
    }


def _select_best(candidates: list[dict], weight: np.ndarray) -> dict:
    selected = max(
        candidates,
        key=lambda candidate: (
            _score_candidate(candidate, weight),
            candidate["candidate_policy"],
        ),
    )
    return {
        **selected,
        "score": _score_candidate(selected, weight),
    }


def _score_candidate(candidate: dict, weight: np.ndarray) -> float:
    return float(candidate["delta_r"]) + float(
        np.dot(weight, np.asarray(candidate["delta_n"], dtype=np.float64))
    )


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

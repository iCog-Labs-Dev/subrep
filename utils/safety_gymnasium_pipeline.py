"""Certification and reuse helpers for Safety-Gymnasium rollout artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.certificate_schema import Certificate
from certification.metta_storage import CertificateStore
from certification.pds_test import PDSGate
from library.skill_library import SkillLibrary
from library.skill_selector import select_best_skill_entry
from utils.admission_report import AdmissionReport


DEFAULT_TASK_WEIGHT = (0.10, 0.90)
DEFAULT_SAFETY_WEIGHT = (0.90, 0.10)


@dataclass(frozen=True)
class SafetyGymnasiumPipelineResult:
    """Return object for the SafeRL certification pilot."""

    stats: dict
    cert_store: CertificateStore
    library: SkillLibrary


def run_safety_gymnasium_certification_pipeline(
    *,
    rollout_dir: str | Path = "data/safety_gymnasium_rollouts",
    pattern: str = "*.npz",
    baseline_candidate_id: str = "zero_action",
    pds_epsilon: float = 1.0,
    gamma: float = 0.99,
    cert_file: str | Path = "data/safety_gymnasium_certificates.metta",
    library_file: str | Path = "data/safety_gymnasium_library.json",
    report_json_path: str | Path = "demo/artifacts/safety_gymnasium_admission_report.json",
    report_md_path: str | Path = "demo/artifacts/safety_gymnasium_admission_report.md",
    task_weight: Sequence[float] = DEFAULT_TASK_WEIGHT,
    safety_weight: Sequence[float] = DEFAULT_SAFETY_WEIGHT,
) -> SafetyGymnasiumPipelineResult:
    """Certify collected Safety-Gymnasium candidate outcomes.

    Each rollout file is expected to contain multiple candidate outcomes from
    the same reset seed. The baseline candidate is used only as the reference
    outcome; all other candidates are certified relative to it.
    """
    files = sorted(Path(rollout_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Safety-Gymnasium rollout files found in {rollout_dir!s} matching {pattern!r}"
        )

    cds_gate = CDSGate()
    pds_gate = PDSGate(epsilon=pds_epsilon)
    cert_store = CertificateStore()
    library = SkillLibrary(cert_store=cert_store, save_path=str(library_file))
    report = AdmissionReport()

    safety_rows: list[dict] = []
    context_comparisons: list[dict] = []
    contexts_processed = 0
    candidate_outcomes_loaded = 0

    for file_index, path in enumerate(files, start=1):
        record = _load_rollout_record(path)
        contexts_processed += 1
        candidate_outcomes_loaded += len(record["candidate_skill_ids"])

        candidate_ids = record["candidate_skill_ids"]
        if baseline_candidate_id not in candidate_ids:
            raise ValueError(
                f"{path} does not contain baseline candidate {baseline_candidate_id!r}"
            )

        baseline_idx = candidate_ids.index(baseline_candidate_id)
        baseline_payoff = float(record["candidate_payoffs"][baseline_idx])
        baseline_motives = np.asarray(record["candidate_motives"][baseline_idx], dtype=np.float32)
        baseline_safety_cost = float(record["candidate_safety_costs"][baseline_idx])
        calculator = ImprovementCalculator(
            {
                "baseline_payoff": baseline_payoff,
                "baseline_motives": baseline_motives,
            }
        )
        context_candidates: list[dict] = []

        for candidate_idx, candidate_id in enumerate(candidate_ids):
            if candidate_id == baseline_candidate_id:
                continue

            skill_id = _make_skill_id(file_index, record["context_seed"], candidate_id)
            payoff = float(record["candidate_payoffs"][candidate_idx])
            motives = np.asarray(record["candidate_motives"][candidate_idx], dtype=np.float32)
            delta_r, delta_n = calculator.compute_improvements(payoff, motives)

            admitted_cds = cds_gate.admit(delta_r, delta_n)
            admitted_pds = pds_gate.admit(delta_r, delta_n)
            admitted = admitted_cds or admitted_pds
            gate_type = "CDS" if admitted_cds else "PDS"
            margin = (
                cds_gate.get_admission_margin(delta_r, delta_n)
                if admitted_cds
                else pds_gate.get_admission_margin(delta_r, delta_n)
            )
            epsilon = pds_epsilon if gate_type == "PDS" else 0.0

            failure_reason = None
            if not admitted:
                worst_case_score = float(delta_r) + float(np.min(delta_n))
                failure_reason = (
                    "delta_r + min(delta_n) below CDS/PDS thresholds "
                    f"(score={worst_case_score:.4f}, PDS threshold={-pds_epsilon:.4f})"
                )
            else:
                certificate = _make_certificate(
                    skill_id=skill_id,
                    gate_type=gate_type,
                    delta_r=delta_r,
                    delta_n=delta_n,
                    margin=margin,
                    epsilon=epsilon,
                    seed=int(record["context_seed"]),
                    gamma=gamma,
                    baseline_candidate_id=baseline_candidate_id,
                    environment=record["env_id"],
                    episode_length=int(record["step_counts"][candidate_idx]),
                )
                store_added = cert_store.add(certificate)
                lib_added = False
                if store_added:
                    lib_added = library.add_skill(
                        skill_id,
                        certificate,
                        _replay_only_policy,
                    )
                if not (store_added and lib_added):
                    if store_added:
                        cert_store.remove_skill(skill_id)
                    admitted = False
                    failure_reason = "library.add_skill() rejected after math re-verification"

            episode_dict = {
                "skill_id": skill_id,
                "candidate_policy": candidate_id,
                "admitted": admitted,
                "gate_type": gate_type if admitted else None,
                "delta_r": float(delta_r),
                "delta_n": (float(delta_n[0]), float(delta_n[1])),
                "margin": float(margin),
                "epsilon": float(epsilon),
                "failure_reason": failure_reason,
            }
            report.add_from_dict(episode_dict)
            context_candidates.append(episode_dict)

            candidate_cost = float(record["candidate_safety_costs"][candidate_idx])
            safety_rows.append(
                {
                    "skill_id": skill_id,
                    "candidate_policy": candidate_id,
                    "admitted": bool(admitted),
                    "gate_type": gate_type if admitted else None,
                    "baseline_safety_cost": baseline_safety_cost,
                    "candidate_safety_cost": candidate_cost,
                    "safety_cost_delta": candidate_cost - baseline_safety_cost,
                    "task_return_delta": float(record["candidate_task_returns"][candidate_idx])
                    - float(record["candidate_task_returns"][baseline_idx]),
                }
            )

            if cert_store.count() != library.count():
                raise AssertionError(
                    "SafeRL certification invariant failed: "
                    f"cert_store.count()={cert_store.count()} != library.count()={library.count()}"
                )

        context_comparisons.append(
            _compile_context_comparison(
                context_seed=int(record["context_seed"]),
                candidates=context_candidates,
                task_weight=task_weight,
                safety_weight=safety_weight,
            )
        )

    stats = report.compile()
    stats.update(
        {
            "benchmark": "Safety-Gymnasium",
            "rollout_dir": str(rollout_dir),
            "baseline_candidate": baseline_candidate_id,
            "contexts_processed": contexts_processed,
            "candidate_outcomes_loaded": candidate_outcomes_loaded,
            "candidate_outcomes_certified": len(safety_rows),
            "pds_epsilon": float(pds_epsilon),
            "cert_store_count": cert_store.count(),
            "library_size": library.count(),
            "rejection_summary": _summarize_failure_reasons(stats.get("failure_reasons", {})),
            "safety_cost_behavior": _compile_safety_cost_behavior(safety_rows),
            "baseline_comparison": _compile_baseline_comparison(context_comparisons),
            "zero_shot_reuse": _compile_zero_shot_reuse(
                library,
                task_weight=task_weight,
                safety_weight=safety_weight,
            ),
        }
    )

    _save_pipeline_outputs(
        stats=stats,
        cert_store=cert_store,
        library=library,
        cert_file=cert_file,
        library_file=library_file,
        report_json_path=report_json_path,
        report_md_path=report_md_path,
    )

    return SafetyGymnasiumPipelineResult(stats=stats, cert_store=cert_store, library=library)


def _load_rollout_record(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    motives = np.asarray(data["candidate_motives"], dtype=np.float32)
    safety_costs = (
        np.asarray(data["candidate_safety_costs"], dtype=np.float32)
        if "candidate_safety_costs" in data
        else -motives[:, 0]
    )
    task_returns = (
        np.asarray(data["candidate_task_returns"], dtype=np.float32)
        if "candidate_task_returns" in data
        else motives[:, 1]
    )

    return {
        "env_id": _scalar_to_string(data["env_id"]) if "env_id" in data else "Safety-Gymnasium",
        "context_seed": int(np.asarray(data["context_seed"]).item()),
        "candidate_skill_ids": [_scalar_to_string(item) for item in data["candidate_skill_ids"]],
        "candidate_payoffs": np.asarray(data["candidate_payoffs"], dtype=np.float32),
        "candidate_motives": motives,
        "candidate_safety_costs": safety_costs,
        "candidate_task_returns": task_returns,
        "step_counts": (
            np.asarray(data["step_counts"], dtype=np.int32)
            if "step_counts" in data
            else np.ones(len(motives), dtype=np.int32)
        ),
    }


def _make_certificate(
    *,
    skill_id: str,
    gate_type: str,
    delta_r: float,
    delta_n: np.ndarray,
    margin: float,
    epsilon: float,
    seed: int,
    gamma: float,
    baseline_candidate_id: str,
    environment: str,
    episode_length: int,
) -> Certificate:
    return Certificate(
        skill_id=skill_id,
        gate_type=gate_type,
        delta_r=float(delta_r),
        delta_n=(float(delta_n[0]), float(delta_n[1])),
        admission_margin=float(margin),
        epsilon=float(epsilon),
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=int(seed),
        gamma=float(gamma),
        baseline_id=f"safety_rollout_baseline:{baseline_candidate_id}",
        environment=environment,
        episode_length=max(1, int(episode_length)),
        version="safety-gymnasium-0.1.0",
        weight_region_type="FULL_SIMPLEX",
    )


def _compile_safety_cost_behavior(rows: list[dict]) -> dict:
    admitted = [row for row in rows if row["admitted"]]
    rejected = [row for row in rows if not row["admitted"]]

    return {
        "mean_baseline_safety_cost": _mean(row["baseline_safety_cost"] for row in rows),
        "mean_admitted_safety_cost": _mean(row["candidate_safety_cost"] for row in admitted),
        "mean_rejected_safety_cost": _mean(row["candidate_safety_cost"] for row in rejected),
        "safety_cost_increase_count": sum(1 for row in rows if row["safety_cost_delta"] > 0.0),
        "admitted_safety_cost_increase_count": sum(
            1 for row in admitted if row["safety_cost_delta"] > 0.0
        ),
        "rejected_safety_cost_increase_count": sum(
            1 for row in rejected if row["safety_cost_delta"] > 0.0
        ),
        "examples": rows[:5],
    }


def _summarize_failure_reasons(failure_reasons: dict[str, int]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for reason, count in failure_reasons.items():
        if "below CDS/PDS thresholds" in reason:
            key = "delta_r + min(delta_n) below CDS/PDS thresholds"
        else:
            key = reason
        summary[key] = summary.get(key, 0) + int(count)
    return summary


def _compile_context_comparison(
    *,
    context_seed: int,
    candidates: list[dict],
    task_weight: Sequence[float],
    safety_weight: Sequence[float],
) -> dict:
    return {
        "context_seed": context_seed,
        "task_focused": _score_context(candidates, np.asarray(task_weight, dtype=np.float64)),
        "safety_focused": _score_context(candidates, np.asarray(safety_weight, dtype=np.float64)),
    }


def _score_context(candidates: list[dict], weight: np.ndarray) -> dict:
    scored = [
        {
            "skill_id": candidate["skill_id"],
            "candidate_policy": candidate["candidate_policy"],
            "admitted": bool(candidate["admitted"]),
            "score": _score_candidate(candidate, weight),
        }
        for candidate in candidates
    ]
    admitted = [item for item in scored if item["admitted"]]
    ppo = [item for item in scored if _is_ppo_candidate(item["candidate_policy"])]

    subrep = max(admitted, key=lambda item: (item["score"], item["skill_id"])) if admitted else None

    return {
        "subrep_selected_skill": subrep["skill_id"] if subrep else None,
        "subrep_selected_policy": subrep["candidate_policy"] if subrep else None,
        "subrep_score": subrep["score"] if subrep else None,
        "zero_action_score": 0.0,
        "random_candidate_expected_score": _mean(item["score"] for item in scored),
        "random_certified_expected_score": _mean(item["score"] for item in admitted),
        "ppo_score": _mean(item["score"] for item in ppo) if ppo else None,
        "ppo_candidate_available": bool(ppo),
        "candidate_count": len(scored),
        "certified_candidate_count": len(admitted),
    }


def _compile_baseline_comparison(context_comparisons: list[dict]) -> dict:
    return {
        "task_focused": _aggregate_comparison(
            context["task_focused"] for context in context_comparisons
        ),
        "safety_focused": _aggregate_comparison(
            context["safety_focused"] for context in context_comparisons
        ),
    }


def _aggregate_comparison(rows) -> dict:
    collected = list(rows)
    with_subrep = [row for row in collected if row["subrep_score"] is not None]
    with_ppo = [row for row in with_subrep if row["ppo_score"] is not None]

    mean_subrep = _mean(row["subrep_score"] for row in with_subrep)
    mean_zero = _mean(row["zero_action_score"] for row in with_subrep)
    mean_random = _mean(row["random_candidate_expected_score"] for row in with_subrep)
    mean_random_certified = _mean(
        row["random_certified_expected_score"] for row in with_subrep
    )
    mean_ppo = _mean(row["ppo_score"] for row in with_ppo) if with_ppo else None

    return {
        "contexts_evaluated": len(collected),
        "contexts_with_certified_candidates": len(with_subrep),
        "mean_subrep_certified_score": mean_subrep,
        "mean_zero_action_score": mean_zero,
        "mean_random_candidate_score": mean_random,
        "mean_random_certified_score": mean_random_certified,
        "mean_ppo_score": mean_ppo,
        "ppo_candidate_available": bool(with_ppo),
        "mean_lift_vs_zero_action": mean_subrep - mean_zero,
        "mean_lift_vs_random_candidate": mean_subrep - mean_random,
        "mean_lift_vs_random_certified": mean_subrep - mean_random_certified,
        "mean_lift_vs_ppo": mean_subrep - mean_ppo if mean_ppo is not None else None,
        "win_rate_vs_zero_action": _win_rate(
            row["subrep_score"] >= row["zero_action_score"] for row in with_subrep
        ),
        "win_rate_vs_random_candidate": _win_rate(
            row["subrep_score"] >= row["random_candidate_expected_score"]
            for row in with_subrep
        ),
        "avg_certified_candidates": _mean(
            row["certified_candidate_count"] for row in collected
        ),
        "avg_candidate_count": _mean(row["candidate_count"] for row in collected),
    }


def _score_candidate(candidate: dict, weight: np.ndarray) -> float:
    delta_n = np.asarray(candidate["delta_n"], dtype=np.float64)
    return float(candidate["delta_r"]) + float(np.dot(weight, delta_n))


def _is_ppo_candidate(candidate_policy: str | None) -> bool:
    if candidate_policy is None:
        return False
    normalized = candidate_policy.lower()
    return "ppo" in normalized or "pilot" in normalized


def _compile_zero_shot_reuse(
    library: SkillLibrary,
    *,
    task_weight: Sequence[float],
    safety_weight: Sequence[float],
) -> dict:
    task = _select_for_weight(library, np.asarray(task_weight, dtype=np.float64))
    safety = _select_for_weight(library, np.asarray(safety_weight, dtype=np.float64))
    return {
        "task_focused_weight": [float(v) for v in task_weight],
        "task_focused_selected_skill": task["selected_skill"],
        "task_focused_selected_score": task["selected_score"],
        "task_focused_admissible_count": task["admissible_count"],
        "safety_focused_weight": [float(v) for v in safety_weight],
        "safety_focused_selected_skill": safety["selected_skill"],
        "safety_focused_selected_score": safety["selected_score"],
        "safety_focused_admissible_count": safety["admissible_count"],
        "selection_changed": task["selected_skill"] != safety["selected_skill"],
        "retraining_performed": False,
    }


def _select_for_weight(library: SkillLibrary, weight: np.ndarray) -> dict:
    admissible = library.query_admissible(current_weight=weight)
    if not admissible:
        return {
            "selected_skill": None,
            "selected_score": None,
            "admissible_count": 0,
        }
    selected_skill, score = select_best_skill_entry(admissible, weight)
    return {
        "selected_skill": selected_skill,
        "selected_score": float(score),
        "admissible_count": len(admissible),
    }


def _save_pipeline_outputs(
    *,
    stats: dict,
    cert_store: CertificateStore,
    library: SkillLibrary,
    cert_file: str | Path,
    library_file: str | Path,
    report_json_path: str | Path,
    report_md_path: str | Path,
) -> None:
    cert_store.save_to_file(cert_file)
    library.save(str(library_file))

    json_path = Path(report_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    md_path = Path(report_md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(_render_safety_markdown(stats)) + "\n", encoding="utf-8")


def _render_safety_markdown(stats: dict) -> list[str]:
    safety = stats["safety_cost_behavior"]
    comparison = stats["baseline_comparison"]
    reuse = stats["zero_shot_reuse"]

    lines = [
        "# SubRep Safety-Gymnasium Admission Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Contexts Processed | {stats['contexts_processed']} |",
        f"| Candidate Outcomes Loaded | {stats['candidate_outcomes_loaded']} |",
        f"| Candidate Outcomes Certified | {stats['candidate_outcomes_certified']} |",
        f"| Admitted | {stats['admitted']} |",
        f"| Rejected | {stats['rejected']} |",
        f"| Admission Rate | {stats['admission_rate']:.1f}% |",
        f"| CDS Admissions | {stats['cds_pass_count']} |",
        f"| PDS Admissions | {stats['pds_pass_count']} |",
        f"| Certificate Store Count | {stats['cert_store_count']} |",
        f"| SkillLibrary Size | {stats['library_size']} |",
        "",
        "## Safety-Cost Behavior",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean Baseline Safety Cost | {safety['mean_baseline_safety_cost']:.4f} |",
        f"| Mean Admitted Safety Cost | {safety['mean_admitted_safety_cost']:.4f} |",
        f"| Mean Rejected Safety Cost | {safety['mean_rejected_safety_cost']:.4f} |",
        f"| Candidates With Higher Cost Than Baseline | {safety['safety_cost_increase_count']} |",
        f"| Admitted Higher-Cost Candidates | {safety['admitted_safety_cost_increase_count']} |",
        f"| Rejected Higher-Cost Candidates | {safety['rejected_safety_cost_increase_count']} |",
        "",
        "## Benchmark Comparison",
        "",
        "Scores use the same SubRep scalarization as reuse: `delta_r + weight dot delta_n`.",
        "Random baselines are reported as expected scores over the available candidates.",
        "",
        "| Query | SubRep Certified | Zero Action | Random Candidate | Random Certified | PPO | Lift vs Random | Win Rate vs Random |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        _comparison_row("Task-focused", comparison["task_focused"]),
        _comparison_row("Safety-focused", comparison["safety_focused"]),
        "",
    ]

    if not (
        comparison["task_focused"]["ppo_candidate_available"]
        or comparison["safety_focused"]["ppo_candidate_available"]
    ):
        lines.extend(
            [
                "_PPO baseline is marked `n/a` because this first Safety-Gymnasium "
                "rollout set contains simple continuous-control candidate policies, "
                "not a trained PPO Safety-Gymnasium policy._",
                "",
            ]
        )

    lines.extend(
        [
        "## Rejection Failure Reasons",
        "",
        ]
    )

    failure_reasons = stats.get("rejection_summary", {})
    if failure_reasons:
        lines.extend(["| Reason | Count |", "|---|---:|"])
        for reason, count in failure_reasons.items():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("_No rejections recorded._")

    example_rejected = stats.get("example_rejected_skill")
    if example_rejected:
        lines.extend(
            [
                "",
                (
                    f"Example rejected skill: `{example_rejected['skill_id']}` "
                    f"with Δr={example_rejected['delta_r']:.4f}, "
                    f"Δn={example_rejected['delta_n']}."
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Zero-Shot Reuse Result",
            "",
            "The certified library is frozen here. Only the current motive weight changes.",
            "",
            "| Query | Weight [Safety, Task] | Selected Skill | Admissible Skills | Score |",
            "|---|---|---|---:|---:|",
            (
                f"| Task-focused | {reuse['task_focused_weight']} | "
                f"{reuse['task_focused_selected_skill']} | "
                f"{reuse['task_focused_admissible_count']} | "
                f"{_format_optional_float(reuse['task_focused_selected_score'])} |"
            ),
            (
                f"| Safety-focused | {reuse['safety_focused_weight']} | "
                f"{reuse['safety_focused_selected_skill']} | "
                f"{reuse['safety_focused_admissible_count']} | "
                f"{_format_optional_float(reuse['safety_focused_selected_score'])} |"
            ),
            "",
            f"- **Selection changed under motive shift**: {reuse['selection_changed']}",
            f"- **Retraining performed**: {reuse['retraining_performed']}",
        ]
    )

    return lines


def _comparison_row(label: str, stats: dict) -> str:
    return (
        f"| {label} | "
        f"{stats['mean_subrep_certified_score']:.4f} | "
        f"{stats['mean_zero_action_score']:.4f} | "
        f"{stats['mean_random_candidate_score']:.4f} | "
        f"{stats['mean_random_certified_score']:.4f} | "
        f"{_format_optional_float(stats['mean_ppo_score'])} | "
        f"{stats['mean_lift_vs_random_candidate']:.4f} | "
        f"{stats['win_rate_vs_random_candidate']:.2f} |"
    )


def _scalar_to_string(value) -> str:
    item = value.item() if hasattr(value, "item") else value
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _make_skill_id(file_index: int, context_seed: int, candidate_id: str) -> str:
    safe_candidate = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in candidate_id)
    return f"safety_{file_index:05d}_{int(context_seed)}_{safe_candidate}"


def _mean(values) -> float:
    collected = [float(v) for v in values]
    if not collected:
        return 0.0
    return float(np.mean(collected))


def _win_rate(values) -> float:
    collected = [bool(v) for v in values]
    if not collected:
        return 0.0
    return float(np.mean(collected))


def _format_optional_float(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _replay_only_policy(_obs):
    raise RuntimeError(
        "This SafeRL skill was certified from collected rollout outcomes. "
        "Register a live environment policy before executing it."
    )

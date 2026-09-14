"""Synthetic multi-objective benchmark for SubRep's M > 2 certification path.

This is a light benchmark, not a replacement for a real 3+ objective simulator.
It verifies that the core SubRep machinery is no longer secretly two-objective:
CDS/PDS admission, SkillLibrary querying, MDN_WX support regions, motive shifts,
negative-transfer cases, and simple baselines all run for M = 3, 4, 5+.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from certification.cds_test import CDSGate
from certification.certificate_schema import Certificate
from library.skill_library import SkillLibrary
from library.skill_metadata import FULL_SIMPLEX, MDN_WX, SkillEntry
from utils.support_geometry import greedy_support_function, make_basis_query_directions


@dataclass(frozen=True)
class CandidateSkill:
    skill_id: str
    delta_r: float
    delta_n: tuple[float, ...]
    source: str


def run_multi_objective_benchmark(
    *,
    objective_counts: tuple[int, ...] = (3, 4, 5),
    candidates_per_objective_count: int = 48,
    seeds: tuple[int, ...] = (11, 23, 37),
    epsilon: float = 0.08,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> dict[str, object]:
    """Run the synthetic benchmark and optionally write JSON/Markdown reports."""
    results = [
        _run_one_setting(
            num_objectives=m,
            candidates_per_seed=candidates_per_objective_count,
            seeds=seeds,
            epsilon=epsilon,
        )
        for m in objective_counts
    ]
    summary = {
        "benchmark": "synthetic_multi_objective_subrep",
        "objective_counts": list(objective_counts),
        "seeds": list(seeds),
        "epsilon": float(epsilon),
        "results": results,
    }
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if output_markdown is not None:
        out = Path(output_markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_multi_objective_benchmark_markdown(summary), encoding="utf-8")
    return summary


def render_multi_objective_benchmark_markdown(summary: dict[str, object]) -> str:
    """Render benchmark metrics as a compact audit report."""
    lines = [
        "# SubRep Multi-Objective Benchmark",
        "",
        "## Summary",
        "",
        f"- **Benchmark**: {summary['benchmark']}",
        f"- **Objective counts**: {summary['objective_counts']}",
        f"- **Seeds**: {summary['seeds']}",
        f"- **PDS epsilon**: {summary['epsilon']}",
        "",
        "## Results",
        "",
        "| M | Candidates | Admitted | Rejected | CDS | PDS | Reuse Success | Negative Transfer | Query ms | Motive Shift Changed Selection |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in summary["results"]:
        lines.append(
            "| {num_objectives} | {candidate_skills_evaluated} | {admitted} | {rejected} | "
            "{cds_admissions} | {pds_admissions} | {reuse_success_rate:.3f} | "
            "{negative_transfer_rate:.3f} | {query_time_ms:.3f} | {selection_changed_under_motive_shift} |".format(
                **result
            )
        )

    lines += ["", "## Reuse Details", ""]
    for result in summary["results"]:
        lines += [
            f"### M={result['num_objectives']}",
            "",
            "| Motive Weight Case | Admissible | Selected Skill | Selected Score | Random Certified Score | Highest Reward Skill | Highest Reward Score Under Weight |",
            "|---|---:|---|---:|---:|---|---:|",
        ]
        for label, reuse in result["reuse"].items():
            lines.append(
                "| {label} | {admissible_count} | {selected_skill} | {selected_score} | "
                "{random_certified_expected_score} | {highest_reward_skill} | {highest_reward_score_under_weight} |".format(
                    label=label,
                    admissible_count=reuse["admissible_count"],
                    selected_skill=reuse["selected_skill"],
                    selected_score=_format_optional(reuse["selected_score"]),
                    random_certified_expected_score=_format_optional(
                        reuse["random_certified_expected_score"]
                    ),
                    highest_reward_skill=reuse["highest_reward_skill"],
                    highest_reward_score_under_weight=_format_optional(
                        reuse["highest_reward_score_under_weight"]
                    ),
                )
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _run_one_setting(
    *,
    num_objectives: int,
    candidates_per_seed: int,
    seeds: tuple[int, ...],
    epsilon: float,
) -> dict[str, object]:
    library = SkillLibrary()
    attempted = cds_count = pds_count = rejected = 0
    rejected_reasons: dict[str, int] = {}
    certification_scores: list[float] = []

    for seed in seeds:
        rng = np.random.default_rng(seed + num_objectives * 10_000)
        for candidate in _generate_candidates(rng, num_objectives, candidates_per_seed, seed=seed):
            attempted += 1
            gate_type, margin, region_type, support_values = _certify_candidate(
                candidate,
                num_objectives=num_objectives,
                epsilon=epsilon,
            )
            if gate_type is None:
                rejected += 1
                reason = "failed CDS/PDS worst-case margin"
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                continue

            cert = _make_certificate(
                candidate,
                gate_type=gate_type,
                margin=margin,
                epsilon=0.0 if gate_type == "CDS" else epsilon,
                num_objectives=num_objectives,
                region_type=region_type,
                support_values=support_values,
                seed=seed,
            )
            added = library.add_skill(
                candidate.skill_id,
                cert,
                _policy_for(candidate.skill_id),
                weight_region_type=region_type,
                certification_context=cert.certification_context,
                mdn_alpha=cert.mdn_alpha,
                wx_support_directions=cert.wx_support_directions,
                wx_support_values=cert.wx_support_values,
            )
            if not added:
                rejected += 1
                reason = "library admission rejected certificate"
                rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
                continue

            certification_scores.append(_score(candidate.delta_r, candidate.delta_n, _uniform(num_objectives)))
            cds_count += int(gate_type == "CDS")
            pds_count += int(gate_type == "PDS")

    query_weights = {
        "uniform": _uniform(num_objectives),
        "objective_0_focused": _focused(num_objectives, 0),
        f"objective_{num_objectives - 1}_focused": _focused(num_objectives, num_objectives - 1),
    }
    support_values = _runtime_support_values(num_objectives)
    support_directions = make_basis_query_directions(num_objectives)
    reuse_results = {}
    start = time.perf_counter()
    total_query_results = 0
    for label, weight in query_weights.items():
        admissible = library.query_admissible(
            current_weight=weight,
            support_directions=support_directions,
            support_values=support_values,
        )
        total_query_results += len(admissible)
        selected = _select_highest(admissible, weight)
        highest_reward = _select_highest_reward(admissible)
        reuse_results[label] = {
            "admissible_count": len(admissible),
            "selected_skill": selected.skill_id if selected else None,
            "selected_score": _entry_score(selected, weight) if selected else None,
            "random_certified_expected_score": (
                float(np.mean([_entry_score(entry, weight) for entry in admissible]))
                if admissible
                else None
            ),
            "highest_reward_skill": highest_reward.skill_id if highest_reward else None,
            "highest_reward_score_under_weight": _entry_score(highest_reward, weight) if highest_reward else None,
        }
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    shift_a = reuse_results["uniform"]["selected_skill"]
    shift_b = reuse_results[f"objective_{num_objectives - 1}_focused"]["selected_skill"]
    negative_transfer_count = sum(
        1
        for entry in library.get_admitted_skills()
        if min(_entry_score(entry, weight) for weight in query_weights.values()) < -epsilon
    )

    return {
        "num_objectives": num_objectives,
        "candidate_skills_evaluated": attempted,
        "admitted": library.count(),
        "rejected": rejected,
        "admission_rate": _safe_ratio(library.count(), attempted),
        "cds_admissions": cds_count,
        "pds_admissions": pds_count,
        "rejected_reasons": rejected_reasons,
        "reuse": reuse_results,
        "reuse_success_rate": _safe_ratio(total_query_results, len(query_weights) * max(library.count(), 1)),
        "negative_transfer_rate": _safe_ratio(negative_transfer_count, max(library.count(), 1)),
        "selection_changed_under_motive_shift": shift_a != shift_b,
        "query_time_ms": elapsed_ms,
        "mean_uniform_certification_score": float(np.mean(certification_scores)) if certification_scores else None,
        "infeasible_support_events": library.infeasible_support_events,
    }


def _generate_candidates(
    rng: np.random.Generator,
    num_objectives: int,
    count: int,
    *,
    seed: int,
) -> list[CandidateSkill]:
    candidates: list[CandidateSkill] = []
    for idx in range(count):
        if idx % 4 == 0:
            delta_n = rng.uniform(0.03, 0.6, size=num_objectives)
            delta_r = float(rng.uniform(0.05, 0.6))
            source = "safe_balanced"
        elif idx % 4 == 1:
            delta_n = rng.uniform(0.2, 0.9, size=num_objectives)
            bad_index = int(rng.integers(0, num_objectives))
            delta_n[bad_index] = -float(rng.uniform(0.15, 0.55))
            delta_r = float(rng.uniform(0.02, 0.18))
            source = "bounded_tradeoff"
        elif idx % 4 == 2:
            delta_n = rng.uniform(-1.0, 0.25, size=num_objectives)
            delta_r = float(rng.uniform(-0.3, 0.25))
            source = "unsafe_negative_transfer"
        else:
            delta_n = rng.normal(0.0, 0.35, size=num_objectives)
            delta_r = float(rng.normal(0.15, 0.35))
            source = "mixed"
        candidates.append(
            CandidateSkill(
                skill_id=f"m{num_objectives}_seed{seed}_{source}_{idx}",
                delta_r=delta_r,
                delta_n=tuple(float(v) for v in delta_n),
                source=source,
            )
        )
    return candidates


def _certify_candidate(
    candidate: CandidateSkill,
    *,
    num_objectives: int,
    epsilon: float,
) -> tuple[str | None, float, str, tuple[float, ...] | None]:
    delta_n = np.asarray(candidate.delta_n, dtype=np.float64)
    cds = CDSGate()
    cds_margin = cds.get_admission_margin(candidate.delta_r, delta_n)
    if cds.admit(candidate.delta_r, delta_n):
        return "CDS", cds_margin, FULL_SIMPLEX, None

    support_values = _candidate_support_values(delta_n)
    h_wx = greedy_support_function(-delta_n, np.asarray(support_values, dtype=np.float64))
    pds_margin = float(candidate.delta_r) - h_wx + float(epsilon)
    if pds_margin >= 0.0:
        return "PDS", pds_margin, MDN_WX, support_values
    return None, min(cds_margin, pds_margin), FULL_SIMPLEX, None


def _candidate_support_values(delta_n: np.ndarray) -> tuple[float, ...]:
    support = np.ones_like(delta_n, dtype=np.float64)
    harmful = np.where(delta_n < 0.0)[0]
    if harmful.size:
        support[harmful] = 0.08
    if support.sum() < 1.0:
        support += (1.0 - support.sum()) / support.size
    support = np.clip(support, 0.02, 1.0)
    return tuple(float(v) for v in support)


def _make_certificate(
    candidate: CandidateSkill,
    *,
    gate_type: str,
    margin: float,
    epsilon: float,
    num_objectives: int,
    region_type: str,
    support_values: tuple[float, ...] | None,
    seed: int,
) -> Certificate:
    kwargs = {}
    if region_type == MDN_WX:
        kwargs = {
            "certification_context": tuple([float(num_objectives), float(seed)]),
            "mdn_alpha": tuple([1.0] * num_objectives),
            "wx_support_directions": tuple(
                tuple(float(v) for v in row)
                for row in make_basis_query_directions(num_objectives)
            ),
            "wx_support_values": support_values,
        }
    return Certificate(
        skill_id=candidate.skill_id,
        gate_type=gate_type,
        delta_r=candidate.delta_r,
        delta_n=candidate.delta_n,
        admission_margin=margin,
        epsilon=epsilon,
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=seed,
        gamma=0.99,
        baseline_id="synthetic_idle_v1",
        environment=f"Synthetic-MO-{num_objectives}D-v0",
        episode_length=50,
        version="0.1.0",
        weight_region_type=region_type,
        **kwargs,
    )


def _runtime_support_values(num_objectives: int) -> np.ndarray:
    return np.full(num_objectives, 0.75, dtype=np.float64)


def _policy_for(skill_id: str) -> Callable:
    return lambda obs: skill_id


def _uniform(num_objectives: int) -> np.ndarray:
    return np.full(num_objectives, 1.0 / num_objectives, dtype=np.float64)


def _focused(num_objectives: int, index: int) -> np.ndarray:
    weight = np.full(num_objectives, 0.1 / max(num_objectives - 1, 1), dtype=np.float64)
    weight[index] = 0.9
    return weight


def _score(delta_r: float, delta_n: tuple[float, ...], weight: np.ndarray) -> float:
    return float(delta_r) + float(np.dot(weight, np.asarray(delta_n, dtype=np.float64)))


def _entry_score(entry: SkillEntry | None, weight: np.ndarray) -> float | None:
    if entry is None:
        return None
    return _score(entry.delta_r, entry.delta_n, weight)


def _select_highest(entries: list[SkillEntry], weight: np.ndarray) -> SkillEntry | None:
    return max(entries, key=lambda entry: _score(entry.delta_r, entry.delta_n, weight), default=None)


def _select_highest_reward(entries: list[SkillEntry]) -> SkillEntry | None:
    return max(entries, key=lambda entry: entry.delta_r, default=None)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _format_optional(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"

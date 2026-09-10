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

from utils.admission_report import AdmissionReport, AdmissionRecord, evaluate_gates
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
    objective_counts: tuple[int, ...] = (3, 4, 5, 6, 8, 10),
    candidates_per_objective_count: int = 48,
    seeds: tuple[int, ...] = (11, 23, 37, 51, 67),
    epsilon: float = 0.08,
    output_json: str | Path | None = None,
    output_markdown: str | Path | None = None,
) -> dict[str, object]:
    """Run the synthetic benchmark and optionally write JSON/Markdown reports."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide non-empty, unique seeds")
    if not objective_counts or any(m < 2 for m in objective_counts):
        raise ValueError("Objective counts must be at least two")
    if candidates_per_objective_count < 0 or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Candidate count and finite epsilon must be non-negative")
    results = [
        _aggregate_seeds(
            num_objectives=m,
            candidates_per_seed=candidates_per_objective_count,
            seeds=seeds,
            epsilon=epsilon,
        )
        for m in objective_counts
    ]
    summary = {
        "benchmark": "synthetic_multi_objective_subrep",
        "schema_version": 3,
        "evidence_type": "synthetic; no policy execution",
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
        "| M | Candidates | Admitted | Rejected | CDS | PDS | Reuse Eligibility | Abstention | Query ms | Motive Shift Changed Selection |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for result in summary["results"]:
        lines.append(
            "| {num_objectives} | {candidate_skills_evaluated} | {admitted} | {rejected} | "
            "{cds_admissions} | {pds_admissions} | {reuse_eligibility_rate:.3f} | "
            "{abstention_rate:.3f} | {query_time_ms:.3f} | {selection_changed_under_motive_shift} |".format(
                **result
            )
        )

    lines += ["", "## Method comparisons", "",
              "Independent seed libraries; random methods are exact uniform expectations.",
              "Empty choices abstain and receive the idle baseline score (0). No policies execute.",
              "| M | Scenario | Step | Method | Mean score | Seed std | Mean delta_r | Mean delta_n | Abstention |",
              "|---|---|---|---|---|---|---|---|---|"]
    for result in summary["results"]:
        for row in result["comparisons"]:
            for method, values in row["methods"].items():
                lines.append(f"| {result['num_objectives']} | {row['scenario']} | {row['step']} | {method} | {values['mean_score']:.4f} | {values['score_std_across_seeds']:.4f} | {values['mean_delta_r']:.4f} | {values['mean_delta_n']} | {values['abstention_rate']:.3f} |")
    lines += ["", "## Reuse Details", "", "Illustrative queries from the first seed only; full seed records are in JSON.", ""]
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
    from utils.admission_report import _render_audit_sections
    lines += ["", "Eligibility is not execution success. No policies were executed.",
              "Selection validity checks score >= -epsilon for the selected skill; abstentions are excluded from that denominator.",
              "Query ms is total library-filter time across the three queries, excluding ranking.",
              "Contextual caps are hand-constructed synthetic evidence, not trained MDN predictions."]
    for result in summary["results"]:
        lines += ["", f"## M={result['num_objectives']} audit", "",
                  f"Selection validity: {result['selection_validity_rate']}; threshold violations: {result['selected_threshold_violation_count']}"]
        lines += _render_audit_sections(result["admission_audit"])
    return "\n".join(lines) + "\n"


def _run_one_setting(
    *,
    num_objectives: int,
    candidates_per_seed: int,
    seeds: tuple[int, ...],
    epsilon: float,
) -> dict[str, object]:
    library = SkillLibrary()
    audit = AdmissionReport()
    attempted = cds_count = pds_count = rejected = 0
    rejected_reasons: dict[str, int] = {}
    certification_scores: list[float] = []
    all_candidates = []

    for seed in seeds:
        rng = np.random.default_rng(seed + num_objectives * 10_000)
        for candidate in _generate_candidates(rng, num_objectives, candidates_per_seed, seed=seed):
            all_candidates.append(candidate)
            attempted += 1
            gate_type, margin, region_type, support_values = _certify_candidate(
                candidate,
                num_objectives=num_objectives,
                epsilon=epsilon,
            )
            # CDS uses the full simplex; the fallback PDS test uses constructed caps.
            cds_eval = evaluate_gates(candidate.delta_r, candidate.delta_n, epsilon)[0]
            caps = None if gate_type == "CDS" else _candidate_support_values(np.asarray(candidate.delta_n))
            pds_eval = evaluate_gates(candidate.delta_r, candidate.delta_n, epsilon, support_values=caps)[1]
            audit_record = AdmissionRecord(
                skill_id=candidate.skill_id, admitted=gate_type is not None,
                gate_type=gate_type, delta_r=candidate.delta_r, delta_n=candidate.delta_n,
                margin=margin, epsilon=0.0 if gate_type == "CDS" else epsilon,
                failure_reason=None if gate_type else "Both CDS and contextual PDS failed",
                candidate_policy=candidate.source,
                weight_region_type="FULL_SIMPLEX" if caps is None else "MDN_WX",
                support_values=caps, support_feasible=True if caps is not None else None,
                gate_evaluations=(cds_eval, pds_eval),
                rejection_category=None if gate_type else "GATE_FAILED",
                baseline_id="synthetic_idle_v1", environment=f"Synthetic-MO-{num_objectives}D-v0",
                seed=seed, episode_length=50,
            )
            audit.add_record(audit_record)
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
                audit_record.admitted = False
                audit_record.gate_type = None
                audit_record.rejection_category = "LIBRARY_REVERIFICATION_FAILED"
                audit_record.failure_reason = "Library re-verification failed"
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
    elapsed_ms = 0.0
    total_query_results = 0
    for label, weight in query_weights.items():
        start = time.perf_counter()
        admissible = library.query_admissible(
            current_weight=weight,
            support_directions=support_directions,
            support_values=support_values,
        )
        elapsed_ms += (time.perf_counter() - start) * 1000.0
        total_query_results += len(admissible)
        selected = _select_highest(admissible, weight)
        highest_reward = _select_highest_reward(admissible)
        reuse_results[label] = {
            "weights": weight.tolist(),
            "support_values": support_values.tolist(),
            "admissible_count": len(admissible),
            "abstained": selected is None,
            "selection_threshold": -selected.epsilon if selected else None,
            "selection_valid": bool(_entry_score(selected, weight) >= -selected.epsilon - 1e-9) if selected else None,
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
    shift_a = reuse_results["uniform"]["selected_skill"]
    shift_b = reuse_results[f"objective_{num_objectives - 1}_focused"]["selected_skill"]
    selections = [r for r in reuse_results.values() if not r["abstained"]]
    valid = sum(r["selection_valid"] for r in selections)

    return {
        "seed": seeds[0],
        "comparisons": _comparisons(library, all_candidates, num_objectives, epsilon),
        "num_objectives": num_objectives,
        "candidate_skills_evaluated": attempted,
        "admitted": library.count(),
        "rejected": rejected,
        "admission_rate": _safe_ratio(library.count(), attempted),
        "cds_admissions": cds_count,
        "pds_admissions": pds_count,
        "rejected_reasons": rejected_reasons,
        "reuse": reuse_results,
        "rejection_rate": _safe_ratio(rejected, attempted),
        "reuse_eligibility_rate": _safe_ratio(total_query_results, len(query_weights) * library.count()),
        "selection_validity_rate": valid / len(selections) if selections else None,
        "selected_threshold_violation_count": len(selections) - valid,
        "abstention_rate": (len(query_weights) - len(selections)) / len(query_weights),
        "execution_performance": None,
        "admission_audit": audit.compile(),
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
    return None, pds_margin, MDN_WX, support_values


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
    weight = np.full(num_objectives, 0.3 / max(num_objectives - 1, 1), dtype=np.float64)
    weight[index] = 0.7
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


def _comparisons(library, candidates, m, epsilon):
    """Exact uniform-random expectations; all methods share each candidate pool."""
    balanced, focused = _uniform(m), _focused(m, 0)
    scenarios = [("balanced", 0, balanced)]
    scenarios += [(f"objective_{i}_focused", 0, _focused(m, i)) for i in range(m)]
    scenarios += [("gradual_shift", i, (1-t)*balanced+t*focused)
                  for i, t in enumerate(np.linspace(0, 1, 5))]
    scenarios += [("abrupt_shift", i, w) for i, w in enumerate(
        [focused, _focused(m, m-1), focused])]
    # Explicit rejected-only pool, not artificially hiding admissible candidates.
    rejected = [CandidateSkill("empty_case_rejected", -1.-epsilon, tuple([-1.]*m), "constructed_rejection")]
    scenarios += [("empty_admissible", 0, balanced)]
    rows = []
    for name, step, w in scenarios:
        pool = rejected if name == "empty_admissible" else candidates
        active_library = SkillLibrary() if name == "empty_admissible" else library
        eligible = active_library.query_admissible(w, np.eye(m), _runtime_support_values(m))
        methods = {
            "subrep": sorted(eligible, key=lambda c: (-_score(c.delta_r,c.delta_n,w), c.skill_id))[:1],
            "random_candidate": pool,
            "random_admissible": eligible,
            "idle": [],
            "unrestricted_best": sorted(pool, key=lambda c: (-_score(c.delta_r,c.delta_n,w), c.skill_id))[:1],
        }
        outcomes = {}
        for method, choices in methods.items():
            dr = float(np.mean([c.delta_r for c in choices])) if choices else 0.
            dn = np.mean([c.delta_n for c in choices], axis=0) if choices else np.zeros(m)
            outcomes[method] = {
                "score": _score(dr, dn, w), "delta_r": dr, "delta_n": dn.tolist(),
                "selected_skill_id": choices[0].skill_id if choices and not method.startswith("random") else None,
                "abstained": not choices and method != "idle",
                "baseline_fallback": not choices and method != "idle",
                "evaluation": "uniform_random_expectation" if method.startswith("random") else "deterministic",
                "candidate_count": len(choices),
            }
        rows.append({"scenario": name, "step": step, "weights": w.tolist(),
                     "objective_names": [f"objective_{i}" for i in range(m)],
                     "support_region_source": "hand-designed synthetic caps, not trained MDN",
                     "admissible_count": len(eligible), "methods": outcomes})
    return rows


def _aggregate_seeds(*, num_objectives, candidates_per_seed, seeds, epsilon):
    runs = [_run_one_setting(num_objectives=num_objectives,
            candidates_per_seed=candidates_per_seed, seeds=(seed,), epsilon=epsilon) for seed in seeds]
    result = dict(runs[0])
    result.pop("seed")
    result["seed_results"] = runs
    result["reuse_example_seed"] = seeds[0]
    for key in ("candidate_skills_evaluated", "admitted", "rejected", "cds_admissions", "pds_admissions",
                "selected_threshold_violation_count", "infeasible_support_events", "query_time_ms"):
        result[key] = sum(r[key] for r in runs)
    for key in ("admission_rate", "rejection_rate", "reuse_eligibility_rate", "abstention_rate",
                "selection_validity_rate", "mean_uniform_certification_score"):
        values = [r[key] for r in runs if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    audit = AdmissionReport()
    for run in runs:
        for entry in run["admission_audit"]["audit_entries"]:
            audit.add_from_dict(entry)
    result["admission_audit"] = audit.compile()
    result["rejected_reasons"] = result["admission_audit"]["failure_reasons"]
    result["selection_changed_under_motive_shift"] = any(r["selection_changed_under_motive_shift"] for r in runs)
    result["comparisons"] = []
    for idx, first in enumerate(runs[0]["comparisons"]):
        row = {k:v for k,v in first.items() if k not in ("methods", "admissible_count")}
        row["methods"] = {}
        for method in first["methods"]:
            samples = [r["comparisons"][idx]["methods"][method] for r in runs]
            row["methods"][method] = {
                "mean_score": float(np.mean([v["score"] for v in samples])),
                "score_std_across_seeds": float(np.std([v["score"] for v in samples])),
                "mean_delta_r": float(np.mean([v["delta_r"] for v in samples])),
                "mean_delta_n": np.mean([v["delta_n"] for v in samples],axis=0).tolist(),
                "abstention_rate": float(np.mean([v["abstained"] for v in samples])),
                "seed_count": len(samples),
            }
        result["comparisons"].append(row)
    return result

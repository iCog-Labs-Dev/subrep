"""
admission_report.py — Admission audit report generator for the SubRep pipeline.

Collects per-episode admission records during a pipeline run and produces
a structured JSON and human-readable Markdown report at the end.

The report is a certification audit, not just a tally: every candidate that was
evaluated gets its own entry carrying the exact inequality values the gates
compared, so a reader can reconstruct why each skill was admitted or rejected.

Usage:
    report = AdmissionReport()
    report.add_record(AdmissionRecord(...))
    report.save_json("demo/artifacts/admission_report.json")
    report.save_markdown("demo/artifacts/admission_report.md")
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from utils.support_geometry import worst_case_over_support_region


class RejectionCategory:
    """Stable rejection labels so the report can group by cause, not by message.

    Failure messages embed float values, so grouping by message text splits one
    cause into many "categories". These constants give a fixed vocabulary.
    """

    GATE_FAILED = "GATE_FAILED"
    DUPLICATE_SKILL_ID = "DUPLICATE_SKILL_ID"
    LIBRARY_REVERIFICATION_FAILED = "LIBRARY_REVERIFICATION_FAILED"
    INFEASIBLE_SUPPORT = "INFEASIBLE_SUPPORT"


@dataclass
class GateEvaluation:
    """One gate inequality with both sides exposed.

    Both the CDS and the PDS evaluation are recorded for every candidate,
    whether it was admitted or rejected, so the audit shows which gates were
    tried and how each one fared -- not only the gate that happened to admit.
    """

    gate: str          # "CDS" | "PDS"
    lhs: float
    rhs: float
    satisfied: bool
    expression: str    # e.g. "delta_r + min(delta_n) >= 0"


def evaluate_gates(
    delta_r: float,
    delta_n: Sequence[float],
    epsilon: float,
    *,
    support_values: Sequence[float] | None = None,
) -> tuple[GateEvaluation, ...]:
    """Return the CDS and PDS inequalities with their exact numeric sides.

    Both gates reduce to the same comparison and differ only in the threshold:

        CDS:  delta_r + worst_case >= 0
        PDS:  delta_r + worst_case >= -epsilon

    where ``worst_case`` is ``min(delta_n)`` over the full simplex, or
    ``min_w (w . delta_n)`` over ``W_x`` when support values are supplied. This
    mirrors ``CDSGate.admit``/``PDSGate.admit`` exactly, so the recorded numbers
    are the ones the gates actually compared rather than a re-derivation.

    Choosing the worst-case source lives here so the full-simplex and W_x
    branches cannot drift apart, and so the W_x path has a callable target that
    tests can check against ``worst_case_over_support_region`` directly.

    Args:
        delta_r: Payoff improvement over the baseline.
        delta_n: Motive improvement vector, any length >= 1.
        epsilon: PDS trade-off budget (non-negative).
        support_values: W_x support geometry. When omitted the full simplex is
            assumed.

    Returns:
        A ``(CDS, PDS)`` pair of evaluations.
    """
    motives = [float(v) for v in delta_n]
    if not motives:
        raise ValueError("delta_n must be non-empty")

    if support_values is None:
        worst_case = min(motives)
        worst_case_expression = "min(delta_n)"
    else:
        worst_case = worst_case_over_support_region(motives, support_values)
        worst_case_expression = "min_w(w . delta_n) over W_x"

    lhs = float(delta_r) + float(worst_case)
    pds_threshold = -float(epsilon)

    return (
        GateEvaluation(
            gate="CDS",
            lhs=lhs,
            rhs=0.0,
            satisfied=bool(lhs >= 0.0),
            expression=f"delta_r + {worst_case_expression} >= 0",
        ),
        GateEvaluation(
            gate="PDS",
            lhs=lhs,
            rhs=pds_threshold,
            satisfied=bool(lhs >= pds_threshold),
            expression=f"delta_r + {worst_case_expression} >= -epsilon",
        ),
    )


@dataclass
class AdmissionRecord:
    """Holds the certification result for a single pipeline episode.

    Every field after ``epsilon`` is optional with a default, so callers written
    against the original nine-field schema keep working unchanged.
    """

    skill_id: str
    admitted: bool
    gate_type: Optional[str]          # "CDS", "PDS", or None when rejected
    delta_r: float
    delta_n: tuple[float, ...]
    margin: float
    failure_reason: Optional[str]     # Populated only when admitted=False
    candidate_policy: Optional[str] = None
    epsilon: float = 0.0

    # Audit context. `gate_type` deliberately keeps its original meaning -- the
    # admitting gate, or None on rejection -- because callers and tests depend
    # on it. Which gates were evaluated is carried by `gate_evaluations`, which
    # is populated for admitted and rejected candidates alike.
    weight_region_type: str = "FULL_SIMPLEX"
    support_values: Optional[tuple[float, ...]] = None
    support_feasible: Optional[bool] = None
    gate_evaluations: tuple[GateEvaluation, ...] = field(default_factory=tuple)
    rejection_category: Optional[str] = None
    baseline_id: Optional[str] = None
    environment: Optional[str] = None
    seed: Optional[int] = None
    episode_length: Optional[int] = None


class AdmissionReport:
    """Compile and persist admission statistics for a pipeline run.

    Records are added incrementally via :meth:`add_record`.  Call
    :meth:`compile` to obtain the aggregate statistics dict, and
    :meth:`save_json` / :meth:`save_markdown` to persist the report.
    """

    def __init__(self) -> None:
        self._records: list[AdmissionRecord] = []
        self._mdn_metadata: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_record(self, record: AdmissionRecord) -> None:
        """Append one episode record to the report."""
        self._records.append(record)

    def add_from_dict(self, ep_dict: dict) -> None:
        """Convenience method: build a record from the dicts produced by
        pipeline runners and append it.

        Every audit field is read with a default, so a caller that supplies only
        the original nine keys stays fully supported and simply leaves the audit
        fields at their defaults.
        """
        support_values = ep_dict.get("support_values")
        gate_evaluations = ep_dict.get("gate_evaluations") or ()

        self.add_record(
            AdmissionRecord(
                skill_id=ep_dict["skill_id"],
                admitted=ep_dict["admitted"],
                gate_type=ep_dict.get("gate_type"),
                delta_r=ep_dict["delta_r"],
                delta_n=tuple(ep_dict["delta_n"]),
                margin=ep_dict["margin"],
                failure_reason=ep_dict.get("failure_reason"),
                candidate_policy=ep_dict.get("candidate_policy"),
                epsilon=float(ep_dict.get("epsilon", 0.0)),
                weight_region_type=ep_dict.get("weight_region_type", "FULL_SIMPLEX"),
                support_values=tuple(support_values) if support_values is not None else None,
                support_feasible=ep_dict.get("support_feasible"),
                gate_evaluations=tuple(gate_evaluations),
                rejection_category=ep_dict.get("rejection_category"),
                baseline_id=ep_dict.get("baseline_id"),
                environment=ep_dict.get("environment"),
                seed=ep_dict.get("seed"),
                episode_length=ep_dict.get("episode_length"),
            )
        )

    def set_mdn_metadata(
        self,
        source: str,
        checkpoint_path: str,
        alpha_values: list[float],
        derived_weights: list[float],
        support_values: list[float],
        support_geometry_feasible: bool,
        infeasible_support_events: int = 0,
    ) -> None:
        """Record which MDN was used and its outputs.

        Args:
            source: "trained_checkpoint" or "stub"
            checkpoint_path: Path to the MDN checkpoint file
            alpha_values: MDN alpha output (mixture weights)
            derived_weights: Mean weights derived from alpha
            support_values: MDN support output (support geometry)
            support_geometry_feasible: Whether support values satisfy constraints
            infeasible_support_events: Count of runtime steps where the library
                had to exclude MDN_WX skills because support values described an
                empty region. Expected to be 0 under SASP; any nonzero value
                signals a regression rather than a tuning problem. Defaults to 0
                so existing callers keep working.
        """
        self._mdn_metadata = {
            "mdn_source": source,
            "checkpoint_path": checkpoint_path,
            "alpha_values": alpha_values,
            "derived_weights": derived_weights,
            "support_values": support_values,
            "support_geometry_feasible": support_geometry_feasible,
            "infeasible_support_events": int(infeasible_support_events),
        }

    def compile(self) -> dict:
        """Return a dict of aggregate admission statistics."""
        total = len(self._records)
        admitted_records = [r for r in self._records if r.admitted]
        rejected_records = [r for r in self._records if not r.admitted]

        admitted = len(admitted_records)
        rejected = len(rejected_records)
        admission_rate = (admitted / total * 100.0) if total > 0 else 0.0

        cds_count = sum(1 for r in admitted_records if r.gate_type == "CDS")
        pds_count = sum(1 for r in admitted_records if r.gate_type == "PDS")

        # Collect unique failure reasons with counts
        failure_reasons: dict[str, int] = {}
        for r in rejected_records:
            reason = r.failure_reason or "unknown"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

        # Example admitted / rejected skill (first occurrence of each)
        example_admitted = asdict(admitted_records[0]) if admitted_records else None
        example_pds = next((asdict(r) for r in admitted_records if r.gate_type == "PDS"), None)
        example_rejected = asdict(rejected_records[0]) if rejected_records else None

        # Rejection categories, grouped by stable label rather than by message
        # text. Failure messages embed float values, so grouping by message
        # would split a single cause across many one-count "categories".
        rejection_categories: dict[str, int] = {}
        for r in rejected_records:
            category = r.rejection_category or "UNCATEGORIZED"
            rejection_categories[category] = rejection_categories.get(category, 0) + 1

        result = {
            "total_attempted": total,
            "admitted": admitted,
            "rejected": rejected,
            "admission_rate": round(admission_rate, 2),
            "cds_pass_count": cds_count,
            "pds_pass_count": pds_count,
            "failure_reasons": failure_reasons,
            "example_admitted_skill": example_admitted,
            "example_pds_skill": example_pds,
            "example_rejected_skill": example_rejected,
            # Full certification audit: one entry per evaluated candidate, so no
            # decision is dropped from the report the way the example_* fields
            # necessarily drop everything past the first match.
            "audit_entries": [asdict(r) for r in self._records],
            "rejection_categories": rejection_categories,
            "by_policy": _summarize(self._records, lambda r: r.candidate_policy or "unknown"),
            "by_weight_region": _summarize(self._records, lambda r: r.weight_region_type),
        }

        # Add MDN metadata if available
        if self._mdn_metadata:
            result.update(self._mdn_metadata)

        return result

    def save_json(self, path: str | Path) -> None:
        """Write the compiled report to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        stats = self.compile()
        out.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    def save_markdown(self, path: str | Path) -> None:
        """Write the compiled report to a Markdown file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        stats = self.compile()
        lines = _render_markdown(stats)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _summarize(records, key_fn) -> dict:
    """Group records by ``key_fn`` and count outcomes within each group."""
    summary: dict[str, dict[str, int]] = {}
    for record in records:
        bucket = summary.setdefault(
            str(key_fn(record)),
            {"attempted": 0, "admitted": 0, "rejected": 0, "cds": 0, "pds": 0},
        )
        bucket["attempted"] += 1
        if record.admitted:
            bucket["admitted"] += 1
            if record.gate_type == "CDS":
                bucket["cds"] += 1
            elif record.gate_type == "PDS":
                bucket["pds"] += 1
        else:
            bucket["rejected"] += 1
    return summary


def _format_vector(values) -> str:
    """Render a numeric vector at any length, so M > 2 stays readable."""
    if values is None:
        return "-"
    return "[" + ", ".join(f"{float(v):.4f}" for v in values) + "]"


def _failed_condition(entry: dict) -> str:
    """Describe the inequality that blocked admission, with both sides."""
    if entry.get("admitted"):
        return "-"

    for evaluation in entry.get("gate_evaluations") or ():
        if not evaluation.get("satisfied"):
            return (
                f"{evaluation['expression']} "
                f"({evaluation['lhs']:.4f} < {evaluation['rhs']:.4f})"
            )

    # No recorded inequality failed, so the rejection came from a later stage
    # (duplicate id, library re-verification) rather than from the gate math.
    return entry.get("rejection_category") or entry.get("failure_reason") or "unknown"


def _render_markdown(stats: dict) -> list[str]:
    """Render admission statistics as a Markdown document."""
    admitted = stats["admitted"]
    rejected = stats["rejected"]
    total = stats["total_attempted"]
    rate = stats["admission_rate"]
    rejection_rate = round(100.0 - rate, 2) if total > 0 else 0.0

    lines: list[str] = [
        "# SubRep Admission Report",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Attempted Skills | {total} |",
        f"| Admitted | {admitted} ({rate:.1f}%) |",
        f"| Rejected | {rejected} ({rejection_rate:.1f}%) |",
        f"| CDS Admissions | {stats['cds_pass_count']} |",
        f"| PDS Admissions | {stats['pds_pass_count']} |",
        "",
    ]

    # Failure reasons
    lines += ["## Rejection Failure Reasons", ""]
    failure_reasons = stats.get("failure_reasons", {})
    if failure_reasons:
        lines += ["| Reason | Count |", "|---|---|"]
        for reason, count in failure_reasons.items():
            lines.append(f"| {reason} | {count} |")
    else:
        lines.append("_No rejections recorded._")
    lines.append("")

    # Example admitted skill
    lines += ["## Example Admitted Skill", ""]
    ex_admitted = stats.get("example_admitted_skill")
    if ex_admitted:
        lines += [
            f"- **Skill ID**: `{ex_admitted['skill_id']}`",
            f"- **Candidate Policy**: {ex_admitted.get('candidate_policy') or 'unknown'}",
            f"- **Gate**: {ex_admitted['gate_type']}",
            f"- **Δr**: {ex_admitted['delta_r']:.4f}",
            f"- **Δn**: {ex_admitted['delta_n']}",
            f"- **Admission Margin**: {ex_admitted['margin']:.4f}",
        ]
        if ex_admitted["gate_type"] == "PDS":
            lines.append(f"- **PDS Epsilon**: {ex_admitted.get('epsilon', 0.0):.4f}")
    else:
        lines.append("_No skills were admitted._")
    lines.append("")

    # Example PDS skill
    lines += ["## Example PDS Trade-Off Skill", ""]
    ex_pds = stats.get("example_pds_skill")
    if ex_pds:
        lines += [
            f"- **Skill ID**: `{ex_pds['skill_id']}`",
            f"- **Candidate Policy**: {ex_pds.get('candidate_policy') or 'unknown'}",
            "- **Why PDS**: CDS failed, but the deficit stayed within the PDS epsilon budget.",
            f"- **Δr**: {ex_pds['delta_r']:.4f}",
            f"- **Δn**: {ex_pds['delta_n']}",
            f"- **PDS Margin**: {ex_pds['margin']:.4f}",
            f"- **PDS Epsilon**: {ex_pds.get('epsilon', 0.0):.4f}",
        ]
    else:
        lines.append("_No PDS-only trade-off admission recorded._")
    lines.append("")

    # Example rejected skill
    lines += ["## Example Rejected Skill", ""]
    ex_rejected = stats.get("example_rejected_skill")
    if ex_rejected:
        lines += [
            f"- **Skill ID**: `{ex_rejected['skill_id']}`",
            f"- **Candidate Policy**: {ex_rejected.get('candidate_policy') or 'unknown'}",
            f"- **Δr**: {ex_rejected['delta_r']:.4f}",
            f"- **Δn**: {ex_rejected['delta_n']}",
            f"- **Failure Reason**: {ex_rejected['failure_reason']}",
        ]
    else:
        lines.append("_No skills were rejected._")
    lines.append("")

    # MDN metadata section (if available)
    if "mdn_source" in stats:
        lines += ["## MDN Selection Metadata", ""]
        lines += [
            f"- **MDN Source**: {stats['mdn_source']}",
            f"- **Checkpoint Path**: `{stats['checkpoint_path']}`",
            f"- **Alpha Values**: {stats['alpha_values']}",
            f"- **Derived Weights**: {stats['derived_weights']}",
            f"- **Support Values**: {stats['support_values']}",
            f"- **Support Geometry Feasible**: {stats['support_geometry_feasible']}",
        ]
        # Permanent feasibility telemetry. The original support-geometry bug was
        # invisible: MDN_WX skills vanished from selection behind a log line.
        # Surfacing the count makes any recurrence loud.
        events = stats.get("infeasible_support_events")
        if events is not None:
            status = "OK" if int(events) == 0 else "REGRESSION - investigate"
            lines.append(f"- **Infeasible Support Events**: {events} ({status})")
        lines.append("")

    lines += _render_audit_sections(stats)

    return lines


def _render_audit_sections(stats: dict) -> list[str]:
    """Render the full certification audit: one row per evaluated candidate."""
    lines: list[str] = []
    entries = stats.get("audit_entries") or []

    lines += ["## Per-Candidate Audit", ""]
    if entries:
        lines += [
            "Every evaluated candidate, with the exact inequality the gates compared.",
            "",
            "| Skill | Policy | Region | Gate | Result | Δr | Δn | Margin | ε | Failed condition |",
            "|---|---|---|---|---|---:|---|---:|---:|---|",
        ]
        for entry in entries:
            lines.append(
                f"| `{entry['skill_id']}` "
                f"| {entry.get('candidate_policy') or 'unknown'} "
                f"| {entry.get('weight_region_type', 'FULL_SIMPLEX')} "
                f"| {entry.get('gate_type') or '-'} "
                f"| {'ADMITTED' if entry['admitted'] else 'REJECTED'} "
                f"| {entry['delta_r']:.4f} "
                f"| {_format_vector(entry.get('delta_n'))} "
                f"| {entry['margin']:.4f} "
                f"| {entry.get('epsilon', 0.0):.4f} "
                f"| {_failed_condition(entry)} |"
            )
        lines.append("")

        # Support geometry only concerns contextual MDN_WX decisions, so the
        # table is emitted only when such a decision is actually present.
        wx_entries = [
            e for e in entries if e.get("weight_region_type") == "MDN_WX"
        ]
        if wx_entries:
            lines += [
                "### MDN_WX Support Geometry",
                "",
                "| Skill | Support Values | Support Feasible | Gate Inequalities |",
                "|---|---|---|---|",
            ]
            for entry in wx_entries:
                inequalities = "; ".join(
                    f"{ev['gate']}: {ev['lhs']:.4f} vs {ev['rhs']:.4f} "
                    f"({'pass' if ev['satisfied'] else 'fail'})"
                    for ev in entry.get("gate_evaluations") or ()
                ) or "-"
                lines.append(
                    f"| `{entry['skill_id']}` "
                    f"| {_format_vector(entry.get('support_values'))} "
                    f"| {entry.get('support_feasible')} "
                    f"| {inequalities} |"
                )
            lines.append("")
    else:
        lines += ["_No candidates were recorded._", ""]

    lines += ["## Summary by Candidate Policy", ""]
    by_policy = stats.get("by_policy") or {}
    if by_policy:
        lines += [
            "| Policy | Attempted | Admitted | Rejected | CDS | PDS |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for policy, counts in sorted(by_policy.items()):
            lines.append(
                f"| {policy} | {counts['attempted']} | {counts['admitted']} "
                f"| {counts['rejected']} | {counts['cds']} | {counts['pds']} |"
            )
    else:
        lines.append("_No candidates were recorded._")
    lines.append("")

    lines += ["## Summary by Weight Region", ""]
    by_region = stats.get("by_weight_region") or {}
    if by_region:
        lines += [
            "| Weight Region | Attempted | Admitted | Rejected |",
            "|---|---:|---:|---:|",
        ]
        for region, counts in sorted(by_region.items()):
            lines.append(
                f"| {region} | {counts['attempted']} | {counts['admitted']} "
                f"| {counts['rejected']} |"
            )
    else:
        lines.append("_No candidates were recorded._")
    lines.append("")

    lines += ["## Rejection Categories", ""]
    categories = stats.get("rejection_categories") or {}
    if categories:
        lines += ["| Category | Count |", "|---|---:|"]
        for category, count in sorted(categories.items()):
            lines.append(f"| {category} | {count} |")
    else:
        lines.append("_No rejections recorded._")
    lines.append("")

    return lines

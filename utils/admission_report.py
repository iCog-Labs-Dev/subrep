"""
admission_report.py — Admission audit report generator for the SubRep pipeline.

Collects per-episode admission records during a pipeline run and produces
a structured JSON and human-readable Markdown report at the end.

Usage:
    report = AdmissionReport()
    report.add_record(AdmissionRecord(...))
    report.save_json("demo/artifacts/admission_report.json")
    report.save_markdown("demo/artifacts/admission_report.md")
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AdmissionRecord:
    """Holds the certification result for a single pipeline episode."""
    skill_id: str
    admitted: bool
    gate_type: Optional[str]          # "CDS", "PDS", or None when rejected
    delta_r: float
    delta_n: tuple[float, ...]
    margin: float
    failure_reason: Optional[str]     # Populated only when admitted=False


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
        ``run_full_pipeline`` and append it."""
        self.add_record(
            AdmissionRecord(
                skill_id=ep_dict["skill_id"],
                admitted=ep_dict["admitted"],
                gate_type=ep_dict.get("gate_type"),
                delta_r=ep_dict["delta_r"],
                delta_n=tuple(ep_dict["delta_n"]),
                margin=ep_dict["margin"],
                failure_reason=ep_dict.get("failure_reason"),
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
    ) -> None:
        """Record which MDN was used and its outputs.
        
        Args:
            source: "trained_checkpoint" or "stub"
            checkpoint_path: Path to the MDN checkpoint file
            alpha_values: MDN alpha output (mixture weights)
            derived_weights: Mean weights derived from alpha
            support_values: MDN support output (support geometry)
            support_geometry_feasible: Whether support values satisfy constraints
        """
        self._mdn_metadata = {
            "mdn_source": source,
            "checkpoint_path": checkpoint_path,
            "alpha_values": alpha_values,
            "derived_weights": derived_weights,
            "support_values": support_values,
            "support_geometry_feasible": support_geometry_feasible,
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
        example_rejected = asdict(rejected_records[0]) if rejected_records else None

        result = {
            "total_attempted": total,
            "admitted": admitted,
            "rejected": rejected,
            "admission_rate": round(admission_rate, 2),
            "cds_pass_count": cds_count,
            "pds_pass_count": pds_count,
            "failure_reasons": failure_reasons,
            "example_admitted_skill": example_admitted,
            "example_rejected_skill": example_rejected,
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
            f"- **Gate**: {ex_admitted['gate_type']}",
            f"- **Δr**: {ex_admitted['delta_r']:.4f}",
            f"- **Δn**: {ex_admitted['delta_n']}",
            f"- **Admission Margin**: {ex_admitted['margin']:.4f}",
        ]
    else:
        lines.append("_No skills were admitted._")
    lines.append("")

    # Example rejected skill
    lines += ["## Example Rejected Skill", ""]
    ex_rejected = stats.get("example_rejected_skill")
    if ex_rejected:
        lines += [
            f"- **Skill ID**: `{ex_rejected['skill_id']}`",
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
        lines.append("")

    return lines

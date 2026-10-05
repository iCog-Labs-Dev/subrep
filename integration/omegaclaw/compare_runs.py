"""Generate a Markdown comparison from OmegaClaw demo JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare SubRep baseline, MiniMax, and GPT OSS run reports."
    )
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Input JSON report. Repeat once per run.",
    )
    parser.add_argument("--output", required=True, help="Markdown output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reports = [_load_report(Path(path)) for path in args.report]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_markdown(reports), encoding="utf-8")
    print(f"Wrote comparison report to {output}")
    return 0


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"run_label", "backend", "evidence_label", "results", "summary"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"{path} is missing fields: {sorted(missing)}")
    if payload["evidence_label"] != "SYNTHETIC":
        raise ValueError(f"{path} must be labeled SYNTHETIC")
    return payload


def _render_markdown(reports: list[dict[str, Any]]) -> str:
    rows = []
    for report in reports:
        summary = report["summary"]
        rows.append(
            "| {label} | {backend} | {provider} | {model} | {valid:.1%} | "
            "{first_pass:.1%} | {retries:.1%} | {median:.3f} | {p95:.3f} |".format(
                label=report["run_label"],
                backend=report["backend"],
                provider=report.get("provider") or "N/A",
                model=report.get("model") or "N/A",
                valid=float(summary["valid_outcome_rate"]),
                first_pass=float(summary["first_pass_valid_rate"]),
                retries=float(summary["retry_rate"]),
                median=float(summary["median_latency_seconds"]),
                p95=float(summary["p95_latency_seconds"]),
            )
        )

    return "\n".join(
        [
            "# OmegaClaw Three-Run Comparison",
            "",
            "All scenarios use synthetic evidence. No selected skill was executed.",
            "The current contract keeps selection authority in SubRep, so this table",
            "compares response validity, retry behavior, and end-to-end latency rather",
            "than asking the models to compete as selectors.",
            "",
            "| Run | Backend | Provider | Model | Valid outcomes | First-pass valid | Retry rate | Median latency (s) | P95 latency (s) |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Interpretation",
            "",
            "- A valid outcome exactly echoes SubRep's authoritative decision.",
            "- Invalid or backend-error outcomes never replace SubRep's decision.",
            "- The synthetic baseline does not call a live model and is not a provider",
            "  latency comparison target.",
            "- Review each input JSON report for explanations, evidence references,",
            "  attempt history, and exact error details.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
